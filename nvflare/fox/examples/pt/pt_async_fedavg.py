# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Asynchronous Federated Averaging with Decoupled Update and Dispatch

This module provides an async FedAvg strategy with DECOUPLED operations:

## Key Innovation: Separation of Concerns

Traditional async FL couples model update with task dispatch. This implementation
DECOUPLES them into two independent operations:

### 1. Global Model Update (Aggregation Trigger)
- Triggered when M responses are buffered (aggregation_threshold)
- Takes M responses from buffer and aggregates using FedAvg
- Updates global model and increments version
- Independent of client availability

### 2. Task Dispatch (Availability Trigger)
- Triggered when N concurrent slots become available (dispatch_threshold)
- Samples N clients from available pool
- Dispatches current global model (whatever version it is)
- Independent of buffer state

## Benefits of Decoupling

1. **Flexible Aggregation**: Can aggregate frequently (small M) or accumulate more updates (large M)
2. **Optimal Resource Utilization**: Dispatch immediately when slots open, don't wait for aggregation
3. **Reduced Staleness**: Quick dispatch means clients get newer models faster
4. **Better Throughput**: No artificial synchronization between update and dispatch

## Example Scenarios

Scenario A: Fast updates, controlled dispatch
- M=1 (aggregate each response immediately)
- N=3 (only dispatch when 3 slots available)
- Result: Very fresh models, but batch dispatching

Scenario B: Batch updates, eager dispatch  
- M=5 (aggregate 5 responses at a time)
- N=1 (dispatch as soon as any slot opens)
- Result: More stable aggregation, maximum client utilization

Scenario C: Balanced
- M=2, N=2
- Result: Balanced between freshness and stability

## Parameters

- aggregation_threshold (M): Number of responses needed to trigger model update
- dispatch_threshold (N): Number of available slots needed to trigger dispatch
- max_concurrent_clients: Maximum number of clients training simultaneously
- max_version: Stop condition (version-based instead of round-based)

## Architecture

```
Client Pool:
┌─────────────┐
│ Available   │ ──> When N slots free ──> Dispatch Trigger
│ Busy        │                            (independent)
└─────────────┘
       ↓
  Response Queue
┌─────────────┐
│ Buffer      │ ──> When M responses  ──> Aggregation Trigger
└─────────────┘                            (independent)
       ↓
Global Model (version V)
```

## Critical Implementation Detail: Non-Blocking Selective Dispatch

The dispatch mechanism uses `group()` with `blocking=False` to dispatch ONLY to
selected/sampled clients (not all clients). This is crucial because:

1. **Selective Dispatch**: Only N sampled clients receive tasks (not all clients)
2. **Non-Blocking**: Returns immediately without waiting for responses
3. **Re-entrant**: Can dispatch again when threshold is reached
4. **No Starvation**: Other available clients can be dispatched to independently

Without this approach:
- `all_clients()` would dispatch to everyone (defeating client sampling)
- `blocking=True` would wait for all responses (defeating concurrent dispatch)
"""
import logging
import random
import threading
import time

import torch

from nvflare.fox.api.app import ClientApp, ServerApp
from nvflare.fox.api.constants import ContextKey, EnvType
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.dec import collab
from nvflare.fox.api.group import group
from nvflare.fox.api.strategy import Strategy
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.pt.utils import parse_state_dict
from nvflare.fox.sim.simulator import Simulator
from nvflare.fox.sys.tensor_downloader import download_state_dict, prepare_for_download
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:
    """Helper class for accumulating aggregation results."""

    def __init__(self):
        self.total = {}
        self.count = 0


class AsyncPTFedAvg(Strategy):
    """
    Asynchronous Federated Averaging Strategy with Decoupled Update and Dispatch.

    Features:
    - No traditional rounds: long-running server tracks global model versions
    - Buffered aggregation: collects responses as they arrive (non-blocking)
    - DECOUPLED OPERATIONS:
      * Global model update: triggered when M responses buffered (aggregation_threshold)
      * Task dispatch: triggered when N slots available (dispatch_threshold)
    - Continues until max_version is reached
    - Dynamic client sampling from available client pool

    Args:
        initial_model: Initial model weights (dict of tensors or lists)
        max_version: Maximum version number to reach before stopping
        aggregation_threshold: Number of responses (M) needed to trigger model aggregation/update
        dispatch_threshold: Number of available slots (N) needed to trigger task dispatch
        max_concurrent_clients: Maximum number of clients that can train concurrently
        timeout: Communication timeout in seconds
    """

    def __init__(
        self,
        initial_model,
        max_version=10,
        aggregation_threshold=2,
        dispatch_threshold=2,
        max_concurrent_clients=None,
        timeout=2.0,
    ):
        self.initial_model = initial_model
        self.max_version = max_version
        self.aggregation_threshold = aggregation_threshold  # M: responses to aggregate
        self.dispatch_threshold = dispatch_threshold  # N: available slots to dispatch
        self.max_concurrent_clients = max_concurrent_clients  # Max concurrent training slots
        self.timeout = timeout
        self.name = "AsyncPTFedAvg"
        self.logger = get_obj_logger(self)
        self._init_model = parse_state_dict(initial_model)

        # State tracking
        self.current_version = 0
        self.global_model = None
        self.lock = threading.Lock()
        self.response_buffer = []  # Buffer for client responses
        self.available_clients = set()  # Pool of available clients
        self.busy_clients = set()  # Clients currently training
        self.total_clients = 0  # Total number of clients

    def execute(self, context: Context):
        """
        Main execution loop for async federated averaging with DECOUPLED operations.

        Two independent triggers:
        1. Model update: when buffer has M responses -> aggregate and update global model
        2. Task dispatch: when N slots available -> dispatch tasks to N clients

        Runs continuously until max_version is reached.
        """
        self.logger.info(
            f"[{context.header_str()}] Starting Async FedAvg (Decoupled Mode) - "
            f"max_version={self.max_version}, "
            f"aggregation_threshold(M)={self.aggregation_threshold}, "
            f"dispatch_threshold(N)={self.dispatch_threshold}, "
            f"max_concurrent_clients={self.max_concurrent_clients}"
        )

        # Initialize global model
        self.global_model = context.get_prop(ContextKey.INPUT, self._init_model)

        # Get all client names from client proxies
        all_client_proxies = context.clients
        all_client_names = [p.name for p in all_client_proxies]

        self.total_clients = len(all_client_names)
        self.available_clients = set(all_client_names)

        # Record the dispatch sequence: {client_name: model_version}
        # Use list of tuples to allow client to appear multiple times in the sequence
        self.dispatch_sequence = []

        # Set max concurrent if not specified
        if self.max_concurrent_clients is None:
            self.max_concurrent_clients = self.total_clients

        self.logger.info(
            f"[{context.header_str()}] Total clients: {self.total_clients}, "
            f"Max concurrent: {self.max_concurrent_clients}, "
            f"Available: {sorted(list(self.available_clients))}"
        )

        # Initial dispatch - fill up to max_concurrent_clients
        if self.max_concurrent_clients > self.total_clients:
            self.max_concurrent_clients = self.total_clients
            self.logger.warning(
                f"[{context.header_str()}] Max concurrent clients is greater than total clients, setting to {self.total_clients}"
            )

        initial_dispatch = self.max_concurrent_clients
        self.logger.info(f"[{context.header_str()}] Initial dispatch: {initial_dispatch}")
        self._dispatch_to_clients(context, num_clients=initial_dispatch)

        # Main loop: continue until max_version reached
        # NOTE: Loop exits when current_version reaches max_version
        # Any responses remaining in buffer at that point are DISCARDED (not aggregated)
        # This ensures final version = max_version, not max_version + 1
        while self.current_version < self.max_version:
            # Check context abort first
            if context.is_aborted():
                self.logger.info(f"[{context.header_str()}] Context aborted, stopping")
                break

            # DECOUPLED CHECKS:
            # Check 1: Do we have M responses to aggregate?
            should_aggregate = False
            should_dispatch = False

            with self.lock:
                buffer_size = len(self.response_buffer)
                num_available = len(self.available_clients)
                num_busy = len(self.busy_clients)

            if buffer_size >= self.aggregation_threshold:
                should_aggregate = True

            # Check 2: Do we have N available slots to dispatch?
            # Only dispatch if we have space (not at max concurrent limit)
            if num_available >= self.dispatch_threshold and num_busy < self.max_concurrent_clients:
                should_dispatch = True

            # Execute operations independently
            if should_aggregate:
                self._aggregate_and_update(context)

            if should_dispatch:
                # Dispatch up to N clients, respecting max_concurrent limit
                num_to_dispatch = min(self.dispatch_threshold, self.max_concurrent_clients - num_busy, num_available)
                if num_to_dispatch > 0:
                    self._dispatch_to_clients(context, num_clients=num_to_dispatch)

            # If neither condition met, wait a bit to avoid busy waiting
            if not should_aggregate and not should_dispatch:
                time.sleep(0.1)

        # Log final state including any discarded responses
        with self.lock:
            remaining_buffer = len(self.response_buffer)

        self.logger.info(
            f"[{context.header_str()}] Async FedAvg completed - "
            f"final version={self.current_version}, final model={self.global_model}"
        )

        if remaining_buffer > 0:
            self.logger.info(
                f"[{context.header_str()}] NOTE: {remaining_buffer} responses remain in buffer "
                f"(DISCARDED - max_version {self.max_version} reached)"
            )

        self.logger.info(f"[{context.header_str()}] Dispatch sequence: {self.dispatch_sequence}")

        return self.global_model

    def _dispatch_to_clients(self, context: Context, num_clients: int):
        """
        Sample available clients and dispatch training tasks.

        This operation is INDEPENDENT of model aggregation - dispatches
        current global model when N slots become available.

        CRITICAL: Uses non-blocking dispatch to ONLY selected clients,
        allowing immediate return so we can dispatch again when threshold is reached.

        Args:
            context: Execution context
            num_clients: Number of clients to sample and dispatch to
        """
        with self.lock:
            # Sample from available clients
            num_to_sample = min(num_clients, len(self.available_clients))
            if num_to_sample == 0:
                self.logger.warning(f"[{context.header_str()}] No available clients to dispatch")
                return

            sampled_clients = random.sample(list(self.available_clients), num_to_sample)

            # Move sampled clients to busy pool
            for client in sampled_clients:
                self.available_clients.remove(client)
                self.busy_clients.add(client)

            current_model = self.global_model
            current_version = self.current_version
            num_available_after = len(self.available_clients)
            num_busy_after = len(self.busy_clients)

        self.logger.info(
            f"[{context.header_str()}] [DISPATCH TRIGGER] "
            f"Dispatching version {current_version} to {num_to_sample} clients: {sampled_clients} "
            f"(Available: {num_available_after}, Busy: {num_busy_after})"
        )

        # Prepare model for dispatch (streaming if in SYSTEM mode)
        if context.env_type == EnvType.SYSTEM:
            model = prepare_for_download(
                state_dict=current_model,
                ctx=context,
                timeout=5.0,
                num_tensors_per_chunk=2,
            )
            model_type = "ref"
            self.logger.info(f"[{context.header_str()}] Prepared model as ref: {model}")
        else:
            model = current_model
            model_type = "model"

        # Get proxies for ONLY the sampled clients
        all_client_proxies = context.clients
        sampled_client_proxies = [p for p in all_client_proxies if p.name in sampled_clients]

        if len(sampled_client_proxies) != len(sampled_clients):
            self.logger.error(
                f"[{context.header_str()}] Could not find all sampled clients! "
                f"Expected {len(sampled_clients)}, found {len(sampled_client_proxies)}"
            )

        # Record the dispatch sequence
        # Add client name and model version tuple to the sequence list
        for client in sampled_clients:
            self.dispatch_sequence.append((client, current_version))

        # Dispatch to ONLY sampled clients with NON-BLOCKING mode
        # This allows us to return immediately and dispatch again when threshold is reached
        group(
            context,
            sampled_client_proxies,
            blocking=False,  # NON-BLOCKING: returns immediately
            process_resp_cb=self._handle_client_response,
        ).train(current_version, model, model_type)

    def _handle_client_response(self, result, context: Context):
        """
        Callback to handle client responses - add to buffer and update client pools.

        Args:
            result: Training result from client
            context: Response context with caller information
        """
        client_name = context.caller

        with self.lock:
            buffer_size_before = len(self.response_buffer)

            # Move client back to available pool
            if client_name in self.busy_clients:
                self.busy_clients.remove(client_name)
                self.available_clients.add(client_name)

            # Add response to buffer
            self.response_buffer.append((client_name, result, context))
            buffer_size_after = len(self.response_buffer)

        self.logger.info(
            f"[{context.header_str()}] [RESPONSE] Received response from {client_name} "
            f"(Buffer: {buffer_size_before} -> {buffer_size_after})"
        )

        return None

    def _aggregate_and_update(self, context: Context):
        """
        Aggregate buffered responses and update global model.

        Takes M (aggregation_threshold) responses from buffer, aggregates them
        using FedAvg (simple average), and updates the global model version.

        This operation is INDEPENDENT of task dispatch.

        Args:
            context: Execution context
        """
        with self.lock:
            if len(self.response_buffer) == 0:
                return

            # Take M responses from buffer for aggregation
            responses_to_aggregate = self.response_buffer[: self.aggregation_threshold]
            self.response_buffer = self.response_buffer[self.aggregation_threshold :]

            self.logger.info(
                f"[{context.header_str()}] [AGGREGATION TRIGGER] "
                f"Aggregating {len(responses_to_aggregate)} responses (M={self.aggregation_threshold}) "
                f"for version {self.current_version}, buffer remaining: {len(self.response_buffer)}"
            )

        # Perform aggregation
        aggr_result = _AggrResult()
        aggregated_clients = []
        for client_name, result, resp_context in responses_to_aggregate:
            self._accept_train_result(result, aggr_result, resp_context)
            aggregated_clients.append(client_name)

        if aggr_result.count == 0:
            self.logger.warning(f"[{context.header_str()}] No valid results to aggregate")
            return

        # Compute average
        averaged_model = {}
        for k, v in aggr_result.total.items():
            averaged_model[k] = torch.div(v, aggr_result.count)

        with self.lock:
            self.global_model = averaged_model
            self.current_version += 1

        self.logger.info(
            f"[{context.header_str()}] [MODEL UPDATE] Updated global model to version {self.current_version} "
            f"from {aggr_result.count} clients: {aggregated_clients}"
        )
        self.logger.debug(f"[{context.header_str()}] Version {self.current_version} model: {averaged_model}")

    def _accept_train_result(self, result, aggr_result: _AggrResult, context: Context):
        """
        Process a single training result for aggregation.

        Handles both direct model transfer and streaming (ref-based) transfer.

        Args:
            result: Training result (model, model_type) tuple
            aggr_result: Accumulator for aggregation
            context: Response context
        """
        model, model_type = result

        if model_type == "ref":
            # Download model via streaming
            err, model = download_state_dict(
                ref=model,
                per_request_timeout=5.0,
                ctx=context,
                tensors_received_cb=self._aggregate_tensors,
                aggr_result=aggr_result,
                context=context,
            )
            if err:
                self.logger.error(f"[{context.header_str()}] Failed to download model: {err}")
                return
        else:
            # Direct model aggregation
            for k, v in model.items():
                if k not in aggr_result.total:
                    aggr_result.total[k] = v
                else:
                    aggr_result.total[k] += v

        aggr_result.count += 1

    def _aggregate_tensors(self, td: dict[str, torch.Tensor], aggr_result: _AggrResult, context: Context):
        """
        Aggregate tensors as they arrive (for streaming).

        Args:
            td: Tensor dictionary received
            aggr_result: Accumulator for aggregation
            context: Response context
        """
        self.logger.info(f"[{context.header_str()}] Aggregating received tensor: {td}")
        for k, v in td.items():
            if k not in aggr_result.total:
                aggr_result.total[k] = v
            else:
                aggr_result.total[k] += v


class AsyncPTTrainer(ClientApp):
    """
    Simple PyTorch trainer for async federated learning demo.

    Adds a delta to each weight to simulate training.
    """

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    @collab
    def train(self, current_version, weights, model_type: str, context: Context):
        """
        Simulate training by adding delta to weights.

        Args:
            current_version: Model version number
            weights: Model weights (or reference to download)
            model_type: "model" for direct transfer, "ref" for streaming
            context: Training context

        Returns:
            Tuple of (model, model_type)
        """
        if context.is_aborted():
            self.logger.debug("training aborted")
            return None

        self.logger.info(f"[{context.header_str()}] Training on version {current_version}: " f"{model_type=}")

        # Download model if it's a reference
        if model_type == "ref":
            err, model = download_state_dict(ref=weights, per_request_timeout=5.0, ctx=context)
            if err:
                raise RuntimeError(f"failed to download model {weights}: {err}")
            self.logger.info(f"Downloaded model {model}")
            weights = model

        # Get client name
        client_name = context.callee
        self.logger.info(f"[{context.header_str()}] Client: {client_name} Started Training")
        # remove "site-" from client name
        client_name = client_name.replace("site-", "")
        # convert client name to int
        client_name = int(client_name)
        # sleep for 1 second
        time.sleep(1)

        # Simulate training: add delta to each weight
        result = {}
        for k, v in weights.items():
            result[k] = v + self.delta * client_name

        # Prepare result for return (streaming if received as ref)
        if model_type == "ref":
            model = prepare_for_download(
                state_dict=result,
                ctx=context,
                timeout=5.0,
                num_tensors_per_chunk=2,
            )
            model_type = "ref"
            self.logger.info(f"Prepared result as ref: {model}")
        else:
            model = result
            model_type = "model"

        return model, model_type


def main():
    """
    Demo for asynchronous FedAvg with DECOUPLED aggregation and dispatch.

    This demo shows:
    - M=2: Aggregate global model when 2 responses buffered
    - N=2: Dispatch to 2 clients when 2 slots available
    - Max 4 concurrent clients training at any time

    These two operations are independent!
    """
    simple_logging(logging.INFO)

    server_app = ServerApp(
        strategy_name="async_fed_avg",
        strategy=AsyncPTFedAvg(
            initial_model={
                "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
            max_version=10,  # Run until version 10
            aggregation_threshold=2,  # M: Aggregate when 2 responses received
            dispatch_threshold=1,  # N: Dispatch immediately whenever a response is received (1-slot available)
            max_concurrent_clients=4,  # Max 4 clients training concurrently
        ),
    )

    client_app = AsyncPTTrainer(delta=1.0)

    simulator = Simulator(
        root_dir="/tmp/nvflare/fox_sim_async",
        experiment_name="pt_async_fedavg_decoupled",
        server_app=server_app,
        client_app=client_app,
        num_clients=50,  # Use 5 clients for async demo
    )

    simulator.run()


if __name__ == "__main__":
    main()
