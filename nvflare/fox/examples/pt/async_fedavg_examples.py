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
Example Configurations for Async FedAvg with Decoupled Update and Dispatch

This module demonstrates various configuration scenarios and their behaviors.

IMPORTANT: The implementation uses non-blocking selective dispatch:
- Dispatches to ONLY N sampled clients (not all clients)
- Returns immediately (non-blocking) to allow re-dispatch when threshold reached
- This enables true asynchronous operation without waiting for all responses
"""
import logging

from nvflare.fox.api.app import ServerApp
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.pt.pt_async_fedavg import AsyncPTFedAvg, AsyncPTTrainer
from nvflare.fox.sim.simulator import Simulator


def example_eager_updates_controlled_dispatch():
    """
    Scenario A: Eager Updates + Controlled Dispatch

    Config:
    - M=1: Aggregate immediately on each response
    - N=3: Wait for 3 slots before dispatching
    - Max 5 concurrent clients

    Behavior:
    - Very fresh global model (updates frequently)
    - Batched dispatch for better efficiency
    - Good when model freshness is critical
    """
    print("\n" + "=" * 70)
    print("SCENARIO A: Eager Updates + Controlled Dispatch")
    print("=" * 70)

    simple_logging(logging.INFO)

    server_app = ServerApp(
        strategy_name="async_fed_avg",
        strategy=AsyncPTFedAvg(
            initial_model={"layer": [[1, 2], [3, 4]]},
            max_version=8,
            aggregation_threshold=1,  # M=1: Update on every response
            dispatch_threshold=3,  # N=3: Dispatch in batches of 3
            max_concurrent_clients=5,
        ),
    )

    client_app = AsyncPTTrainer(delta=0.5)

    simulator = Simulator(
        root_dir="/tmp/nvflare/async_scenario_a",
        experiment_name="eager_updates",
        server_app=server_app,
        client_app=client_app,
        num_clients=6,
    )

    simulator.run()


def example_batch_updates_eager_dispatch():
    """
    Scenario B: Batch Updates + Eager Dispatch

    Config:
    - M=4: Aggregate every 4 responses
    - N=1: Dispatch immediately when any slot opens
    - Max 3 concurrent clients

    Behavior:
    - More stable aggregation (uses more updates)
    - Maximum client utilization (no idle clients)
    - Good for high-throughput scenarios
    """
    print("\n" + "=" * 70)
    print("SCENARIO B: Batch Updates + Eager Dispatch")
    print("=" * 70)

    simple_logging(logging.INFO)

    server_app = ServerApp(
        strategy_name="async_fed_avg",
        strategy=AsyncPTFedAvg(
            initial_model={"layer": [[1, 2], [3, 4]]},
            max_version=8,
            aggregation_threshold=4,  # M=4: Batch aggregation
            dispatch_threshold=1,  # N=1: Eager dispatch
            max_concurrent_clients=3,
        ),
    )

    client_app = AsyncPTTrainer(delta=0.5)

    simulator = Simulator(
        root_dir="/tmp/nvflare/async_scenario_b",
        experiment_name="batch_updates",
        server_app=server_app,
        client_app=client_app,
        num_clients=6,
    )

    simulator.run()


def example_balanced():
    """
    Scenario C: Balanced Configuration

    Config:
    - M=2: Aggregate every 2 responses
    - N=2: Dispatch when 2 slots available
    - Max 4 concurrent clients

    Behavior:
    - Balanced between freshness and stability
    - Reasonable client utilization
    - Good general-purpose configuration
    """
    print("\n" + "=" * 70)
    print("SCENARIO C: Balanced Configuration")
    print("=" * 70)

    simple_logging(logging.INFO)

    server_app = ServerApp(
        strategy_name="async_fed_avg",
        strategy=AsyncPTFedAvg(
            initial_model={"layer": [[1, 2], [3, 4]]},
            max_version=10,
            aggregation_threshold=2,  # M=2: Moderate aggregation
            dispatch_threshold=2,  # N=2: Moderate dispatch
            max_concurrent_clients=4,
        ),
    )

    client_app = AsyncPTTrainer(delta=0.5)

    simulator = Simulator(
        root_dir="/tmp/nvflare/async_scenario_c",
        experiment_name="balanced",
        server_app=server_app,
        client_app=client_app,
        num_clients=6,
    )

    simulator.run()


def example_high_concurrency():
    """
    Scenario D: High Concurrency

    Config:
    - M=3: Moderate aggregation
    - N=5: Large batch dispatch
    - Max 10 concurrent clients (high parallelism)

    Behavior:
    - Many clients training simultaneously
    - Good for systems with high compute capacity
    - Maximizes parallelism
    """
    print("\n" + "=" * 70)
    print("SCENARIO D: High Concurrency")
    print("=" * 70)

    simple_logging(logging.INFO)

    server_app = ServerApp(
        strategy_name="async_fed_avg",
        strategy=AsyncPTFedAvg(
            initial_model={"layer": [[1, 2], [3, 4]]},
            max_version=10,
            aggregation_threshold=3,  # M=3
            dispatch_threshold=5,  # N=5: Large batches
            max_concurrent_clients=10,  # High parallelism
        ),
    )

    client_app = AsyncPTTrainer(delta=0.5)

    simulator = Simulator(
        root_dir="/tmp/nvflare/async_scenario_d",
        experiment_name="high_concurrency",
        server_app=server_app,
        client_app=client_app,
        num_clients=15,  # Many clients
    )

    simulator.run()


def example_low_latency():
    """
    Scenario E: Low Latency (Minimal Buffering)

    Config:
    - M=1: Update immediately
    - N=1: Dispatch immediately
    - Max 2 concurrent clients

    Behavior:
    - Minimal latency (no buffering)
    - Fastest response to changes
    - Good for real-time scenarios
    """
    print("\n" + "=" * 70)
    print("SCENARIO E: Low Latency (Minimal Buffering)")
    print("=" * 70)

    simple_logging(logging.INFO)

    server_app = ServerApp(
        strategy_name="async_fed_avg",
        strategy=AsyncPTFedAvg(
            initial_model={"layer": [[1, 2], [3, 4]]},
            max_version=10,
            aggregation_threshold=1,  # M=1: Immediate update
            dispatch_threshold=1,  # N=1: Immediate dispatch
            max_concurrent_clients=2,
        ),
    )

    client_app = AsyncPTTrainer(delta=0.5)

    simulator = Simulator(
        root_dir="/tmp/nvflare/async_scenario_e",
        experiment_name="low_latency",
        server_app=server_app,
        client_app=client_app,
        num_clients=4,
    )

    simulator.run()


def print_scenarios_summary():
    """Print a summary of all scenarios."""
    print("\n" + "=" * 70)
    print("ASYNC FEDAVG CONFIGURATION SCENARIOS")
    print("=" * 70)
    print(
        """
Scenario A: Eager Updates + Controlled Dispatch
    M=1, N=3, Max=5 clients
    → Very fresh models, batched dispatch

Scenario B: Batch Updates + Eager Dispatch
    M=4, N=1, Max=3 clients
    → Stable aggregation, maximum utilization

Scenario C: Balanced Configuration
    M=2, N=2, Max=4 clients
    → General purpose, balanced approach

Scenario D: High Concurrency
    M=3, N=5, Max=10 clients
    → High parallelism, many clients

Scenario E: Low Latency
    M=1, N=1, Max=2 clients
    → Minimal buffering, real-time response

Key Parameters:
    M = aggregation_threshold (responses to trigger model update)
    N = dispatch_threshold (slots to trigger task dispatch)
    Max = max_concurrent_clients (training parallelism limit)
    """
    )
    print("=" * 70)


if __name__ == "__main__":
    print_scenarios_summary()

    # Run one scenario at a time by uncommenting:

    # example_eager_updates_controlled_dispatch()  # Scenario A
    # example_batch_updates_eager_dispatch()  # Scenario B
    example_balanced()  # Scenario C (default)
    # example_high_concurrency()  # Scenario D
    # example_low_latency()  # Scenario E
