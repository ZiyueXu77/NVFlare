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


import numpy as np
import torch

"""
Verification Script for Async FedAvg Results

This script numerically computes the expected final model based on:
1. Initial model
2. Dispatch sequence (which client trains on which model version)
3. Training logic (client adds delta * client_id to weights)
4. Aggregation logic (FedAvg - simple averaging)

Usage:
    python verify_async_fedavg.py
"""


class AsyncFedAvgVerifier:
    """
    Verifies async federated averaging results by computing expected values.
    """

    def __init__(self, initial_model, delta, aggregation_threshold, max_version=10):
        """
        Args:
            initial_model: Dict of initial weights
            delta: Training delta parameter
            aggregation_threshold: M - number of responses to aggregate (from config)
            max_version: Maximum version (training stops when this is reached)
        """
        self.initial_model = self._convert_to_torch(initial_model)
        self.delta = delta
        self.M = aggregation_threshold
        self.max_version = max_version

        # Track state
        self.global_models = {0: self.initial_model.copy()}  # version -> model
        self.current_version = 0

    def _convert_to_torch(self, model_dict):
        """Convert model dict to torch tensors."""
        result = {}
        for k, v in model_dict.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.clone()
            else:
                result[k] = torch.tensor(v, dtype=torch.float32)
        return result

    def _model_to_numpy(self, model_dict):
        """Convert model dict to numpy for printing."""
        result = {}
        for k, v in model_dict.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.numpy()
            else:
                result[k] = np.array(v)
        return result

    def client_train(self, client_id, model_version):
        """
        Simulate client training.

        Training logic: result[k] = weights[k] + delta * client_id

        Args:
            client_id: Integer client ID (1-50)
            model_version: Version of global model client trains on

        Returns:
            Updated model after training
        """
        # Get the global model for this version
        global_model = self.global_models[model_version]

        # Apply training: add delta * client_id to each weight
        trained_model = {}
        for k, v in global_model.items():
            trained_model[k] = v + self.delta * client_id

        return trained_model

    def aggregate_and_update(self, client_updates):
        """
        Aggregate M client updates using FedAvg (simple average).

        Args:
            client_updates: List of (client_id, trained_model) tuples
        """
        assert len(client_updates) == self.M, f"Expected {self.M} updates, got {len(client_updates)}"

        # Average the models
        aggregated = {}
        for k in self.initial_model.keys():
            # Sum all updates for this key
            total = torch.zeros_like(self.initial_model[k], dtype=torch.float32)
            for client_id, model in client_updates:
                total += model[k]
            # Average
            aggregated[k] = total / len(client_updates)

        # Update global model
        self.current_version += 1
        self.global_models[self.current_version] = aggregated

        return self.current_version, aggregated

    def simulate_from_dispatch_sequence(self, dispatch_sequence, response_order):
        """
        Simulate the async federated learning process.

        Args:
            dispatch_sequence: List of (client_name, model_version) tuples OR
                             Dict {client_name: model_version} - which version each client trains on
            response_order: List of client_names in the order they respond

        Returns:
            Final model, version history
        """
        print("=" * 80)
        print("SIMULATING ASYNC FEDAVG")
        print("=" * 80)

        # Convert dispatch_sequence to dict for lookup
        # If a client appears multiple times, we need to track which dispatch corresponds to which response
        if isinstance(dispatch_sequence, list):
            # List of tuples: build a lookup that handles multiple dispatches per client
            dispatch_lookup = {}
            for client_name, version in dispatch_sequence:
                if client_name not in dispatch_lookup:
                    dispatch_lookup[client_name] = []
                dispatch_lookup[client_name].append(version)
        else:
            # Legacy dict format
            dispatch_lookup = {k: [v] for k, v in dispatch_sequence.items()}

        # Track which dispatch index we're processing for each client
        client_dispatch_idx = {client: 0 for client in dispatch_lookup.keys()}

        # Track responses
        response_buffer = []
        version_history = []

        # Process responses in order
        for i, client_name in enumerate(response_order):
            # CHECK: Have we reached max_version? If so, stop processing!
            # This matches the actual implementation: while current_version < max_version
            if self.current_version >= self.max_version:
                remaining_responses = len(response_order) - i
                print(f"\n[MAX VERSION REACHED] Version {self.current_version} = max_version {self.max_version}")
                print(f"  Stopping processing. {remaining_responses} remaining responses will be DISCARDED.")
                print(f"  Responses {i + 1}-{len(response_order)} from: {response_order[i:]}")
                break

            # Process this response
            # Parse client ID
            client_id = int(client_name.replace("site-", ""))

            # Get the model version for this client's current dispatch
            dispatch_idx = client_dispatch_idx.get(client_name, 0)
            versions = dispatch_lookup.get(client_name, [0])

            if dispatch_idx < len(versions):
                model_version = versions[dispatch_idx]
                client_dispatch_idx[client_name] = dispatch_idx + 1
            else:
                print(f"  WARNING: Client {client_name} responded more times than dispatched!")
                model_version = versions[-1]  # Use last known version

            print(f"\n[Response {i + 1}] Client: {client_name} (ID={client_id}), Trained on version: {model_version}")

            # Simulate training
            trained_model = self.client_train(client_id, model_version)
            response_buffer.append((client_id, trained_model))

            print(f"  Response buffer size: {len(response_buffer)}")

            # Check if we have M responses to aggregate
            if len(response_buffer) >= self.M:
                # Check if we've already reached max_version (important for edge cases)
                if self.current_version >= self.max_version:
                    print(
                        f"  [NOTE] Buffer has {len(response_buffer)} responses, but max_version already reached. Skipping aggregation."
                    )
                else:
                    # Take M responses
                    updates_to_aggregate = response_buffer[: self.M]
                    response_buffer = response_buffer[self.M :]

                    client_ids = [cid for cid, _ in updates_to_aggregate]
                    print(
                        f"  [AGGREGATION TRIGGER] Aggregating {len(updates_to_aggregate)} updates from clients: {client_ids}"
                    )

                    # Aggregate
                    new_version, new_model = self.aggregate_and_update(updates_to_aggregate)
                    version_history.append(
                        {
                            "version": new_version,
                            "aggregated_clients": client_ids,
                            "model": self._model_to_numpy(new_model),
                        }
                    )

                    print(f"  Updated to version {new_version}")
                    print(f"  New model: {self._model_to_numpy(new_model)}")

                    # Check if we just reached max_version
                    if self.current_version >= self.max_version:
                        print(
                            f"  [MAX VERSION REACHED] Version {self.current_version} = max_version {self.max_version}"
                        )
                        print("  No more aggregations will be performed.")

        # Handle remaining responses in buffer (if any)
        if response_buffer:
            print(f"\n[NOTE] {len(response_buffer)} responses remain in buffer (DISCARDED - not aggregated)")
            print("  These responses arrived but max_version was already reached.")
            print(f"  In async fedavg, training stops at max_version={self.max_version}.")
            print("  Buffered responses that don't complete aggregation are discarded.")

        final_model = self.global_models[self.current_version]
        print("\n" + "=" * 80)
        print(f"FINAL MODEL (Version {self.current_version})")
        print("  Note: Version stops at max_version, not max_version+1")
        print("=" * 80)
        for k, v in self._model_to_numpy(final_model).items():
            print(f"{k}:")
            print(v)

        return final_model, version_history

    def verify_against_actual(self, actual_model, actual_version):
        """
        Verify computed model matches actual output.

        Args:
            actual_model: The actual final model from the run
            actual_version: The actual final version
        """
        print("\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)

        expected_model = self.global_models[self.current_version]

        print(f"Expected version: {self.current_version}")
        print(f"Actual version: {actual_version}")

        if self.current_version != actual_version:
            print("❌ VERSION MISMATCH!")
            return False
        else:
            print(f"✅ Version matches: {actual_version}")

        # Convert actual model to torch
        actual_torch = self._convert_to_torch(actual_model)

        # Compare each key
        all_match = True
        for k in expected_model.keys():
            expected = expected_model[k]
            actual = actual_torch[k]

            # Check shape
            if expected.shape != actual.shape:
                print(f"❌ Shape mismatch for key '{k}': expected {expected.shape}, got {actual.shape}")
                all_match = False
                continue

            # Check values (with small tolerance for floating point)
            if torch.allclose(expected, actual, rtol=1e-5, atol=1e-5):
                print(f"✅ Key '{k}' matches")
            else:
                print(f"❌ Key '{k}' MISMATCH!")
                print(f"   Expected:\n{expected.numpy()}")
                print(f"   Actual:\n{actual.numpy()}")
                print(f"   Difference:\n{(expected - actual).numpy()}")
                all_match = False

        if all_match:
            print("\n" + "🎉" * 20)
            print("✅ ALL CHECKS PASSED! Model verification successful!")
            print("🎉" * 20)
        else:
            print("\n" + "❌" * 20)
            print("❌ VERIFICATION FAILED! See differences above.")
            print("❌" * 20)

        return all_match


def example_manual_verification():
    """
    Example: Manually verify with a known dispatch sequence.
    """
    print("\n" + "#" * 80)
    print("# EXAMPLE: Manual Verification")
    print("#" * 80)

    # Configuration
    initial_model = {
        "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }
    delta = 1.0
    aggregation_threshold = 2  # M=2
    max_version = 2  # Stop at version 2

    # Example dispatch sequence (list of tuples to allow clients to appear multiple times)
    dispatch_sequence = [
        ("site-12", 0),
        ("site-45", 0),
        ("site-10", 0),
        ("site-25", 0),
        ("site-36", 0),
        ("site-42", 1),
        ("site-27", 1),
        ("site-5", 2),
        ("site-16", 2),
        ("site-7", 3),
        ("site-1", 3),
        ("site-38", 4),
        ("site-49", 4),
        ("site-20", 5),
        ("site-36", 5),
        ("site-45", 6),
        ("site-2", 6),
        ("site-10", 7),
        ("site-41", 7),
        ("site-50", 8),
        ("site-44", 8),
        ("site-9", 9),
        ("site-28", 9),
        ("site-23", 10),
    ]

    # Order in which clients respond (critical for aggregation order!)
    # Must match the actual response order from your run
    response_order = [
        "site-12",
        "site-45",
        "site-10",
        "site-25",
        "site-36",
        "site-42",
        "site-27",
        "site-5",
        "site-16",
        "site-7",
        "site-1",
        "site-38",
        "site-49",
        "site-20",
        "site-36",
        "site-45",
        "site-2",
        "site-10",
        "site-41",
        "site-50",
        "site-44",
        "site-9",
        "site-28",
        "site-23",
    ]

    # Create verifier
    verifier = AsyncFedAvgVerifier(initial_model, delta, aggregation_threshold, max_version)

    # Simulate
    final_model, history = verifier.simulate_from_dispatch_sequence(dispatch_sequence, response_order)

    # Manual computation explanation
    print("\n" + "=" * 80)
    print("MANUAL COMPUTATION BREAKDOWN")
    print("=" * 80)

    print("\nInitial model (V0):")
    print("  x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]")

    print("\n[Step 1] Client-1 trains on V0:")
    print("  x_1 = V0_x + delta * 1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] + 1.0 * 1")
    print("      = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]")

    print("\n[Step 2] Client-2 trains on V0:")
    print("  x_2 = V0_x + delta * 2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] + 1.0 * 2")
    print("      = [[3, 4, 5], [6, 7, 8], [9, 10, 11]]")

    print("\n[Aggregation 1] Client-1 and Client-2 respond (M=2):")
    print("  V1_x = (x_1 + x_2) / 2")
    print("       = ([[2, 3, 4], [5, 6, 7], [8, 9, 10]] + [[3, 4, 5], [6, 7, 8], [9, 10, 11]]) / 2")
    print("       = [[2.5, 3.5, 4.5], [5.5, 6.5, 7.5], [8.5, 9.5, 10.5]]")

    print("\n[Step 3] Client-3 trains on V1:")
    print("  x_3 = V1_x + delta * 3 = [[2.5, 3.5, 4.5], [5.5, 6.5, 7.5], [8.5, 9.5, 10.5]] + 1.0 * 3")
    print("      = [[5.5, 6.5, 7.5], [8.5, 9.5, 10.5], [11.5, 12.5, 13.5]]")

    print("\n[Step 4] Client-4 trains on V1:")
    print("  x_4 = V1_x + delta * 4 = [[2.5, 3.5, 4.5], [5.5, 6.5, 7.5], [8.5, 9.5, 10.5]] + 1.0 * 4")
    print("      = [[6.5, 7.5, 8.5], [9.5, 10.5, 11.5], [12.5, 13.5, 14.5]]")

    print("\n[Aggregation 2] Client-3 and Client-4 respond (M=2):")
    print("  V2_x = (x_3 + x_4) / 2")
    print(
        "       = ([[5.5, 6.5, 7.5], [8.5, 9.5, 10.5], [11.5, 12.5, 13.5]] + [[6.5, 7.5, 8.5], [9.5, 10.5, 11.5], [12.5, 13.5, 14.5]]) / 2"
    )
    print("       = [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]")

    print("\n✅ Final model V2 should be:")
    print("  x = [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]")

    print("\n" + "=" * 80)
    print("IMPORTANT NOTE: Buffer Discarding")
    print("=" * 80)
    print("If max_version=2 is set, training stops once version 2 is reached.")
    print("Any responses remaining in the buffer are DISCARDED (not aggregated).")
    print("This means:")
    print("  - Version stops at max_version, NOT max_version + 1")
    print("  - Partial buffers at the end don't create another version")
    print("  - Only complete aggregations (M responses) update the model")
    print("=" * 80)


def verify_from_log(
    dispatch_sequence,
    response_order,
    initial_model,
    delta,
    aggregation_threshold,
    max_version,
    actual_final_model,
    actual_version,
):
    """
    Verify results from actual run logs.

    Args:
        dispatch_sequence: Dict from the actual run {client_name: version}
        response_order: List of client names in order they responded
        initial_model: Initial model dict
        delta: Training delta
        aggregation_threshold: M value
        max_version: Maximum version (training stops at this version)
        actual_final_model: Final model from the run
        actual_version: Final version from the run
    """
    print("\n" + "#" * 80)
    print("# VERIFICATION FROM ACTUAL RUN")
    print("#" * 80)

    verifier = AsyncFedAvgVerifier(initial_model, delta, aggregation_threshold, max_version)
    final_model, history = verifier.simulate_from_dispatch_sequence(dispatch_sequence, response_order)

    # Verify
    verifier.verify_against_actual(actual_final_model, actual_version)

    return verifier


if __name__ == "__main__":
    # Run the example
    example_manual_verification()

    print("\n" + "#" * 80)
    print("# TO USE WITH YOUR ACTUAL RUN:")
    print("#" * 80)
    print(
        """
# 1. Run your async fedavg experiment
# 2. Extract the dispatch_sequence from the logs
# 3. Extract the response_order from the logs (order of 'Received response' messages)
# 4. Call verify_from_log() with your data:

dispatch_sequence = {
    # Copy from your logs: {client_name: model_version}
}

response_order = [
    # List client names in the order they responded
]

actual_final_model = {
    # Copy the final model from your logs
}

actual_version = 10  # Final version

verify_from_log(
    dispatch_sequence=dispatch_sequence,
    response_order=response_order,
    initial_model={"x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
    delta=1.0,
    aggregation_threshold=2,
    max_version=10,  # Training stops at version 10
    actual_final_model=actual_final_model,
    actual_version=actual_version
)
    """
    )
