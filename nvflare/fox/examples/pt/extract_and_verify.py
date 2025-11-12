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


import ast
import re
import sys

from verify_async_fedavg import verify_from_log

"""
Extract Dispatch Sequence and Response Order from Logs

This script parses the async fedavg logs to extract:
1. Dispatch sequence (which client trains on which version)
2. Response order (order in which clients respond)
3. Final model and version

Then automatically verifies the results.

Usage:
    python extract_and_verify.py log_file.txt

Or run the main async fedavg with output capture:
    python pt_async_fedavg.py 2>&1 | tee async_run.log
    python extract_and_verify.py async_run.log
"""


def parse_log_file(log_file_path):
    """
    Parse log file to extract dispatch sequence, response order, and final model.

    Returns:
        tuple: (dispatch_sequence, response_order, final_model, final_version, config)
    """
    dispatch_sequence = []  # List of (client_name, version) tuples to handle multiple dispatches
    response_order = []
    final_model = None
    final_version = None
    config = {}

    with open(log_file_path, "r") as f:
        lines = f.readlines()

    # Join lines for better multi-line parsing
    full_log = "".join(lines)

    for i, line in enumerate(lines):
        # Extract configuration
        if "aggregation_threshold(M)=" in line:
            match = re.search(r"aggregation_threshold\(M\)=(\d+)", line)
            if match:
                config["aggregation_threshold"] = int(match.group(1))

        if "dispatch_threshold(N)=" in line:
            match = re.search(r"dispatch_threshold\(N\)=(\d+)", line)
            if match:
                config["dispatch_threshold"] = int(match.group(1))

        # Extract max_version from either starting line or completion line
        if "max_version=" in line and ("Async FedAvg" in line or "Starting" in line):
            match = re.search(r"max_version=(\d+)", line)
            if match:
                config["max_version"] = int(match.group(1))

        # Extract dispatch sequence
        # Pattern: [DISPATCH TRIGGER] Dispatching version X to N clients: [...]
        if "[DISPATCH TRIGGER]" in line:
            # Extract version
            version_match = re.search(r"version (\d+)", line)
            if version_match:
                version = int(version_match.group(1))

                # Extract client list
                clients_match = re.search(r"clients: \[(.*?)\]", line)
                if clients_match:
                    clients_str = clients_match.group(1)
                    # Parse client names (handles both 'site-1' and "site-1")
                    clients = re.findall(r"'([^']+)'|\"([^\"]+)\"", clients_str)
                    clients = [c[0] or c[1] for c in clients]

                    for client in clients:
                        dispatch_sequence.append((client, version))

        # Extract response order
        # Pattern: Received response from site-X
        if "Received response from" in line:
            match = re.search(r"Received response from (site-\d+)", line)
            if match:
                client = match.group(1)
                response_order.append(client)

        # Extract dispatch sequence (alternative: from final log)
        if "Dispatch sequence:" in line:
            # Parse the list of tuples
            list_match = re.search(r"Dispatch sequence: (\[.*\])", line)
            if list_match:
                try:
                    dispatch_seq_from_log = ast.literal_eval(list_match.group(1))
                    # Use the logged one as source of truth if available
                    if isinstance(dispatch_seq_from_log, list):
                        dispatch_sequence = dispatch_seq_from_log
                except:
                    pass

        # Extract final model and version
        if "Async FedAvg completed" in line and "final version=" in line:
            # Extract version
            version_match = re.search(r"final version=(\d+)", line)
            if version_match:
                final_version = int(version_match.group(1))

            # Extract model - may span multiple lines
            # Look for "final model=" and extract the dict with balanced braces
            line_start_pos = full_log.find(line)
            if line_start_pos >= 0:
                model_section = full_log[line_start_pos:]
                model_start = model_section.find("final model=")
                if model_start >= 0:
                    model_start += len("final model=")
                    # Find the matching closing brace by counting
                    brace_count = 0
                    model_end = model_start
                    in_dict = False

                    for idx in range(model_start, len(model_section)):
                        char = model_section[idx]
                        if char == "{":
                            brace_count += 1
                            in_dict = True
                        elif char == "}":
                            brace_count -= 1
                            if in_dict and brace_count == 0:
                                model_end = idx + 1
                                break

                    if model_end > model_start:
                        model_str = model_section[model_start:model_end]
                        try:
                            # Replace tensor([[...]]) with just [[...]]
                            model_str_clean = re.sub(
                                r"tensor\((.*?)\)", lambda m: m.group(1), model_str, flags=re.DOTALL
                            )
                            final_model = ast.literal_eval(model_str_clean)
                        except Exception as e:
                            print(f"  Warning: Could not parse final model: {e}")
                            print(f"  Extracted model string: {model_str[:200] if len(model_str) > 200 else model_str}")

    # Print extracted data
    print("=" * 80)
    print("EXTRACTED DATA FROM LOGS")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Max version: {config.get('max_version', 'Unknown')}")
    print(f"  Aggregation threshold (M): {config.get('aggregation_threshold', 'Unknown')}")
    print(f"  Dispatch threshold (N): {config.get('dispatch_threshold', 'Unknown')}")

    print(f"\nDispatch Sequence ({len(dispatch_sequence)} dispatches):")
    for i, (client, version) in enumerate(dispatch_sequence):
        if i < 10 or i >= len(dispatch_sequence) - 5:  # Show first 10 and last 5
            print(f"  {i + 1}. {client}: version {version}")
        elif i == 10:
            print(f"  ... ({len(dispatch_sequence) - 15} more) ...")

    print(f"\nResponse Order ({len(response_order)} responses):")
    for i, client in enumerate(response_order):
        if i < 10 or i >= len(response_order) - 5:
            print(f"  {i + 1}. {client}")
        elif i == 10:
            print(f"  ... ({len(response_order) - 15} more) ...")

    print(f"\nFinal Version: {final_version}")
    print(f"Final Model: {final_model if final_model else 'Could not extract (check logs)'}")

    return dispatch_sequence, response_order, final_model, final_version, config


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_and_verify.py <log_file>")
        print("\nExample:")
        print("  python pt_async_fedavg.py 2>&1 | tee async_run.log")
        print("  python extract_and_verify.py async_run.log")
        sys.exit(1)

    log_file = sys.argv[1]

    print(f"Parsing log file: {log_file}")
    dispatch_sequence, response_order, final_model, final_version, config = parse_log_file(log_file)

    # Check if we have enough data to verify
    if not dispatch_sequence:
        print("\n❌ ERROR: Could not extract dispatch sequence from logs!")
        print("Make sure your logs contain [DISPATCH TRIGGER] messages.")
        sys.exit(1)

    if not response_order:
        print("\n❌ ERROR: Could not extract response order from logs!")
        print("Make sure your logs contain 'Received response from' messages.")
        sys.exit(1)

    if final_model is None:
        print("\n⚠️  WARNING: Could not extract final model from logs!")
        print("You may need to manually copy it from the logs.")
        print("\nLooking for line like:")
        print("  'Async FedAvg completed - final version=X, final model={...}'")

        # Prompt user for manual input
        print("\nWould you like to manually input the final model? (y/n)")
        response = input().strip().lower()
        if response == "y":
            print("Paste the final model dict (e.g., {'x': tensor(...)}): ")
            model_str = input().strip()
            try:
                final_model = ast.literal_eval(model_str)
            except:
                print("❌ Could not parse the model. Exiting.")
                sys.exit(1)
        else:
            print("Skipping verification without final model.")
            sys.exit(0)

    # Configuration for verification
    initial_model = {
        "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }
    delta = 1.0
    aggregation_threshold = config.get("aggregation_threshold", 2)
    max_version = config.get("max_version", final_version)  # Use final_version as fallback

    print("\n\nUsing configuration:")
    print(f"  Initial model: {initial_model}")
    print(f"  Delta: {delta}")
    print(f"  Max version: {max_version}")
    print(f"  Aggregation threshold (M): {aggregation_threshold}")

    # Verify
    verify_from_log(
        dispatch_sequence=dispatch_sequence,
        response_order=response_order,
        initial_model=initial_model,
        delta=delta,
        aggregation_threshold=aggregation_threshold,
        max_version=max_version,
        actual_final_model=final_model,
        actual_version=final_version,
    )


if __name__ == "__main__":
    main()
