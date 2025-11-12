#!/usr/bin/env python3
"""
Analyze timing data from TensorBoard logs for FedAvg job.

This script reads TensorBoard event files and generates timing statistics
to help identify performance bottlenecks compared to pure PyTorch simulation.
"""

import os
import sys
from collections import defaultdict

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard package not found")
    print("Install with: pip install tensorboard")
    sys.exit(1)


def load_timing_data(log_dir):
    """Load timing data from TensorBoard event files."""
    print(f"Loading timing data from: {log_dir}")

    # Find event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))

    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None, None, None

    print(f"Found {len(event_files)} event file(s)")

    # Load all scalars - store with index for synchronization
    timing_data = defaultdict(list)
    timestamp_data = defaultdict(list)
    metadata = defaultdict(list)

    for event_file in event_files:
        print(f"Processing: {event_file}")
        ea = EventAccumulator(event_file)
        ea.Reload()

        # Get all scalar tags
        scalar_tags = ea.Tags()["scalars"]

        # Load timing data
        for tag in scalar_tags:
            if tag.startswith("timing/"):
                events = ea.Scalars(tag)
                for event in events:
                    timing_data[tag].append(event.value)
            elif tag.startswith("timestamp/"):
                events = ea.Scalars(tag)
                for event in events:
                    timestamp_data[tag].append(event.value)
            elif tag.startswith("metadata/"):
                events = ea.Scalars(tag)
                for event in events:
                    metadata[tag].append(event.value)

    return timing_data, timestamp_data, metadata


def print_per_round_analysis(timing_data, timestamp_data, metadata):
    """Print per-round timing analysis."""
    if "metadata/round" not in metadata or len(metadata["metadata/round"]) == 0:
        print("\nNo round information available for per-round analysis.")
        return

    print("\n" + "=" * 80)
    print("PER-ROUND TIMING ANALYSIS")
    print("=" * 80)

    # Group data by round
    rounds_data = defaultdict(lambda: defaultdict(list))
    round_timestamps = defaultdict(lambda: {"starts": [], "ends": []})

    # Assuming all arrays have the same length (one entry per task)
    num_tasks = len(metadata["metadata/round"])

    for i in range(num_tasks):
        round_num = int(metadata["metadata/round"][i])

        # Collect timing data for this round
        for tag in timing_data:
            if i < len(timing_data[tag]):
                rounds_data[round_num][tag].append(timing_data[tag][i])

        # Collect timestamps for this round
        if "timestamp/task_start" in timestamp_data and i < len(timestamp_data["timestamp/task_start"]):
            round_timestamps[round_num]["starts"].append(timestamp_data["timestamp/task_start"][i])
        if "timestamp/task_end" in timestamp_data and i < len(timestamp_data["timestamp/task_end"]):
            round_timestamps[round_num]["ends"].append(timestamp_data["timestamp/task_end"][i])

    # Print per-round summary
    print(
        f"\n{'Round':<8} {'Tasks':<8} {'Wall-Clock (s)':<18} {'Avg Task (s)':<15} {'Total Comp (s)':<18} {'Training (s)':<15}"
    )
    print("-" * 100)

    sorted_rounds = sorted(rounds_data.keys())
    for round_num in sorted_rounds:
        round_data = rounds_data[round_num]
        num_round_tasks = len(round_data.get("timing/total_task_time", []))

        # Calculate wall-clock time for this round
        if round_timestamps[round_num]["starts"] and round_timestamps[round_num]["ends"]:
            round_wall_clock = max(round_timestamps[round_num]["ends"]) - min(round_timestamps[round_num]["starts"])
        else:
            round_wall_clock = 0

        # Calculate average task time
        if "timing/total_task_time" in round_data:
            avg_task_time = np.mean(round_data["timing/total_task_time"])
        else:
            avg_task_time = 0

        # Calculate total computation time
        if "timing/actual_computation" in round_data:
            total_comp = np.sum(round_data["timing/actual_computation"])
        else:
            total_comp = 0

        # Calculate total training time
        if "timing/training" in round_data:
            total_training = np.sum(round_data["timing/training"])
        else:
            total_training = 0

        print(
            f"{round_num:<8} {num_round_tasks:<8} {round_wall_clock:<18.2f} {avg_task_time:<15.2f} {total_comp:<18.2f} {total_training:<15.2f}"
        )

    print("-" * 100)

    # Print detailed statistics for each round
    print("\n" + "=" * 80)
    print("DETAILED PER-ROUND BREAKDOWN")
    print("=" * 80)

    for round_num in sorted_rounds:
        round_data = rounds_data[round_num]
        print(f"\n--- Round {round_num} ---")

        # Key metrics for this round
        metrics = {
            "Model Receive": "timing/model_receive",
            "Data Loading": "timing/data_loading",
            "Model Init": "timing/model_initialization",
            "Training": "timing/training",
            "Diff Computation": "timing/diff_computation",
            "Result Creation": "timing/result_creation",
            "Framework Overhead": "timing/framework_overhead",
            "Simulated Delay": "timing/simulated_delay",
        }

        print(f"  {'Component':<25} {'Mean':<10} {'Std':<10} {'Total':<10}")
        print("  " + "-" * 55)

        for name, tag in metrics.items():
            if tag in round_data and len(round_data[tag]) > 0:
                values = np.array(round_data[tag])
                mean = np.mean(values)
                std = np.std(values)
                total = np.sum(values)
                print(f"  {name:<25} {mean:<10.4f} {std:<10.4f} {total:<10.2f}")

        # Wall-clock time for this round
        if round_timestamps[round_num]["starts"] and round_timestamps[round_num]["ends"]:
            round_wall_clock = max(round_timestamps[round_num]["ends"]) - min(round_timestamps[round_num]["starts"])
            print(f"\n  Round {round_num} Wall-Clock Time: {round_wall_clock:.2f}s")

            # Calculate parallelism for this round
            if "timing/total_task_time" in round_data:
                total_task_time = np.sum(round_data["timing/total_task_time"])
                parallelism = total_task_time / round_wall_clock if round_wall_clock > 0 else 1
                print(f"  Average Parallelism: {parallelism:.2f}x")


def print_timing_statistics(timing_data, timestamp_data, metadata):
    """Print detailed timing statistics."""
    print("\n" + "=" * 80)
    print("OVERALL TIMING STATISTICS (all times in seconds)")
    print("=" * 80)

    # Calculate wall-clock time from timestamps
    wall_clock_time = None
    if "timestamp/task_start" in timestamp_data and "timestamp/task_end" in timestamp_data:
        all_start_times = timestamp_data["timestamp/task_start"]
        all_end_times = timestamp_data["timestamp/task_end"]
        if all_start_times and all_end_times:
            min_start = min(all_start_times)
            max_end = max(all_end_times)
            wall_clock_time = max_end - min_start
            print(f"\nWall-Clock Time Analysis:")
            print(f"  First task started at:  {min_start:.2f} (epoch timestamp)")
            print(f"  Last task ended at:     {max_end:.2f} (epoch timestamp)")
            print(f"  Total wall-clock time:  {wall_clock_time:.2f}s ({wall_clock_time/3600:.2f} hours)")
            print()

    # Order of components to display
    component_order = [
        "timing/model_receive",
        "timing/data_loading",
        "timing/model_initialization",
        "timing/training",
        "timing/diff_computation",
        "timing/result_creation",
        "timing/actual_computation",
        "timing/framework_overhead",
        "timing/simulated_delay",
        "timing/total_task_time",
    ]

    # Print header
    print(f"\n{'Component':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Total':<10} {'%':<10}")
    print("-" * 110)

    # Calculate total time for percentage calculation
    total_time_data = timing_data.get("timing/total_task_time", [])
    if total_time_data:
        total_sum = np.sum(total_time_data)
    else:
        total_sum = 1.0  # Avoid division by zero

    # Print statistics for each component
    for tag in component_order:
        if tag in timing_data:
            values = np.array(timing_data[tag])
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            total = np.sum(values)
            percentage = (total / total_sum * 100) if total_sum > 0 else 0

            component_name = tag.replace("timing/", "").replace("_", " ").title()
            print(
                f"{component_name:<30} {mean:<10.4f} {std:<10.4f} {min_val:<10.4f} {max_val:<10.4f} {total:<10.2f} {percentage:<10.1f}"
            )

    print("-" * 110)

    # Print task count and timing summary
    if total_time_data:
        print(f"\nTotal tasks processed: {len(total_time_data)}")
        print(f"Sum of all task times (sequential): {total_sum:.2f}s")
        if wall_clock_time is not None:
            print(f"Actual wall-clock time (parallel):  {wall_clock_time:.2f}s")
            parallelism = total_sum / wall_clock_time if wall_clock_time > 0 else 1
            print(f"Average parallelism factor:         {parallelism:.2f}x")

    # Print pure computation breakdown
    print("\n" + "=" * 80)
    print("PURE COMPUTATION BREAKDOWN (excluding simulated delays)")
    print("=" * 80)

    computation_components = [
        "timing/model_receive",
        "timing/data_loading",
        "timing/model_initialization",
        "timing/training",
        "timing/diff_computation",
        "timing/result_creation",
    ]

    computation_total = 0
    for tag in computation_components:
        if tag in timing_data:
            component_sum = np.sum(timing_data[tag])
            computation_total += component_sum

    print(f"\n{'Component':<30} {'Total Time':<15} {'% of Pure Computation':<25}")
    print("-" * 70)

    for tag in computation_components:
        if tag in timing_data:
            component_sum = np.sum(timing_data[tag])
            percentage = (component_sum / computation_total * 100) if computation_total > 0 else 0
            component_name = tag.replace("timing/", "").replace("_", " ").title()
            print(f"{component_name:<30} {component_sum:<15.2f} {percentage:<25.1f}")

    print("-" * 70)
    print(f"{'TOTAL PURE COMPUTATION':<30} {computation_total:<15.2f} {100.0:<25.1f}")

    # Compare to pure PyTorch simulation
    if "timing/training" in timing_data:
        training_time = np.sum(timing_data["timing/training"])
        overhead_ratio = (computation_total / training_time) if training_time > 0 else 0

        print(f"\n{'Component':<45} {'Time (s)':<15} {'Ratio':<10}")
        print("-" * 70)
        print(f"{'Pure PyTorch Training Time (sum):':<45} {training_time:<15.2f} {'1.00x':<10}")
        print(f"{'NVFlare Total Computation Time (sum):':<45} {computation_total:<15.2f} {overhead_ratio:<10.2f}x")
        print(
            f"{'Framework Overhead (sum):':<45} {(computation_total - training_time):<15.2f} {((computation_total - training_time)/training_time):<10.2f}x"
        )

        if wall_clock_time is not None:
            print("\n" + "-" * 70)
            print("Wall-Clock Time Analysis (with parallel execution):")
            print("-" * 70)
            wall_overhead_ratio = (wall_clock_time / training_time) if training_time > 0 else 0
            print(f"{'Actual wall-clock time (parallel):':<45} {wall_clock_time:<15.2f} {wall_overhead_ratio:<10.2f}x")
            print(f"{'Speedup from parallelization:':<45} {(computation_total / wall_clock_time):<15.2f}x")

            # Calculate what percentage of wall-clock time is pure training
            training_percentage = (training_time / wall_clock_time * 100) if wall_clock_time > 0 else 0
            print(f"{'Training as % of wall-clock time:':<45} {training_percentage:<15.1f}%")


def main():
    """Main function to analyze timing data."""
    # Default log directory
    log_dir = "/tmp/nvflare/tensorboard_logs/fedavg_timing"

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]

    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        print(f"Usage: python {sys.argv[0]} [log_dir]")
        sys.exit(1)

    # Load timing data
    timing_data, timestamp_data, metadata = load_timing_data(log_dir)

    if timing_data is None or len(timing_data) == 0:
        print("No timing data found")
        sys.exit(1)

    # Print overall statistics
    print_timing_statistics(timing_data, timestamp_data, metadata)

    # Print per-round analysis
    print_per_round_analysis(timing_data, timestamp_data, metadata)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
