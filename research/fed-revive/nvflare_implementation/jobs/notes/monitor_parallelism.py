#!/usr/bin/env python3
"""
Monitor script to check if training is actually parallel or sequential.
Run this in a separate terminal while your job is running.
"""

import subprocess
import sys
import time


def get_gpu_processes():
    """Get number of Python processes using GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        processes = [line for line in result.stdout.strip().split("\n") if "python" in line.lower()]
        return len(processes) if processes and processes[0] else 0
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return 0


def get_gpu_memory():
    """Get GPU memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception as e:
        return 0


def get_gpu_utilization():
    """Get GPU utilization percentage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception as e:
        return 0


def main():
    print("=" * 80)
    print("GPU Parallelism Monitor")
    print("=" * 80)
    print("Monitoring GPU usage to detect parallel vs sequential execution...")
    print("Press Ctrl+C to stop\n")

    print(f"{'Time':<8} {'Processes':<12} {'Memory (MB)':<15} {'GPU %':<10} {'Status'}")
    print("-" * 80)

    process_history = []
    memory_history = []

    try:
        while True:
            processes = get_gpu_processes()
            memory = get_gpu_memory()
            utilization = get_gpu_utilization()

            # Determine status
            if processes == 0:
                status = "Idle"
            elif processes == 1:
                status = "Sequential (1 device)"
            elif processes <= 3:
                status = f"Limited Parallel ({processes} devices)"
            else:
                status = f"✓ Parallel ({processes} devices)"

            # Keep history for analysis
            process_history.append(processes)
            memory_history.append(memory)
            if len(process_history) > 100:
                process_history.pop(0)
                memory_history.pop(0)

            # Print current status
            time_str = time.strftime("%H:%M:%S")
            print(f"{time_str:<8} {processes:<12} {memory:<15} {utilization:<10} {status}", flush=True)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        if process_history:
            avg_processes = sum(process_history) / len(process_history)
            max_processes = max(process_history)
            avg_memory = sum(memory_history) / len(memory_history)
            max_memory = max(memory_history)

            print(f"\nAverage GPU processes: {avg_processes:.1f}")
            print(f"Maximum GPU processes: {max_processes}")
            print(f"Average GPU memory: {avg_memory:.0f} MB")
            print(f"Maximum GPU memory: {max_memory:.0f} MB")

            print("\n" + "Analysis:")
            if max_processes == 1:
                print("❌ SEQUENTIAL: Only 1 process at a time")
                print("   Reason: ThreadPoolExecutor + single GPU serialization")
                print("   This is NORMAL with single GPU + NVFlare simulator")
            elif max_processes <= 3:
                print("⚠️  LIMITED PARALLEL: 2-3 processes")
                print("   Some parallelism, but limited by GPU memory")
            else:
                print("✓ PARALLEL: Multiple processes simultaneously")
                print(f"   Successfully running {max_processes} devices in parallel")

            # Memory analysis
            if max_memory < 3000:
                print(f"\n✓ Memory per model: ~{max_memory}MB (1 model at a time)")
            elif max_memory < 10000:
                models_on_gpu = max_memory / 1917
                print(f"\n✓ Memory suggests ~{models_on_gpu:.1f} models on GPU simultaneously")
            else:
                print(f"\n✓ High memory usage: {max_memory}MB (multiple models)")

            print("\n" + "Recommendations:")
            if max_processes == 1:
                print("1. Accept this as optimal for single GPU")
                print("2. Or reduce device_selection_size for faster testing")
                print("3. Or use multiple GPUs if available")
            elif max_memory > 10000:
                print("1. Reduce local_batch_size to fit more models")
                print("2. Or reduce subset_size")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
