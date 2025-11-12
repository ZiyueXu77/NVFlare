# Performance Optimization for Device Simulation

## Problem: Sequential/Slow Training

The device simulation appears to be processing one device at a time (or very few at a time), making training extremely slow.

## Root Cause

The bottleneck is the `num_workers` parameter in `SimulationConfig`:

```python
# Current configuration (TOO LOW!)
num_workers=device_selection_size // num_leaf_nodes + 1
# With device_selection_size=100, num_leaf_nodes=4:
# num_workers = 100 // 4 + 1 = 26
```

**Only 26 worker threads** are processing **100 devices**, causing serialization!

## How num_workers Works

`num_workers` controls the `ThreadPoolExecutor` size:
- Each worker can process one device at a time
- With 26 workers processing 100 devices: ~4 sequential batches
- With 100 workers processing 100 devices: 1 parallel batch ✓

## Solution Options

### Option 1: Match Workers to Devices (Fastest)

**For maximum parallelism**, set workers equal to devices:

```python
# Fast: All devices train in parallel
num_workers = device_selection_size  # 100 workers for 100 devices
```

**Pros:**
- All devices train simultaneously
- Minimum wall-clock time per round
- Best for simulation/testing

**Cons:**
- High CPU/GPU memory usage
- May overwhelm system resources

### Option 2: CPU Core-Based (Balanced)

**For balanced performance**, match available CPU cores:

```python
import os
# Balanced: Based on CPU cores
num_workers = min(os.cpu_count(), device_selection_size)
# Example: 32 cores → 32 workers
```

**Pros:**
- Efficient CPU utilization
- Prevents resource exhaustion
- Good for production

**Cons:**
- Slower than full parallelization
- Still need adequate CPU cores

### Option 3: GPU Memory-Based (GPU-Constrained)

**For GPU memory constraints**, limit based on available VRAM:

```python
# Conservative: Limit based on GPU memory
# Rule of thumb: ~1GB VRAM per worker for CIFAR10 CNN
# For 11GB GPU: ~8-10 workers
num_workers = 8  # Adjust based on your GPU
```

**Pros:**
- Prevents GPU OOM errors
- Safe for large models
- Predictable memory usage

**Cons:**
- Slowest option
- Underutilizes CPU

### Option 4: Hybrid (Recommended)

**For best balance**, consider both CPU and GPU:

```python
import os
import torch

# Recommended: Smart selection
if torch.cuda.is_available():
    # GPU available: limit by VRAM (conservative)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    max_workers_gpu = int(gpu_mem_gb / 1.5)  # ~1.5GB per worker
    num_workers = min(max_workers_gpu, device_selection_size)
else:
    # CPU only: use all cores
    num_workers = min(os.cpu_count(), device_selection_size)

print(f"Using {num_workers} workers for {device_selection_size} devices")
```

## Quick Fixes

### Fix 1: Update pt_job_fedavg.py (Aggressive - Best for Testing)

```python
# BEFORE (slow)
simulation_config = SimulationConfig(
    task_processor=task_processor,
    job_timeout=20.0,
    num_workers=device_selection_size // num_leaf_nodes + 1,  # Only 26!
    num_devices=devices_per_leaf,
)

# AFTER (fast)
simulation_config = SimulationConfig(
    task_processor=task_processor,
    job_timeout=20.0,
    num_workers=device_selection_size,  # 100 workers for 100 devices!
    num_devices=devices_per_leaf,
)
```

### Fix 2: Update pt_job_fedavg.py (Conservative - Safe)

```python
import os

# Calculate optimal workers
max_workers = min(os.cpu_count() or 16, device_selection_size)

simulation_config = SimulationConfig(
    task_processor=task_processor,
    job_timeout=20.0,
    num_workers=max_workers,  # Based on CPU cores
    num_devices=devices_per_leaf,
)
```

### Fix 3: Update pt_job_fedavg.py (GPU-Aware - Production)

```python
import os
import torch

# Smart worker selection
if torch.cuda.is_available():
    # Conservative for GPU memory
    num_workers = min(16, device_selection_size)  # Limit to 16 for safety
else:
    # Use all CPU cores
    num_workers = min(os.cpu_count() or 8, device_selection_size)

print(f"Using {num_workers} parallel workers")

simulation_config = SimulationConfig(
    task_processor=task_processor,
    job_timeout=20.0,
    num_workers=num_workers,
    num_devices=devices_per_leaf,
)
```

## Performance Comparison

### Current Configuration (26 workers)
```
Round 1: 100 devices / 26 workers = ~4 sequential batches
Time per batch: ~30-60s (with GPU)
Total time per round: ~120-240s
```

### With 100 Workers (Full Parallel)
```
Round 1: 100 devices / 100 workers = 1 parallel batch
Time per batch: ~30-60s (with GPU)
Total time per round: ~30-60s ✓ (4x faster!)
```

### With 50 Workers (Half Parallel)
```
Round 1: 100 devices / 50 workers = 2 sequential batches
Time per batch: ~30-60s (with GPU)
Total time per round: ~60-120s ✓ (2x faster!)
```

## Additional Optimizations

### 1. Reduce Training Iterations (Faster Convergence Testing)

```python
local_iters = 10  # Instead of 25, for quick testing
```

### 2. Reduce Device Selection Size

```python
device_selection_size = 50  # Instead of 100
# Faster rounds, but may affect convergence
```

### 3. Increase Job Timeout (Prevent Premature Termination)

```python
job_timeout=60.0,  # Instead of 20.0, for slower systems
```

### 4. Disable Delays During Testing

```python
# Remove artificial delays for testing
communication_delay = {"mean": 0.0, "std": 0.0}
device_speed = {"mean": [0.0], "std": [0.0]}
```

### 5. Reduce Evaluation Frequency

```python
eval_frequency = 5  # Evaluate every 5 rounds instead of every round
```

## Monitoring Performance

### Check Parallel Execution

Add this to your code to verify parallelism:

```python
import time

start_time = time.time()
run = recipe.execute(env)
end_time = time.time()

print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
print(f"Average time per round: {(end_time - start_time) / max_model_version:.2f} seconds")
```

### Monitor System Resources

```bash
# Monitor CPU usage
htop

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check memory usage
free -h
```

## Recommended Configuration

For **best performance** with safety:

```python
import os
import torch

# Smart worker selection
if torch.cuda.is_available():
    # GPU: Be conservative to avoid OOM
    num_workers = min(32, device_selection_size)  # Up to 32 parallel
else:
    # CPU: Use all cores
    num_workers = min(os.cpu_count() or 16, device_selection_size)

# For testing/debugging: Further reduce devices
# device_selection_size = 20
# num_workers = device_selection_size

print(f"Configuration: {device_selection_size} devices, {num_workers} workers")
print(f"Expected parallelism: {device_selection_size / num_workers:.1f}x batches")

simulation_config = SimulationConfig(
    task_processor=task_processor,
    job_timeout=60.0,  # Increased timeout
    num_workers=num_workers,
    num_devices=devices_per_leaf,
)
```

## Expected Results

### Before Optimization
- 100 devices with 26 workers
- ~4 sequential batches per round
- **~2-4 minutes per round**

### After Optimization (100 workers)
- 100 devices with 100 workers
- 1 parallel batch per round
- **~30-60 seconds per round** ✓

### After Optimization (50 workers)
- 100 devices with 50 workers
- 2 parallel batches per round
- **~1-2 minutes per round** ✓

## Trade-offs

| Configuration | Speed | Memory | GPU Usage | Best For |
|---------------|-------|--------|-----------|----------|
| workers=devices | ⚡⚡⚡ Fastest | 💾💾💾 High | 🔥🔥🔥 High | Testing/Rich hardware |
| workers=CPU cores | ⚡⚡ Fast | 💾💾 Medium | 🔥🔥 Medium | Balanced/Production |
| workers=16 | ⚡ Moderate | 💾 Low | 🔥 Low | GPU-constrained |
| workers=26 (current) | 🐌 Slow | 💾 Low | 🔥 Low | Too conservative! |

## Action Items

1. **Immediate**: Change `num_workers` from 26 to at least 50-100
2. **Test**: Monitor GPU memory usage with `nvidia-smi`
3. **Adjust**: If OOM errors occur, reduce `num_workers`
4. **Measure**: Time each round to verify speedup

The main issue is that only 26 workers are handling 100 devices. Increasing to 50-100 workers will give you a **2-4x speedup**! 🚀

