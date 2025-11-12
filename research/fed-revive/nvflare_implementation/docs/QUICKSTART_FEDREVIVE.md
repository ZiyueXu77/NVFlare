# FedRevive Quick Start Guide

## Overview

This guide helps you quickly get started with the FedRevive implementation for NVFlare Edge.

## What is FedRevive?

FedRevive is an asynchronous federated learning algorithm that:
- ✓ Handles **model staleness** through knowledge distillation
- ✓ Uses **adaptive weighting** based on update age
- ✓ Supports **non-IID data** distribution
- ✓ Works with **asynchronous** device participation

## Installation

```bash
# Ensure NVFlare is installed
pip install nvflare

# Install additional dependencies
pip install torch torchvision filelock numpy
```

## Quick Test (Verification)

```bash
cd examples/advanced/edge/jobs/

# Verify the implementation
python verify_fedrevive.py
```

Expected output:
```
✓ All imports verified!
✓ All configurations created!
✓ Model created successfully!
✓ VERIFICATION COMPLETE
```

## Running Your First FedRevive Job

### Step 1: Check the Configuration

Open `pt_job_fedrevive.py` and review the default settings:

```python
# Key parameters (you can modify these)
devices_per_leaf = 10000      # Total device pool
device_selection_size = 200   # Devices per round
global_lr = 0.1              # Global learning rate
local_lr = 0.01              # Local learning rate (Adam)
kd_buffer_size = 3           # KD buffer size
```

### Step 2: Run the Job

```bash
python pt_job_fedrevive.py
```

### Step 3: Monitor Progress

The job will:
1. Export the recipe
2. Start the federated learning process
3. Display progress and metrics
4. Save results

## Understanding the Output

### During Training

```
FedRevive: processed child update V42 (staleness=2) with 5 devices: accepted=True
FedRevive: Globally got 20 updates: generate new model version
FedRevive: generated new model version 43 with 20 updates, KD buffer size: 3
```

This shows:
- **V42**: Global model version
- **staleness=2**: Update is 2 versions old
- **20 updates**: Buffer full, triggering aggregation
- **KD buffer size: 3**: Number of recent models for KD

### Final Results

```
Job Execution Complete
Results: /path/to/results
Status: SUCCESS
```

## Customization

### Change Non-IID Level

More non-IID (more challenging):
```python
dirichlet_alpha = 0.1  # Lower = more non-IID
```

Less non-IID (easier):
```python
dirichlet_alpha = 1.0  # Higher = closer to IID
```

### Adjust KD Strength

More aggressive KD:
```python
kd_buffer_size = 5    # More teachers
kd_beta = 0.7         # Higher KD weight
```

Less KD influence:
```python
kd_buffer_size = 2    # Fewer teachers
kd_beta = 0.3         # Lower KD weight
```

### Control Asynchrony

More asynchronous (faster):
```python
min_hole_to_fill = 1   # Dispatch immediately
```

More synchronous (more coordinated):
```python
min_hole_to_fill = 50  # Wait for many updates
```

## Example Configurations

### Configuration 1: Fast Testing

```python
devices_per_leaf = 1000        # Smaller device pool
device_selection_size = 50     # Fewer devices
max_model_version = 50         # Fewer rounds
local_iters = 10               # Fewer iterations
```

### Configuration 2: High Non-IID Challenge

```python
dirichlet_alpha = 0.1          # Very non-IID
kd_buffer_size = 5             # More KD
kd_beta = 0.7                  # Strong KD
local_iters = 30               # More local training
```

### Configuration 3: Nearly Synchronous

```python
min_hole_to_fill = 180         # Wait for most devices
device_reuse = True            # Reuse same devices
```

## Common Commands

### Verify Implementation
```bash
python verify_fedrevive.py
```

### Run Standard Job
```bash
python pt_job_fedrevive.py
```

### Check Files
```bash
ls -la processors/cifar10_fedrevive_task_processor.py
ls -la ../../nvflare/edge/assessors/fedrevive_model_manager.py
ls -la ../../nvflare/edge/tools/edge_fed_revive_recipe.py
```

## Troubleshooting

### Issue: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'nvflare'`

**Solution**:
```bash
pip install nvflare
# Or if using conda:
conda install -c conda-forge nvflare
```

### Issue: CUDA Out of Memory

**Problem**: GPU memory error during training

**Solution**:
```python
# Reduce batch size
local_batch_size = 16  # Instead of 32

# Or reduce device selection
device_selection_size = 100  # Instead of 200
```

### Issue: Dataset Download Fails

**Problem**: CIFAR10 download fails

**Solution**:
```python
# Manually download CIFAR10 first
from torchvision import datasets
datasets.CIFAR10(root='/tmp/nvflare/datasets/cifar10', train=True, download=True)
```

### Issue: Job Runs Slowly

**Problem**: Job takes too long

**Solution**:
```python
# Reduce problem size
max_model_version = 50        # Fewer rounds
device_selection_size = 50    # Fewer devices
local_iters = 10              # Fewer iterations
```

## Next Steps

### 1. Read the Documentation

```bash
# Comprehensive guide
cat FEDREVIVE_README.md

# Implementation details
cat ../../../FEDREVIVE_IMPLEMENTATION_SUMMARY.md
```

### 2. Experiment with Parameters

Try different configurations:
- Non-IID levels (dirichlet_alpha)
- KD settings (kd_buffer_size, kd_beta)
- Asynchrony (min_hole_to_fill)
- Learning rates (global_lr, local_lr)

### 3. Extend to Other Datasets

Follow the pattern in `cifar10_fedrevive_task_processor.py`:
- Create new task processor
- Implement `_prepare_dataset()`
- Implement `_pytorch_training()`
- Update model architecture if needed

### 4. Compare with FedBuff

Run both algorithms and compare:
```bash
# FedRevive
python pt_job_fedrevive.py

# FedBuff (for comparison)
python pt_job_adv.py
```

## Tips for Success

1. **Start Small**: Begin with fewer devices and rounds for testing
2. **Monitor Staleness**: Check logs for staleness values
3. **Adjust KD**: If staleness is high, increase KD strength
4. **Balance Async**: More async = faster but potentially less accurate
5. **Non-IID Matters**: Lower alpha = more challenging, may need more KD

## Performance Expectations

### Typical Training Time
- **Fast test** (50 rounds, 50 devices): ~5-10 minutes
- **Standard** (200 rounds, 200 devices): ~30-60 minutes
- **Full experiment** (500 rounds, 1000 devices): ~2-4 hours

### Expected Accuracy (CIFAR10)
- **IID data** (alpha=1.0): ~70-75% after 200 rounds
- **Moderate non-IID** (alpha=0.3): ~65-70% after 200 rounds
- **High non-IID** (alpha=0.1): ~60-65% after 200 rounds

*Note: Actual results depend on configuration and system*

## Getting Help

### Documentation
- `FEDREVIVE_README.md`: Comprehensive documentation
- `FEDREVIVE_IMPLEMENTATION_SUMMARY.md`: Implementation details
- Code comments: Extensive inline documentation

### Verification
```bash
python verify_fedrevive.py
```

### Example Code
Check `pt_job_fedrevive.py` for a complete working example

## Summary

**5-Minute Quick Start:**
```bash
# 1. Verify
cd examples/advanced/edge/jobs/
python verify_fedrevive.py

# 2. Run
python pt_job_fedrevive.py

# 3. Done! Check output for results
```

**Key Takeaways:**
- ✓ FedRevive handles staleness through KD
- ✓ Supports non-IID data distribution
- ✓ Works asynchronously with adaptive weighting
- ✓ Ready to use with sensible defaults
- ✓ Easily customizable for your needs

Happy federated learning with FedRevive! 🚀

