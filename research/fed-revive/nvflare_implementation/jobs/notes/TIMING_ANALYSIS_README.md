# Timing Analysis for FedAvg Job

This document explains how to use the comprehensive timing instrumentation added to track and analyze performance compared to pure PyTorch simulation.

## Overview

The FedAvg task processor now includes detailed timing measurements for every component of the federated learning workflow:

### Measured Components

1. **Model Receiving** - Time to receive and parse global model from server
2. **Data Loading** - Time to load CIFAR10 dataset and create DataLoader
3. **Model Initialization** - Time to initialize ResNet18 and load weights
4. **Training** - Pure PyTorch training time (forward + backward passes)
5. **Diff Computation** - Time to compute parameter differences
6. **Result Creation** - Time to create and serialize DXO result
7. **Framework Overhead** - Unmeasured time in NVFlare framework
8. **Simulated Delays** - Artificial delays for heterogeneity simulation

## Real-time Monitoring with TensorBoard

### Start TensorBoard

```bash
tensorboard --logdir /tmp/nvflare/tensorboard_logs/fedavg_timing
```

Then open your browser to: http://localhost:6006

### Available Visualizations

TensorBoard will show:

- **timing/** - All timing metrics over tasks
  - `timing/model_receive` - Model receiving time
  - `timing/data_loading` - Data loading time
  - `timing/model_initialization` - Model setup time
  - `timing/training` - Pure training time (compare to PyTorch baseline)
  - `timing/diff_computation` - Model diff computation time
  - `timing/result_creation` - Result serialization time
  - `timing/actual_computation` - Sum of all computation (no simulated delays)
  - `timing/framework_overhead` - NVFlare framework overhead
  - `timing/simulated_delay` - Artificial delays
  - `timing/total_task_time` - End-to-end task time

- **training/** - Training metrics
  - `training/loss` - Training loss per task
  - `training/accuracy` - Training accuracy per task

### Interpreting the Graphs

1. **Pure PyTorch Comparison**: Look at `timing/training` to see actual training time
2. **Framework Overhead**: Compare `timing/actual_computation` to `timing/training`
3. **Bottlenecks**: Identify which components take the most time
4. **Stability**: Check if times are consistent or have high variance

## Post-Run Analysis

### Automated Analysis Script

Run the analysis script to get detailed statistics:

```bash
python analyze_timing.py
```

Or specify a custom log directory:

```bash
python analyze_timing.py /path/to/tensorboard/logs
```

### Sample Output

```
TIMING STATISTICS (all times in seconds)
================================================================================

Component                      Mean       Std        Min        Max        Total      %         
--------------------------------------------------------------------------------------------------------------
Model Receive                  0.0234     0.0045     0.0189     0.0456     2.34       2.1       
Data Loading                   0.1523     0.0234     0.1234     0.2345     15.23      13.8      
Model Initialization           0.0856     0.0123     0.0678     0.1234     8.56       7.7       
Training                       2.3456     0.1234     2.1234     2.6789     234.56     212.0     
Diff Computation               0.0456     0.0089     0.0345     0.0678     4.56       4.1       
Result Creation                0.0123     0.0023     0.0089     0.0178     1.23       1.1       
...

PURE COMPUTATION BREAKDOWN (excluding simulated delays)
================================================================================
Pure PyTorch Training Time:              234.56s
NVFlare Total Computation Time:          266.48s
Overhead Ratio (NVFlare/PyTorch):        1.14x
Framework Overhead:                      31.92s (13.6%)
```

## Console Logs

Each task logs a detailed timing breakdown:

```
Device device_123 Task 42 Timing Breakdown:
  Model Receive:      0.023s
  Data Loading:       0.152s
  Model Init:         0.086s
  Training:           2.346s
  Diff Computation:   0.046s
  Result Creation:    0.012s
  Simulated Delay:    0.000s (speed: 0.000s, comm: 0.000s)
  Framework Overhead: 0.089s
  TOTAL:              2.754s
  Pure Computation:   2.665s (96.8% of total)
```

## Customization

### Changing Log Directory

Edit `cifar10_fedavg_task_processor.py`:

```python
TENSORBOARD_LOG_DIR = "/your/custom/path/tensorboard_logs"
```

### Adjusting Flush Frequency

By default, TensorBoard logs are flushed every 10 tasks. To change:

```python
# In process_task method
if self.task_counter % 10 == 0:  # Change 10 to your desired frequency
    self.tensorboard_writer.flush()
```

## Performance Tips

### If Timing Shows High Overhead

1. **Data Loading** is slow:
   - Dataset is being recreated per task (by design for memory efficiency)
   - Trade-off: Speed vs Memory usage
   - To speed up: Keep dataset in memory (revert to original caching)

2. **Model Initialization** is slow:
   - Model creation + weight loading per task
   - Expected behavior for federated learning

3. **Framework Overhead** is high:
   - Check for serialization bottlenecks
   - Look at `timing/result_creation` and `timing/model_receive`
   - Consider reducing communication frequency

4. **Diff Computation** is slow:
   - Computing parameter differences on CPU
   - Normal for large models like ResNet18

## Comparing to Pure PyTorch Baseline

To establish a baseline, run pure PyTorch training without NVFlare:

```python
# Simple PyTorch benchmark
import time
import torch
from models.resnet_18 import ResNet18Local

# Setup model and data (use same hyperparameters)
model = ResNet18Local()
# ... load data ...

# Time pure training
start = time.time()
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward + backward
        ...
end = time.time()

print(f"Pure PyTorch time: {end - start:.2f}s")
```

Compare this to the `timing/training` metric in TensorBoard.

## Troubleshooting

### No TensorBoard Data

1. Check if log directory exists: `ls -la /tmp/nvflare/tensorboard_logs/fedavg_timing`
2. Check for event files: `ls -la /tmp/nvflare/tensorboard_logs/fedavg_timing/events.out.*`
3. Check console logs for "TensorBoard logging initialized" message

### TensorBoard Shows Old Data

Clear old runs:
```bash
rm -rf /tmp/nvflare/tensorboard_logs/fedavg_timing/*
```

### Analysis Script Errors

Ensure tensorboard package is installed:
```bash
pip install tensorboard
```

## Integration with Existing Workflow

The timing instrumentation is fully integrated and requires no additional configuration. Simply run your FedAvg job as normal:

```bash
python pt_job_fedavg.py
```

All timing data is automatically logged!

## Advanced: Custom Timing Measurements

To add your own timing measurements, use this pattern:

```python
# In task processor
custom_start = time.time()
# ... your code ...
custom_time = time.time() - custom_start

# Log to TensorBoard
self.tensorboard_writer.add_scalar('timing/custom_metric', custom_time, self.task_counter)
```

## References

- TensorBoard documentation: https://www.tensorflow.org/tensorboard
- NVFlare documentation: https://nvidia.github.io/NVFlare/

