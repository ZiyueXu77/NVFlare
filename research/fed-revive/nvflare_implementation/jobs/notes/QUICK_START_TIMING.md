# Quick Start: Timing Analysis

## 1. Run Your FedAvg Job

```bash
cd /media/ziyuexu/Research/Experiment/NVFlare/nvflare_fedrevive
python research/fed-revive/nvflare_implementation/jobs/pt_job_fedavg.py
```

Timing data is automatically collected!

## 2. View Real-time in TensorBoard

**In a new terminal:**

```bash
tensorboard --logdir /tmp/nvflare/tensorboard_logs/fedavg_timing
```

**Then open:** http://localhost:6006

## 3. Analyze After Run

```bash
cd research/fed-revive/nvflare_implementation/jobs
python analyze_timing.py
```

## What to Look For

### Key Metrics

| Metric | What It Tells You |
|--------|-------------------|
| `timing/training` | **Pure PyTorch training time** - Your baseline |
| `timing/actual_computation` | Total computation time (all components) |
| `timing/framework_overhead` | NVFlare overhead vs pure PyTorch |
| `timing/total_task_time` | End-to-end time including delays |

### Comparing to Pure PyTorch

The ratio you care about:
```
Overhead Ratio = timing/actual_computation / timing/training
```

**Example:**
- Pure PyTorch training: 2.35s
- NVFlare total computation: 2.75s
- **Overhead: 1.17x (17% slower)**

### Component Breakdown

Check which components take the most time:

```
Component               % of Total Computation
----------------------------------------------
Training                85.3%  ← Pure PyTorch time
Data Loading            8.7%   ← Dataset creation overhead
Model Initialization    3.2%   ← Model setup
Diff Computation        1.7%   ← Computing updates
Model Receive           0.8%   ← Receiving global model
Result Creation         0.3%   ← Serialization
```

## Console Output Example

Look for this in your logs:

```
Device device_001 Task 0 Timing Breakdown:
  Model Receive:      0.023s
  Data Loading:       0.152s
  Model Init:         0.086s
  Training:           2.346s  ← Pure PyTorch baseline
  Diff Computation:   0.046s
  Result Creation:    0.012s
  Simulated Delay:    0.000s
  Framework Overhead: 0.089s
  TOTAL:              2.754s
  Pure Computation:   2.665s (96.8% of total)
```

## Optimization Tips

If overhead is too high:

### 1. Data Loading is Slow (>10% of total)
- Currently: Dataset recreated per task (for memory efficiency)
- Trade-off: **Memory vs Speed**
- To prioritize speed: Cache dataset in memory (if you have RAM)

### 2. Framework Overhead is High (>20%)
- Check `timing/model_receive` and `timing/result_creation`
- Possible causes:
  - Serialization bottleneck
  - Network communication overhead
  - Too frequent aggregation

### 3. Model Initialization is Slow (>5%)
- Expected for federated learning (load weights each round)
- Cannot be eliminated without caching models

## TensorBoard Shortcuts

**Key Views:**
- Click "Scalars" tab
- Filter by "timing/" to see all timing metrics
- Compare multiple runs by selecting different log directories

**Useful Features:**
- Smoothing slider: Reduce noise in graphs
- Download data: Export CSV for external analysis
- Refresh: Updates automatically as job runs

## Common Issues

### Issue: No data in TensorBoard
**Solution:** Check log directory exists
```bash
ls -la /tmp/nvflare/tensorboard_logs/fedavg_timing/
```

### Issue: Old data showing
**Solution:** Clear logs before new run
```bash
rm -rf /tmp/nvflare/tensorboard_logs/fedavg_timing/*
```

### Issue: Analysis script fails
**Solution:** Install tensorboard
```bash
pip install tensorboard
```

## Summary Statistics Format

The analysis script produces:

```
TIMING STATISTICS (all times in seconds)
Component                      Mean       Total      %
--------------------------------------------------------
Training                       2.346      234.56     78.5%
Data Loading                   0.152      15.23      5.1%
...
TOTAL                          2.987      298.70     100%

PURE COMPUTATION BREAKDOWN
Pure PyTorch Training Time:              234.56s
NVFlare Total Computation Time:          266.48s
Overhead Ratio (NVFlare/PyTorch):        1.14x
Framework Overhead:                      31.92s (13.6%)
```

## Next Steps

1. **Establish Baseline**: Run pure PyTorch training to get reference time
2. **Identify Bottlenecks**: Look at component breakdown in TensorBoard
3. **Optimize**: Focus on components with highest time/overhead
4. **Re-test**: Run again and compare improvements

For detailed information, see: `TIMING_ANALYSIS_README.md`

