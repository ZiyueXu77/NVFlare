# Pre-computed Dataset Splits for Fast Initialization

## Overview

To significantly speed up device initialization (from ~12s to ~0.01s per device), you can pre-compute the dataset splits for all devices once and reuse them across experiments.

## Quick Start

### 1. Generate Pre-computed Splits

Run the preparation script to generate splits for 1000 devices:

```bash
cd /media/ziyuexu/Research/Experiment/NVFlare/nvflare_fedrevive/research/fed-revive/nvflare_implementation/jobs

python prepare_dataset_splits.py \
    --data_root /tmp/data \
    --num_devices 1000 \
    --subset_size 350 \
    --distribution non_iid_dirichlet \
    --dirichlet_alpha 0.5
```

This will create a file like:
```
/tmp/data/dataset_splits_non_iid_dirichlet_n1000_s350_a0.5.pkl
```

### 2. Use Pre-computed Splits in Your Job

Update your task processor configuration to use the pre-computed splits:

```python
from processors.cifar10_fedavg_task_processor import Cifar10FedAvgTaskProcessor

processor = Cifar10FedAvgTaskProcessor(
    data_root="/tmp/data",
    subset_size=350,
    communication_delay={"mean": 0.1, "std": 0.02},
    device_speed={"mean": [1.0], "std": [0.0]},
    local_batch_size=32,
    local_iters=25,
    local_lr=0.0003,
    data_distribution="non_iid_dirichlet",
    dirichlet_alpha=0.5,
    # Add this line to use pre-computed splits:
    dataset_splits_file="/tmp/data/dataset_splits_non_iid_dirichlet_n1000_s350_a0.5.pkl"
)
```

## Performance Improvement

**Without pre-computed splits:**
- Dataset preparation: ~12.8s per device
- 1000 devices: ~3.5 hours of initialization time (if sequential)

**With pre-computed splits:**
- One-time generation: ~30-60s for all 1000 devices
- Per-device loading: ~0.01s
- 1000 devices: ~10s total initialization time

**Speedup: ~1200x faster!**

## Script Options

```bash
python prepare_dataset_splits.py --help
```

### Arguments:

- `--data_root`: Root directory for CIFAR10 data (default: `/tmp/data`)
- `--num_devices`: Number of devices to generate splits for (default: 1000)
- `--subset_size`: Number of samples per device (default: 350)
- `--distribution`: Data distribution type - `iid` or `non_iid_dirichlet` (default: `non_iid_dirichlet`)
- `--dirichlet_alpha`: Dirichlet concentration parameter (default: 0.5)
- `--output`: Custom output file path (optional)

### Examples:

**IID distribution:**
```bash
python prepare_dataset_splits.py \
    --distribution iid \
    --num_devices 1000 \
    --subset_size 350
```

**Different alpha value:**
```bash
python prepare_dataset_splits.py \
    --distribution non_iid_dirichlet \
    --dirichlet_alpha 0.1 \
    --num_devices 1000
```

**More devices:**
```bash
python prepare_dataset_splits.py \
    --num_devices 5000 \
    --subset_size 350
```

## Output Statistics

The script provides detailed statistics about the generated splits:

- Class distribution per device (mean, std, min, max)
- Entropy measurements (higher = more uniform distribution)
- Total samples and file size

Example output:
```
DATASET SPLIT STATISTICS
================================================================================

Class      Mean            Std             Min             Max            
----------------------------------------------------------------------
0          25.34           12.45           3               65             
1          28.91           14.23           2               72             
...

Class Distribution Entropy (higher = more uniform):
  Mean:   1.8234
  Std:    0.2145
  Min:    0.9876
  Max:    2.2543
  Max possible (uniform): 2.3026
```

## File Format

The splits file is a pickled dictionary containing:

```python
{
    'device_splits': {0: [indices...], 1: [indices...], ...},
    'num_devices': 1000,
    'subset_size': 350,
    'data_distribution': 'non_iid_dirichlet',
    'dirichlet_alpha': 0.5,
    'total_samples': 50000,
}
```

## Fallback Behavior

If the splits file is not provided or doesn't match the configuration, the task processor will automatically fall back to computing splits on-the-fly, ensuring backward compatibility.

## Notes

- The splits file is deterministic based on device ID
- Each device gets the same split across different runs
- The file can be reused across multiple experiments with the same configuration
- File size is typically 5-10 MB for 1000 devices

