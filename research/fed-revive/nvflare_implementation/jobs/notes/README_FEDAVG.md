# FedAvg Implementation for NVFlare

This directory contains a complete implementation of the standard FedAvg algorithm for NVFlare's Edge Federated Learning framework, designed to match the configuration from `research/fed-revive/codes/SampleConfigFiles/cifar10/fedavg.json`.

## Files

### Core Implementation

1. **`pt_job_fedavg.py`** - Main job script for running FedAvg
   - Configures synchronous FedAvg with 100 devices per round
   - Matches all hyperparameters from the original codes implementation
   - Includes detailed documentation and configuration printing

2. **`processors/cifar10_fedavg_task_processor.py`** - Task processor for FedAvg local training
   - Implements local training with Adam optimizer
   - Supports non-IID data distribution via Dirichlet sampling
   - Fixed iteration training (25 iterations per device)
   - Computational heterogeneity simulation

### Documentation

3. **`FEDAVG_CONFIG_COMPARISON.md`** - Detailed configuration comparison
   - Side-by-side comparison of all parameters
   - Explains differences and similarities
   - Documents expected behavior

4. **`README_FEDAVG.md`** (this file) - Quick start guide

## Key Configuration Parameters

The implementation uses the following key parameters from `codes/cifar10/fedavg.json`:

### Global FL Settings
- **Total devices**: 1000 per leaf node
- **Devices per round**: 100 (synchronous)
- **Server learning rate**: 1.4
- **Algorithm**: Synchronous FedAvg

### Data Settings
- **Dataset**: CIFAR10
- **Data distribution**: Non-IID Dirichlet (alpha=0.5)
- **Samples per device**: 350

### Local Training
- **Optimizer**: Adam
- **Learning rate**: 0.0003
- **Batch size**: 32
- **Iterations per round**: 25 (fixed, not epochs)

## Quick Start

### Prerequisites

1. NVFlare edge environment is set up
2. CIFAR10 dataset will be auto-downloaded to `/tmp/nvflare/datasets/cifar10`
3. Workspace is initialized at `/tmp/nvflare/workspaces/edge_example/`

### Running FedAvg

```bash
# Navigate to the jobs directory
cd research/fed-revive/nvflare_implementation/jobs

# Run the FedAvg job
python pt_job_fedavg.py
```

The script will:
1. Print detailed configuration information
2. Create the FedAvg recipe with all components
3. Export the job configuration
4. Execute the federated learning job
5. Display results and status

### Expected Output

```
================================================================================
Creating FedAvg Federated Learning Recipe
================================================================================
Algorithm: FedAvg (Synchronous)
Dataset: CIFAR10
Data Distribution: non_iid_dirichlet (alpha=0.5)
Total Devices per Leaf: 1000
Total Devices Across 4 Leaves: 4000
Selected Devices per Round: 100
Samples per Device: 350
Synchronous Aggregation: Wait for all 100 devices
Global Learning Rate (Server LR): 1.4
Local Learning Rate: 0.0003
Local Batch Size: 32
Local Iterations: 25
Max Rounds: 200
================================================================================
```

## Comparing with Original Implementation

To compare results with the original implementation:

### Run Original Implementation
```bash
cd research/fed-revive/codes
python main.py --config SampleConfigFiles/cifar10/fedavg.json
```

### Run NVFlare Implementation
```bash
cd research/fed-revive/nvflare_implementation/jobs
python pt_job_fedavg.py
```

Both should produce similar:
- Convergence rates
- Final test accuracies
- Training dynamics

See `FEDAVG_CONFIG_COMPARISON.md` for detailed parameter matching.

## Customization

### Adjusting Number of Rounds

In `pt_job_fedavg.py`, modify:
```python
max_model_version = 200  # Change to desired number of rounds
```

### Changing Data Distribution

To use IID data distribution:
```python
data_distribution = "iid"  # Instead of "non_iid_dirichlet"
```

To adjust non-IID level (lower alpha = more non-IID):
```python
dirichlet_alpha = 0.3  # More non-IID
dirichlet_alpha = 1.0  # Less non-IID
```

### Modifying Device Selection

To change the number of devices selected per round:
```python
device_selection_size = 50  # Instead of 100

# IMPORTANT: Also update for synchronous behavior
num_updates_for_model = device_selection_size
min_hole_to_fill = device_selection_size
```

### Adjusting Local Training

To change local training iterations:
```python
local_iters = 50  # Instead of 25
```

To change learning rate:
```python
local_lr = 0.001  # Instead of 0.0003
```

## Architecture

### Component Structure

```
pt_job_fedavg.py
│
├── Cifar10FedAvgTaskProcessor (processors/cifar10_fedavg_task_processor.py)
│   ├── Data loading and Dirichlet sampling
│   ├── Local training with Adam optimizer
│   └── Model update calculation
│
├── EdgeFedBuffRecipe (configured for synchronous FedAvg)
│   ├── ModelManagerConfig (aggregation settings)
│   ├── DeviceManagerConfig (device selection)
│   ├── EvaluatorConfig (global evaluation)
│   └── SimulationConfig (device simulation)
│
└── ProdEnv (execution environment)
```

### Key Design Choices

1. **Synchronous Behavior**: Achieved by setting `num_updates_for_model = device_selection_size`
   - Server waits for all 100 devices before aggregating
   - No buffered asynchronous behavior

2. **Data Distribution**: Identical Dirichlet sampling as original codes
   - Each device gets a Dirichlet-sampled class distribution
   - Reproducible via device-specific random seed

3. **Local Training**: Fixed iterations (not epochs)
   - Trains for exactly 25 iterations per round
   - Matches the original codes implementation

4. **Model Updates**: Returns weight differences (not full weights)
   - More efficient for large models
   - Standard practice in federated learning

## Troubleshooting

### Issue: CUDA Out of Memory

Reduce batch size or number of workers:
```python
local_batch_size = 16  # Reduce from 32
# In task processor: num_workers=1  # Reduce from 2
```

### Issue: Job Timeout

Increase timeout:
```python
simulation_config = SimulationConfig(
    task_processor=task_processor,
    job_timeout=50.0,  # Increase from 20.0
    ...
)
```

### Issue: Different Results from Original Implementation

Check:
1. Random seeds are properly set
2. Data normalization is applied (should be automatic)
3. Device-specific seeding is working (based on device_id hash)
4. All hyperparameters match (see FEDAVG_CONFIG_COMPARISON.md)

## Performance Tips

1. **GPU Acceleration**: Automatically uses CUDA if available
2. **Parallel Workers**: Adjust `num_workers` in SimulationConfig
3. **Evaluation Frequency**: Reduce `eval_frequency` for faster training
4. **Model History**: Set `max_model_history` to a smaller value to save memory

## Additional Resources

- **NVFlare Documentation**: https://nvidia.github.io/NVFlare/
- **FedAvg Paper**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **Original Codes**: `research/fed-revive/codes/`
- **Configuration Comparison**: `FEDAVG_CONFIG_COMPARISON.md`

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0.

