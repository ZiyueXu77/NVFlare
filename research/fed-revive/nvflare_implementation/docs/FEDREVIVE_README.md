# FedRevive Implementation for NVFlare Edge

This directory contains a complete implementation of the **FedRevive** algorithm for NVFlare's Edge federated learning framework. FedRevive addresses challenges in asynchronous federated learning through knowledge distillation and adaptive staleness handling.

## Overview

FedRevive is an asynchronous federated learning algorithm that enhances model convergence and handles staleness through:

1. **Knowledge Distillation (KD)**: Recent model updates serve as teachers to guide new updates
2. **Adaptive Weighting**: Updates are weighted based on their staleness
3. **Buffered Aggregation**: Multiple updates are collected before generating a new global model

This implementation is based on the research code in `research/fed-revive/codes/` and adapted for NVFlare's Edge framework.

## Files Created

### Core Algorithm Components

1. **`nvflare/edge/assessors/fedrevive_model_manager.py`**
   - Implements the `FedReviveModelManager` class
   - Handles model aggregation with KD augmentation
   - Maintains a KD buffer of recent model updates
   - Applies adaptive beta weighting based on staleness
   - Key features:
     - Buffered asynchronous aggregation
     - Knowledge distillation for staleness mitigation
     - Adaptive weighting: `adaptive_beta = kd_beta * (1 / (1 + staleness^0.5))`

2. **`nvflare/edge/tools/edge_fed_revive_recipe.py`**
   - Implements the `EdgeFedReviveRecipe` class
   - High-level recipe for creating FedRevive jobs
   - Includes configuration classes:
     - `FedReviveConfig`: KD-specific parameters
     - `ModelManagerConfig`: Global aggregation settings
     - `DeviceManagerConfig`: Device selection settings
     - `SimulationConfig`: Simulation parameters
     - `EvaluatorConfig`: Evaluation settings

### CIFAR10 Implementation

3. **`examples/advanced/edge/jobs/processors/cifar10_fedrevive_task_processor.py`**
   - Implements `Cifar10FedReviveTaskProcessor` class
   - Device-side training following FedRevive specifications:
     - **Adam optimizer** (instead of SGD) for better async performance
     - **Fixed iterations** (not epochs) for consistent training
     - **Non-IID data** via Dirichlet sampling (α=0.3 default)
     - **Computational heterogeneity** simulation
   - Includes data normalization for CIFAR10

4. **`examples/advanced/edge/jobs/pt_job_fedrevive.py`**
   - Main script to create and run FedRevive jobs
   - Comprehensive configuration with sensible defaults
   - Detailed logging and status reporting

## Key Features

### FedRevive Algorithm

The implementation follows the FedRevive algorithm with these key components:

1. **KD Buffer Management**
   ```python
   # Most recent updates stored as teachers
   kd_buffer = deque(maxlen=kd_buffer_size)  # Default: 3
   # Each entry: (model_params, staleness, device_id)
   ```

2. **Adaptive Beta Calculation**
   ```python
   if staleness > 0:
       adaptive_beta = kd_beta * (1.0 / (1.0 + staleness ** 0.5))
   else:
       adaptive_beta = 0.0
   ```

3. **Aggregation Strategy**
   - Collect `num_updates_for_model` updates (default: 20)
   - Apply KD augmentation to stale updates
   - Aggregate with global learning rate
   - Generate new global model

### Local Training (FedRevive Style)

Following the reference implementation in `research/fed-revive/codes/`:

```python
# Optimizer: Adam (better for async FL)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training: Fixed iterations (not epochs)
for iter in range(local_iters):  # Default: 20 iterations
    # Standard training loop
    ...
```

### Non-IID Data Distribution

Implements Dirichlet-based non-IID sampling:

```python
# Sample class proportions for each device
class_probs = np.random.dirichlet([alpha] * num_classes)

# Lower alpha = more non-IID
# alpha = 0.3 (default): moderately non-IID
# alpha = 0.1: highly non-IID
# alpha = 1.0: nearly IID
```

## Usage

### Basic Usage

```python
from processors.cifar10_fedrevive_task_processor import Cifar10FedReviveTaskProcessor
from processors.models.cifar10_model import Cifar10ConvNet
from nvflare.edge.tools.edge_fed_revive_recipe import (
    EdgeFedReviveRecipe,
    FedReviveConfig,
    ModelManagerConfig,
    DeviceManagerConfig,
)

# Configure FedRevive
fedrevive_config = FedReviveConfig(
    kd_enabled=True,
    kd_buffer_size=3,
    kd_beta=0.5,
    kd_temperature=1.0,
)

# Configure model manager
model_manager_config = ModelManagerConfig(
    global_lr=0.1,
    num_updates_for_model=20,
    max_model_version=200,
)

# Configure device manager
device_manager_config = DeviceManagerConfig(
    device_selection_size=200,
    min_hole_to_fill=10,
    device_reuse=False,
)

# Create recipe
recipe = EdgeFedReviveRecipe(
    job_name="fedrevive_cifar10",
    model=Cifar10ConvNet(),
    model_manager_config=model_manager_config,
    device_manager_config=device_manager_config,
    fedrevive_config=fedrevive_config,
)
```

### Running the Example

```bash
# Run the FedRevive job
python pt_job_fedrevive.py
```

## Configuration Parameters

### FedRevive-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kd_enabled` | `True` | Enable KD augmentation |
| `kd_buffer_size` | `3` | Number of recent models as teachers |
| `kd_beta` | `0.5` | Base weight for KD (adaptive) |
| `kd_temperature` | `1.0` | Temperature for KD softmax |
| `kd_learning_rate` | `0.01` | LR for KD optimization |
| `kd_num_iters` | `10` | KD optimization iterations |

### Global Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `global_lr` | `0.1` | Global learning rate |
| `num_updates_for_model` | `20` | Buffer size for aggregation |
| `max_model_version` | `200` | Maximum rounds |
| `staleness_weight` | `False` | Simple staleness weighting |

### Local Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `local_batch_size` | `32` | Batch size |
| `local_iters` | `20` | Training iterations (not epochs) |
| `local_lr` | `0.01` | Learning rate (Adam) |
| `data_distribution` | `"non_iid_dirichlet"` | Data distribution type |
| `dirichlet_alpha` | `0.3` | Dirichlet concentration |

### Device Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `devices_per_leaf` | `10000` | Total device pool |
| `device_selection_size` | `200` | Devices per round |
| `min_hole_to_fill` | `10` | Updates before dispatch |
| `device_reuse` | `False` | Allow device reuse |

## Comparison with FedBuff

| Feature | FedBuff | FedRevive |
|---------|---------|-----------|
| **Aggregation** | Simple averaging | KD-augmented averaging |
| **Staleness Handling** | Optional weighting | Adaptive KD |
| **Optimizer** | SGD | Adam |
| **Training** | Epochs | Fixed iterations |
| **Data** | Any | Non-IID (Dirichlet) |
| **KD Buffer** | No | Yes (size: 3) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NVFlare Server                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         FedReviveModelManager                          │ │
│  │  - Maintains KD buffer (recent updates)                │ │
│  │  - Calculates adaptive beta                            │ │
│  │  - Aggregates with KD augmentation                     │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         BuffDeviceManager                              │ │
│  │  - Selects devices                                     │ │
│  │  - Manages device pool                                 │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Global Model (version v)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Leaf Clients (Edge)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Device 1   │  │   Device 2   │  │   Device N   │     │
│  │              │  │              │  │              │     │
│  │ Cifar10Fed   │  │ Cifar10Fed   │  │ Cifar10Fed   │     │
│  │ ReviveTask   │  │ ReviveTask   │  │ ReviveTask   │     │
│  │ Processor    │  │ Processor    │  │ Processor    │     │
│  │              │  │              │  │              │     │
│  │ - Non-IID    │  │ - Non-IID    │  │ - Non-IID    │     │
│  │   data       │  │   data       │  │   data       │     │
│  │ - Adam opt   │  │ - Adam opt   │  │ - Adam opt   │     │
│  │ - Fixed iter │  │ - Fixed iter │  │ - Fixed iter │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Model Update (staleness s)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Server Aggregation (FedRevive)                 │
│                                                             │
│  1. Receive update with staleness s                         │
│  2. Add to KD buffer: buffer.append((params, s, device))   │
│  3. Calculate: adaptive_beta = β * (1/(1+s^0.5))           │
│  4. If s > 0 and β > 0.01:                                 │
│     - Use KD buffer models as teachers                      │
│     - Augment update with KD                                │
│  5. Aggregate: new_model += lr * augmented_update          │
│  6. After N updates: Generate new global model             │
└─────────────────────────────────────────────────────────────┘
```

## Differences from Reference Implementation

While this implementation follows the FedRevive algorithm from `research/fed-revive/codes/`, there are some adaptations for NVFlare:

1. **Architecture**: Adapted to NVFlare's Edge framework with Recipe pattern
2. **KD Implementation**: Simplified KD (full implementation would require model inference on server)
3. **Buffer Management**: Uses NVFlare's model versioning system
4. **Evaluation**: Integrated with NVFlare's GlobalEvaluator

The core algorithm (buffered aggregation, adaptive weighting, KD augmentation) remains faithful to the original FedRevive design.

## References

1. FedRevive Paper: [Add paper reference if available]
2. Original Implementation: `research/fed-revive/codes/`
3. NVFlare Edge Documentation: https://nvflare.readthedocs.io/

## Future Enhancements

Potential improvements:

1. **Full KD Implementation**: Complete KD with model inference on server using `kd_dataset`
2. **DFKD Support**: Data-Free Knowledge Distillation as in reference implementation
3. **Adaptive Learning Rates**: Dynamic adjustment of global and local LR
4. **Advanced Metrics**: Per-class accuracy, staleness distribution tracking
5. **Multi-Dataset Support**: Extend to CIFAR100, FEMNIST, etc.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure NVFlare is properly installed
2. **CUDA Errors**: Check GPU availability, falls back to CPU automatically
3. **Dataset Download**: First run downloads CIFAR10, may take time
4. **Memory Issues**: Reduce `device_selection_size` or `kd_buffer_size`

### Performance Tips

1. **For faster training**: Increase `local_lr`, decrease `local_iters`
2. **For better convergence**: Increase `kd_buffer_size`, decrease `global_lr`
3. **For more async**: Decrease `min_hole_to_fill`
4. **For more sync**: Increase `min_hole_to_fill` towards `device_selection_size`

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0.

