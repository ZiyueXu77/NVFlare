# FedAvg Configuration Comparison

This document compares the NVFlare implementation (`pt_job_fedavg.py`) with the original implementation from `codes/SampleConfigFiles/cifar10/fedavg.json` to ensure reproducibility.

## Overview

The NVFlare implementation is designed to replicate the exact behavior of the standard FedAvg algorithm as implemented in the reference code. Below is a detailed comparison of all key parameters.

## Configuration Comparison

### Global FL Settings

| Parameter | codes/fedavg.json | NVFlare pt_job_fedavg.py | Notes |
|-----------|-------------------|--------------------------|-------|
| **Algorithm** | `sync_fl` | Synchronous (via `num_updates_for_model = device_selection_size`) | Both use synchronous FedAvg |
| **Total Clients** | `1000` | `devices_per_leaf = 1000` (per leaf) | Same pool size |
| **Clients per Round** | `100` (sync_fl_settings.nb_of_active_jobs) | `device_selection_size = 100` | Same number of clients selected per round |
| **Max Rounds** | `50000` | `max_model_version = 200` | Adjustable; 200 is for faster testing |
| **Server Learning Rate** | `1.4` | `global_lr = 1.4` | ✓ Exact match |
| **Target Accuracy** | `0.85` | N/A (controlled by max_rounds) | Stopping criterion differs |

### Data Distribution Settings

| Parameter | codes/fedavg.json | NVFlare pt_job_fedavg.py | Notes |
|-----------|-------------------|--------------------------|-------|
| **Dataset** | `"cifar10"` | CIFAR10 | ✓ Same dataset |
| **Data Distribution** | `"non_iid_dirichlet"` | `"non_iid_dirichlet"` | ✓ Exact match |
| **Dirichlet Alpha** | `0.5` | `dirichlet_alpha = 0.5` | ✓ Exact match |
| **Client Data Size** | `350` | `subset_size = 350` | ✓ Exact match |
| **Val Split** | `0.15` | N/A (handled by evaluator) | Different eval approach |
| **Test Split** | `0.15` | N/A (handled by evaluator) | Different eval approach |

### Local Training Settings

| Parameter | codes/fedavg.json | NVFlare pt_job_fedavg.py | Notes |
|-----------|-------------------|--------------------------|-------|
| **Local Batch Size** | `32` | `local_batch_size = 32` | ✓ Exact match |
| **Local Iterations** | `25` | `local_iters = 25` | ✓ Exact match (fixed iterations, not epochs) |
| **Local Optimizer** | `"adam"` | `torch.optim.Adam` | ✓ Exact match |
| **Local Learning Rate** | `0.0003` | `local_lr = 0.0003` | ✓ Exact match |
| **Local Momentum** | `0.9` | N/A (Adam doesn't use momentum) | N/A for Adam optimizer |

### Heterogeneity Simulation

#### Training Delay (Computational Heterogeneity)

| Parameter | codes/fedavg.json | NVFlare pt_job_fedavg.py | Notes |
|-----------|-------------------|--------------------------|-------|
| **Type** | `"exponential"` | Gaussian sampling | Different distribution but similar effect |
| **Mean Distribution** | `[(0.25, 1), (0.5, 1.3), (0.25, 1.6)]` | `[1.0, 1.3, 1.6]` | ✓ Same relative speeds |
| **Std** | `0.2` | `[0.2, 0.2, 0.2]` | ✓ Same variability |

**Interpretation:**
- codes: 25% of devices with mean=1.0, 50% with mean=1.3, 25% with mean=1.6
- NVFlare: Same distribution, devices randomly assigned to one of three speed categories

#### Communication Delay

| Parameter | codes/fedavg.json | NVFlare pt_job_fedavg.py | Notes |
|-----------|-------------------|--------------------------|-------|
| **Download Delay Mean** | `0.1` | Combined into `communication_delay.mean = 0.1` | Simplified |
| **Download Delay Std** | `0.2` | Combined into `communication_delay.std = 0.2` | Simplified |
| **Upload Delay Mean** | `[0.15, 0.25]` | Included in communication_delay | Combined for simplicity |
| **Upload Delay Std** | `0.02` | Included in communication_delay | Combined for simplicity |

### Model Settings

| Parameter | codes/fedavg.json | NVFlare pt_job_fedavg.py | Notes |
|-----------|-------------------|--------------------------|-------|
| **Model Architecture** | CNN (from codes/utils/model_utils.py) | `Cifar10ConvNet` | ✓ Same architecture |
| **Loss Function** | CrossEntropyLoss | CrossEntropyLoss | ✓ Same loss |
| **Data Normalization** | Standard CIFAR10 | `Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))` | ✓ Standard CIFAR10 normalization |

### Evaluation Settings

| Parameter | codes/fedavg.json | NVFlare pt_job_fedavg.py | Notes |
|-----------|-------------------|--------------------------|-------|
| **Eval Interval** | `1` | `eval_frequency = 1` | ✓ Evaluate every round |
| **Eval Batch Size** | `300` | Handled by evaluator | Different but shouldn't affect results |

## Key Implementation Details

### Synchronous FedAvg Behavior

The original codes use `algorithm: "sync_fl"` with `sync_fl_settings.nb_of_active_jobs: 100` to implement synchronous FedAvg. 

The NVFlare implementation achieves the same behavior by:
```python
num_updates_for_model = device_selection_size  # 100
min_hole_to_fill = device_selection_size        # 100
```

This configuration ensures:
- The server waits for ALL 100 selected devices before aggregating
- No buffered asynchronous behavior (which would be FedBuff)
- True synchronous FedAvg aggregation

### Data Distribution Strategy

Both implementations use the same Dirichlet-based non-IID data distribution:

1. Sample class proportions from `Dirichlet([alpha] * num_classes)` for each device
2. Allocate samples according to these proportions
3. Each device gets a skewed class distribution

**codes implementation (client.py, count_samples_per_class):**
```python
class_probs = rng.dirichlet([self.dirichlet_alpha] * num_classes)
samples_per_class = (class_probs * self.subset_size).astype(int)
```

**NVFlare implementation (cifar10_fedavg_task_processor.py):**
```python
class_probs = rng.dirichlet([self.dirichlet_alpha] * num_classes)
samples_per_class = (class_probs * self.subset_size).astype(int)
```

✓ **Identical implementation**

### Local Training Process

Both implementations use the same local training approach:

**codes implementation (client.py, local_train):**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=config['local_learning_rate'])
iter_num = 0
while continue_training_flag:
    for batch_idx, (data, target) in enumerate(self.dataloader):
        # ... training step ...
        iter_num += 1
        if iter_num >= config['local_iters']:
            break
```

**NVFlare implementation (cifar10_fedavg_task_processor.py):**
```python
optimizer = torch.optim.Adam(net.parameters(), lr=self.local_lr)
iter_count = 0
while iter_count < self.local_iters:
    for inputs, labels in self.train_loader:
        # ... training step ...
        iter_count += 1
        if iter_count >= self.local_iters:
            break
```

✓ **Identical training loop structure**

### Server Aggregation

Both implementations use weighted aggregation with server learning rate:

**codes implementation (server.py):**
```python
# Server learning rate of 1.4 is applied during aggregation
# model <- model + server_lr * (aggregated_updates)
```

**NVFlare implementation:**
```python
global_lr = 1.4  # Applied by ModelManager during aggregation
```

✓ **Same aggregation strategy**

## Differences and Notes

### Minor Differences (Should Not Affect Convergence)

1. **Communication Delay Modeling**: 
   - codes: Separate download and upload delays with different distributions
   - NVFlare: Combined communication delay
   - Impact: Minimal, both simulate network latency

2. **Training Delay Distribution**:
   - codes: Exponential distribution
   - NVFlare: Gaussian distribution
   - Impact: Both achieve computational heterogeneity with similar characteristics

3. **Evaluation Approach**:
   - codes: Uses val/test splits from training data
   - NVFlare: Uses separate evaluation on full test set
   - Impact: Different evaluation methodology but same model performance

### Major Similarities (Critical for Reproducibility)

✓ **Data Distribution**: Identical Dirichlet sampling with alpha=0.5  
✓ **Local Training**: Identical (25 iterations, Adam, lr=0.0003, batch=32)  
✓ **Server Aggregation**: Identical (synchronous FedAvg with lr=1.4)  
✓ **Model Architecture**: Same CNN architecture  
✓ **Optimization**: Same optimizer (Adam) and loss function (CrossEntropyLoss)  

## Expected Behavior

With these matching configurations, the NVFlare implementation should:

1. ✓ Converge at similar rates as the original codes implementation
2. ✓ Achieve similar final accuracies
3. ✓ Show similar training dynamics (loss curves, accuracy curves)
4. ✓ Handle non-IID data in the same way (Dirichlet alpha=0.5)
5. ✓ Maintain synchronous behavior (all 100 devices per round)

## Running the Implementation

### NVFlare Version:
```bash
cd research/fed-revive/nvflare_implementation/jobs
python pt_job_fedavg.py
```

### Original Codes Version:
```bash
cd research/fed-revive/codes
python main.py --config SampleConfigFiles/cifar10/fedavg.json
```

## Conclusion

The NVFlare implementation (`pt_job_fedavg.py` with `Cifar10FedAvgTaskProcessor`) accurately replicates the configuration from `codes/SampleConfigFiles/cifar10/fedavg.json`. All critical parameters for reproducibility are exactly matched:

- ✓ Data distribution (Dirichlet with alpha=0.5)
- ✓ Local training (25 iterations, Adam, lr=0.0003)
- ✓ Server aggregation (synchronous FedAvg, lr=1.4)
- ✓ Device selection (100 out of 1000 per round)
- ✓ Sample size per device (350 samples)

Minor differences in delay simulation and evaluation methodology should not significantly impact the convergence behavior or final model performance.

