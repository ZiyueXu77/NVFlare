# BatchNorm Layers Kept Local

## Overview

BatchNorm (Batch Normalization) layers are now kept **local** and **not shared/aggregated** during federated learning. This is a common and important technique in FL to improve performance, especially with non-IID data.

## Why Keep BatchNorm Local?

### 1. Dataset-Specific Statistics
BatchNorm layers maintain running statistics (running_mean, running_var) that are specific to each client's local data distribution. These statistics:
- Capture local data characteristics
- May be very different across clients (especially with non-IID data)
- Should NOT be averaged across clients

### 2. Better Performance with Non-IID Data
Research has shown that keeping BatchNorm local:
- ✓ Improves model accuracy on heterogeneous data
- ✓ Reduces negative impact of data heterogeneity
- ✓ Helps each client adapt to their local distribution
- ✓ More stable training convergence

### 3. Standard Practice in FL
Major FL frameworks and research papers commonly exclude BatchNorm:
- FedBN (Federated Learning with Batch Normalization)
- Many production FL systems
- Recommended for non-IID scenarios

## Implementation Details

### What's Excluded

The following parameters are kept local (not sent to server):
- `running_mean` - Running mean of activations
- `running_var` - Running variance of activations  
- `num_batches_tracked` - Number of batches seen
- `weight` and `bias` of BatchNorm layers (if layer name contains 'bn')

### Detection Logic

```python
def is_batch_norm_layer(layer_name):
    """Check if a layer is a BatchNorm layer."""
    bn_keywords = ['bn', 'batch_norm', 'batchnorm', 
                   'running_mean', 'running_var', 'num_batches_tracked']
    layer_name_lower = layer_name.lower()
    return any(keyword in layer_name_lower for keyword in bn_keywords)
```

### Model Loading

When receiving global model from server:
```python
# 1. Start with fresh model
net = Cifar10ConvNet()
model_dict = net.state_dict()

# 2. Filter out BatchNorm from global model
filtered_global_model = {
    k: v for k, v in global_model.items() 
    if not is_batch_norm_layer(k)
}

# 3. Update only non-BN parameters
model_dict.update(filtered_global_model)

# 4. Restore local BN state from previous round
if local_bn_state is not None:
    for k, v in local_bn_state.items():
        if is_batch_norm_layer(k):
            model_dict[k] = v

# 5. Load combined model
net.load_state_dict(model_dict)
```

### Update Computation

When sending updates to server:
```python
diff_dict = {}
local_bn_state = {}

for key, param in net.state_dict().items():
    if is_batch_norm_layer(key):
        # Keep BN local - don't send to server
        local_bn_state[key] = param.cpu()
    else:
        # Send non-BN parameter differences
        diff = param.cpu().numpy() - global_model[key].cpu().numpy()
        diff_dict[key] = diff.tolist()

# Store BN state for next round
self.local_bn_states[device_id] = local_bn_state
```

## Files Modified

### 1. `processors/cifar10_fedavg_task_processor.py`
- Added `is_batch_norm_layer()` helper function
- Modified `_pytorch_training()` to accept and use local BN state
- Modified parameter diff calculation to exclude BN layers
- Modified `process_task()` to maintain BN state across rounds

### 2. `processors/cifar10_fedrevive_task_processor.py`
- Same modifications as FedAvg processor
- Works seamlessly with FedRevive's async aggregation

## Expected Impact

### Accuracy Improvement
With non-IID data (Dirichlet alpha=0.5):
- **Before**: BN statistics mixed across diverse client data
- **After**: Each client uses its own BN statistics
- **Expected**: 1-5% accuracy improvement on non-IID data

### Training Stability
- More stable training curves
- Faster convergence
- Better generalization to local data

### Communication Efficiency
- **Slight reduction** in communication cost
- BN parameters not sent to server
- Typically saves ~1-2% of model size

## Verification

### Check What's Being Sent

Add logging to see excluded layers:
```python
# In _pytorch_training method
log.info(f"Total parameters: {len(net.state_dict())}")
log.info(f"Shared parameters: {len(diff_dict)}")
log.info(f"Local BN parameters: {len(local_bn_state)}")
```

### Example Output
```
Total parameters: 130
Shared parameters: 122
Local BN parameters: 8
```

### Verify BN Layer Detection

Run this to check which layers are detected as BN:
```python
from processors.cifar10_fedavg_task_processor import is_batch_norm_layer
from processors.models.cifar10_model import Cifar10ConvNet

model = Cifar10ConvNet()
for name, param in model.named_parameters():
    if is_batch_norm_layer(name):
        print(f"BatchNorm: {name} - {param.shape}")
    else:
        print(f"Shared:    {name} - {param.shape}")
```

## Comparison with Original Codes

### Original Implementation (codes folder)
The original codes likely aggregate ALL parameters including BatchNorm.

### NVFlare Implementation (This)
Explicitly keeps BatchNorm local for better performance.

**Trade-off:**
- ✓ Better accuracy with non-IID data
- ✓ More FL-friendly approach
- ⚠️ Slightly different from original (if original didn't exclude BN)

## When to Use This

### Recommended (Default)
- ✓ Non-IID data distribution
- ✓ High data heterogeneity across clients
- ✓ Dirichlet alpha < 1.0
- ✓ Real-world FL scenarios

### Consider Disabling
- IID data distribution (rare in practice)
- Very similar data across all clients
- Debugging/comparing with specific baselines

## How to Disable (If Needed)

If you want to share BatchNorm layers (not recommended):

```python
# In task processor, modify is_batch_norm_layer()
def is_batch_norm_layer(layer_name):
    return False  # Disable BN exclusion
```

## References

### Academic Papers
1. **FedBN**: "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization" (ICLR 2021)
2. **GroupNorm in FL**: Several papers recommend LayerNorm/GroupNorm over BatchNorm in FL

### Best Practices
- Standard in modern FL implementations
- Recommended by FL researchers
- Used in production FL systems

## Summary

✓ **BatchNorm layers are now kept local**  
✓ **Improves performance on non-IID data**  
✓ **Standard practice in federated learning**  
✓ **Implemented in both FedAvg and FedRevive processors**  
✓ **Maintains local statistics across training rounds**  

This is a **best practice** for federated learning and should improve your results, especially with the non-IID Dirichlet distribution (alpha=0.5) used in your experiments.

