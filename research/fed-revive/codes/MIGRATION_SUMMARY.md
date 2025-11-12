# Migration from WandB to TensorBoard - Summary

## Overview
This codebase has been successfully migrated from WandB (Weights & Biases) tracking to local TensorBoard logging.

## Changes Made

### 1. `requirements.txt`
- **Removed**: `wandb`
- **Added**: `tensorboard`

### 2. `server.py`
- **Method renamed**: `set_wandb()` → `set_tensorboard()`
- **Initialization**: Creates TensorBoard `SummaryWriter` in `<out_dir>/tensorboard/`
- **Logging changes**:
  - Train metrics: `wandb.log()` → `tensorboard_writer.add_scalar()`
  - Test metrics: `wandb.log()` → `tensorboard_writer.add_scalar()`
  - Aggregation metrics: `wandb.log()` → `tensorboard_writer.add_scalar()`
  - DFKD metrics: Pass `tensorboard_writer` instead of `wandb_run` to DFKD functions
- **Cleanup**: `wandb.finish()` → `tensorboard_writer.close()`

### 3. `config.py`
- **Updated comments**: Clarified that `wandb_flag` is now used for TensorBoard
- **Backward compatibility**: Config parameter names kept the same (e.g., `wandb_flag`) to avoid breaking existing configurations
- **Deprecated fields**: Marked `wandb_entity`, `wandb_project`, `wandb_run_name`, `wandb_group` as deprecated (not used with TensorBoard)

### 4. `utils/exp_state_utils.py`
- **Deprecated function**: `setup_wandb_resume()` now returns `(None, False)` and is kept for backward compatibility
- **Note**: TensorBoard doesn't require special resume setup like WandB did

### 5. `utils/dfkd_utils.py`
- **Function signature**: `perform_dfkd_with_buffer()` parameter `wandb_run` → `tensorboard_writer`
- **Logging**: Replaced `wandb_run.log()` with individual `tensorboard_writer.add_scalar()` calls
- **Metrics logged**:
  - DFKD synthesis: best_cost, bn_loss, oh_loss, adv_loss, kl_uniform_loss, diversity_loss, time
  - DFKD KD training: avg_loss, min_loss, max_loss, num_steps, time

### 6. `utils/text_dfkd_utils.py`
- **Function signature**: `perform_text_dfkd_with_buffer()` parameter `wandb_run` → `tensorboard_writer`
- **Logging**: Replaced `wandb_run.log()` with individual `tensorboard_writer.add_scalar()` calls
- **Same metrics as dfkd_utils.py**

## How to Use TensorBoard

### 1. Enable TensorBoard Logging
In your config file (JSON), set:
```json
{
  "wandb_flag": true
}
```

### 2. Run Your Experiment
```bash
python main.py --config your_config.json
```

TensorBoard logs will be saved to: `<out_dir>/tensorboard/`

### 3. View TensorBoard
Open a terminal and run:
```bash
tensorboard --logdir=<out_dir>/tensorboard/
```

Then open your browser to: `http://localhost:6006`

### 4. View Metrics
TensorBoard will show the following metric groups:
- **train/**: Training loss, accuracy, client_id, time
- **test/**: Test accuracy, loss, time
- **label_wise/**: Per-class accuracy and loss
- **aggregation/**: Client staleness metrics
- **dfkd_synthesis/**: DFKD synthesis metrics (if using DFKD)
- **dfkd_kd/**: DFKD knowledge distillation metrics (if using DFKD)

## Backward Compatibility

- **Config parameter names**: Kept the same (`wandb_flag`) to avoid breaking existing configurations
- **Deprecated function**: `setup_wandb_resume()` still exists but returns dummy values
- **Old WandB config fields**: Kept in config but marked as deprecated (ignored by TensorBoard)

## Benefits of TensorBoard

1. **Local logging**: No need for internet connection or external service
2. **Privacy**: All data stays on your machine
3. **Fast**: No network overhead for logging
4. **Free**: No API limits or account requirements
5. **Lightweight**: Built into PyTorch, no extra dependencies

## Notes

- TensorBoard logs are stored locally in `<out_dir>/tensorboard/`
- To view results from multiple experiments, use: `tensorboard --logdir=experiments/`
- TensorBoard will automatically aggregate metrics from multiple runs
- Unlike WandB, TensorBoard doesn't have automatic cloud sync, but logs are portable

