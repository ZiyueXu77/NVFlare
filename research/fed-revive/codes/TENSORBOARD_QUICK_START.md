# TensorBoard Quick Start Guide

## Installation

TensorBoard is now included in `requirements.txt`. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments with TensorBoard

### 1. Enable TensorBoard in Config

Edit your config JSON file and set:
```json
{
  "wandb_flag": true
}
```

**Note**: The parameter is still called `wandb_flag` for backward compatibility, but it now enables TensorBoard.

### 2. Run Your Experiment

```bash
python main.py --config path/to/your_config.json
```

The experiment will create TensorBoard logs in: `<out_dir>/tensorboard/`

### 3. View Results in TensorBoard

While your experiment is running (or after it completes), open a new terminal and run:

```bash
tensorboard --logdir=<out_dir>/tensorboard/
```

For example, if your `out_dir` is `./experiments/my_experiment`:
```bash
tensorboard --logdir=./experiments/my_experiment/tensorboard/
```

To view multiple experiments at once:
```bash
tensorboard --logdir=./experiments/
```

### 4. Open TensorBoard in Browser

After running the command, TensorBoard will output:
```
TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
```

Open your browser and go to: **http://localhost:6006**

## Available Metrics

### Training Metrics (train/)
- `train/loss`: Training loss per round
- `train/accuracy`: Training accuracy per round
- `train/client_id`: Which client trained
- `train/time`: Simulated time

### Test Metrics (test/)
- `test/accuracy`: Global model test accuracy
- `test/loss`: Global model test loss
- `test/time`: Simulated time at evaluation

### Label-wise Metrics (label_wise/)
- `label_wise/accuracy_{i}`: Per-class accuracy for class i
- `label_wise/loss_{i}`: Per-class loss for class i

### Aggregation Metrics (aggregation/)
- `aggregation/client_{i}_staleness`: Staleness for each client
- `aggregation/avg_staleness`: Average staleness across clients

### DFKD Metrics (if using DFKD)

**Synthesis Metrics (dfkd_synthesis/):**
- `dfkd_synthesis/best_cost`: Best synthesis cost
- `dfkd_synthesis/bn_loss`: Batch normalization loss
- `dfkd_synthesis/oh_loss`: One-hot loss
- `dfkd_synthesis/adv_loss`: Adversarial loss
- `dfkd_synthesis/kl_uniform_loss`: KL divergence from uniform loss
- `dfkd_synthesis/diversity_loss`: Diversity loss
- `dfkd_synthesis/time`: Synthesis time

**Knowledge Distillation Metrics (dfkd_kd/):**
- `dfkd_kd/avg_loss`: Average KD training loss
- `dfkd_kd/min_loss`: Minimum KD training loss
- `dfkd_kd/max_loss`: Maximum KD training loss
- `dfkd_kd/num_steps`: Number of KD training steps
- `dfkd_kd/time`: KD training time

## TensorBoard Tips

### Smoothing
Use the slider in TensorBoard UI to smooth noisy curves (default is 0.6)

### Comparing Runs
TensorBoard automatically detects multiple runs in the same directory. Each run appears as a different colored line.

### Downloading Data
Click the download button in TensorBoard to export metrics as CSV or JSON

### Filtering Metrics
Use the search box in TensorBoard to filter which metrics to display

### Refresh
TensorBoard auto-refreshes every 30 seconds. You can also manually refresh with F5

## Common Issues

### Port Already in Use
If port 6006 is already in use, specify a different port:
```bash
tensorboard --logdir=<out_dir>/tensorboard/ --port=6007
```

### Can't See New Data
- TensorBoard may take a few seconds to detect new data
- Press F5 to force refresh
- Check that the experiment is writing to the correct directory

### TensorBoard Not Found
Make sure TensorBoard is installed:
```bash
pip install tensorboard
```

## Advantages Over WandB

1. **Privacy**: All data stays on your local machine
2. **No Internet Required**: Works completely offline
3. **No Account Needed**: No sign-up or API keys required
4. **Fast**: No network latency for logging
5. **Free**: No limits on number of experiments or metrics
6. **Portable**: Log files can be copied and viewed anywhere

## Example Workflow

```bash
# Terminal 1: Run experiment
python main.py --config configs/cifar10_async.json

# Terminal 2: Start TensorBoard
tensorboard --logdir=./experiments/

# Browser: Open http://localhost:6006
# Watch metrics update in real-time!
```

