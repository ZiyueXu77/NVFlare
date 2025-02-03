# Asynchronized Federated Learning with CIFAR-10

This example illustrates the asynchronized federated learning with CIFAR-10 training.

## Install requirements

Install required packages for training
```
pip install --upgrade pip
pip install -r ./requirements.txt
```

## Download the CIFAR-10 dataset 
To speed up the following experiments, first download the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:
```
./prepare_data.sh
```

## Run simulated FL experiments
```commandline
python cifar10_sync_fedavg.py 
```
