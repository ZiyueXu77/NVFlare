# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FedAvg Task Processor for CIFAR10.

This task processor implements the standard FedAvg local training scheme:
- Uses Adam optimizer (matching codes/cifar10/fedavg.json)
- Trains for a fixed number of iterations (25 iterations)
- Supports non-IID data distribution via Dirichlet sampling
- Includes computational heterogeneity simulation
- Synchronous aggregation (all clients per round)
"""

import gc
import logging
import os
import pickle
import random
import time
from collections import defaultdict

import filelock
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_dict
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

from .models.resnet_18 import ResNet18Local

log = logging.getLogger(__name__)
# Use GPU to offload computation from system RAM
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Limit PyTorch CPU threads to reduce memory overhead per worker
# Each worker spawns multiple threads which consume memory
torch.set_num_threads(2)  # Limit to 2 threads per worker

# Base TensorBoard log directory
TENSORBOARD_LOG_DIR = "/tmp/nvflare/tensorboard_logs/fedavg_timing"


def is_batch_norm_layer(layer_name):
    """
    Check if a layer is a BatchNorm layer.
    BatchNorm layers contain running_mean, running_var, num_batches_tracked, weight/bias of BN layers.
    """
    bn_keywords = ["bn", "batch_norm", "batchnorm", "running_mean", "running_var", "num_batches_tracked"]
    layer_name_lower = layer_name.lower()
    # Check for explicit BN indicators or if it's a BN parameter
    return any(keyword in layer_name_lower for keyword in bn_keywords)


class Cifar10FedAvgTaskProcessor(DeviceTaskProcessor):
    """
    Task processor implementing standard FedAvg local training scheme for CIFAR10.

    Configuration matches codes/SampleConfigFiles/cifar10/fedavg.json:
    - local_learning_rate: 0.0003
    - local_batch_size: 32
    - local_iters: 25
    - local_optimizer: adam
    - dirichlet_alpha: 0.5
    - client_data_size: 350
    """

    def __init__(
        self,
        data_root: str,
        subset_size: int,
        communication_delay: dict,
        device_speed: dict,
        local_batch_size: int = 32,
        local_iters: int = 25,  # Fixed number of iterations (FedAvg from codes)
        local_lr: float = 0.0003,  # Learning rate from fedavg.json
        data_distribution: str = "non_iid_dirichlet",  # iid or non_iid_dirichlet
        dirichlet_alpha: float = 0.5,  # Dirichlet concentration parameter (from fedavg.json)
        dataset_splits_file: str = None,  # Pre-computed dataset splits file
    ):
        """
        Initialize the FedAvg task processor.

        Args:
            data_root: Root directory for CIFAR10 data
            subset_size: Number of samples per device (350 in fedavg.json)
            communication_delay: Dict with 'mean' and 'std' for communication delay
            device_speed: Dict with 'mean' (list) and 'std' (list) for device speed heterogeneity
            local_batch_size: Batch size for local training (default: 32)
            local_iters: Number of local training iterations (default: 25)
            local_lr: Learning rate for Adam optimizer (default: 0.0003)
            data_distribution: Type of data distribution ('iid' or 'non_iid_dirichlet')
            dirichlet_alpha: Dirichlet concentration parameter for non-IID sampling (default: 0.5)
            dataset_splits_file: Path to pre-computed dataset splits file (optional, for faster initialization)
        """
        DeviceTaskProcessor.__init__(self)
        self.data_root = data_root
        self.subset_size = subset_size
        self.communication_delay = communication_delay
        self.device_speed = device_speed
        self.local_batch_size = local_batch_size
        self.local_iters = local_iters
        self.local_lr = local_lr
        self.data_distribution = data_distribution
        self.dirichlet_alpha = dirichlet_alpha
        self.dataset_splits_file = dataset_splits_file

        # Device speed type for computational heterogeneity
        mean_speed = self.device_speed.get("mean")
        if isinstance(mean_speed, list) and len(mean_speed) > 0:
            self.device_speed_type = random.randint(0, len(mean_speed) - 1)
        else:
            self.device_speed_type = 0

        # Cache for dataset (loaded once per device)
        self.train_subset = None
        self.train_loader = None

        # Timing statistics for TensorBoard logging
        self.timing_stats = defaultdict(list)
        self.task_counter = 0  # Track number of tasks processed
        self._tensorboard_writer = None  # Lazy initialization per device

    def _get_tensorboard_writer(self):
        """Get or initialize device-specific TensorBoard writer."""
        if self._tensorboard_writer is None:
            device_id = self.device.device_id if self.device else "unknown"
            # Create device-specific log directory
            device_log_dir = os.path.join(TENSORBOARD_LOG_DIR, f"device_{device_id}")
            os.makedirs(device_log_dir, exist_ok=True)
            self._tensorboard_writer = SummaryWriter(device_log_dir)
            log.info(f"Device {device_id}: TensorBoard logging initialized at: {device_log_dir}")
            log.info(f"View with: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
        return self._tensorboard_writer

    def setup(self, job: JobResponse) -> None:
        """Setup the task processor for a new job."""
        # Pre-load and cache the dataset for this device
        self._prepare_dataset()

    def shutdown(self) -> None:
        """Clean up resources and free system memory."""
        # Flush and close TensorBoard writer
        if hasattr(self, "_tensorboard_writer") and self._tensorboard_writer is not None:
            self._tensorboard_writer.flush()
            self._tensorboard_writer.close()
            log.info("TensorBoard logs flushed and closed on shutdown")

        # Clear dataset references
        if hasattr(self, "train_subset"):
            del self.train_subset
        if hasattr(self, "train_loader"):
            del self.train_loader
        if hasattr(self, "train_indices"):
            del self.train_indices
        if hasattr(self, "local_bn_states"):
            del self.local_bn_states

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_dataset(self):
        """
        Prepare dataset with non-IID distribution if specified.
        This follows the FedAvg data distribution strategy from codes.

        Optimized for system memory: dataset is created per-task rather than held in memory.
        Can load pre-computed splits from file for fast initialization.
        """
        setup_start = time.time()

        # Data loading with transforms (CIFAR10 standard normalization)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

        # Get device ID
        device_id = self.device.device_id if self.device else "0"

        # Try to load from pre-computed splits file first
        indices = None
        if self.dataset_splits_file and os.path.exists(self.dataset_splits_file):
            try:
                load_start = time.time()
                with open(self.dataset_splits_file, "rb") as f:
                    split_data = pickle.load(f)

                # Verify configuration matches
                if (
                    split_data.get("subset_size") == self.subset_size
                    and split_data.get("data_distribution") == self.data_distribution
                    and (
                        self.data_distribution != "non_iid_dirichlet"
                        or split_data.get("dirichlet_alpha") == self.dirichlet_alpha
                    )
                ):

                    # Convert device_id to int for lookup
                    device_id_int = (
                        int(device_id)
                        if isinstance(device_id, str) and device_id.isdigit()
                        else hash(str(device_id)) % split_data["num_devices"]
                    )

                    if device_id_int in split_data["device_splits"]:
                        indices = split_data["device_splits"][device_id_int]
                        load_time = time.time() - load_start
                        log.info(
                            f"Device {device_id}: Loaded pre-computed indices from {self.dataset_splits_file} "
                            f"({len(indices)} samples) in {load_time:.3f}s"
                        )
                    else:
                        log.warning(
                            f"Device {device_id} (mapped to {device_id_int}) not found in splits file, "
                            f"will compute on-the-fly"
                        )
                else:
                    log.warning(f"Device {device_id}: Split file configuration mismatch, will compute on-the-fly")
                    log.warning(
                        f"  Expected: subset_size={self.subset_size}, distribution={self.data_distribution}, "
                        f"alpha={self.dirichlet_alpha}"
                    )
                    log.warning(
                        f"  Got: subset_size={split_data.get('subset_size')}, "
                        f"distribution={split_data.get('data_distribution')}, "
                        f"alpha={split_data.get('dirichlet_alpha')}"
                    )
            except Exception as e:
                log.error(f"Device {device_id}: Error loading splits file: {e}, will compute on-the-fly")
                indices = None

        # Fall back to on-the-fly computation if no pre-computed splits
        if indices is None:
            log.info(f"Device {device_id}: Computing dataset indices on-the-fly...")

            # Add file lock to prevent multiple simultaneous downloads
            lock_file = os.path.join(self.data_root, "cifar10.lock")
            with filelock.FileLock(lock_file):
                train_set = datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=transform)

            # Generate device-specific seed for reproducibility
            device_seed = hash(str(device_id)) % (2**32)
            rng = np.random.RandomState(device_seed)

            if self.data_distribution == "non_iid_dirichlet":
                # Non-IID distribution using Dirichlet sampling
                indices = self._sample_dirichlet_indices(train_set, rng)
            else:
                # IID distribution (random sampling)
                indices = list(range(len(train_set)))
                rng.shuffle(indices)
                indices = indices[: self.subset_size]

            # Free memory: don't keep train_set in memory
            del train_set
            gc.collect()

        # Store only indices instead of full subset to save memory
        self.train_indices = indices
        self.transform = transform

        setup_time = time.time() - setup_start

        log.info(
            f"Device {device_id}: Prepared dataset indices for {len(self.train_indices)} samples "
            f"({self.data_distribution} distribution, alpha={self.dirichlet_alpha}) in {setup_time:.3f}s"
        )

        # Log to TensorBoard
        self._get_tensorboard_writer().add_scalar("timing/dataset_setup", setup_time, self.task_counter)

    def _sample_dirichlet_indices(self, train_set, rng):
        """
        Sample indices using Dirichlet distribution for non-IID data.

        This implements the Dirichlet-based non-IID sampling used in the codes,
        where each device gets a skewed class distribution.

        Optimized to avoid loading all labels into memory at once.
        """
        num_classes = 10  # CIFAR10 has 10 classes

        # Memory-efficient: use dataset.targets directly (already in memory)
        # instead of iterating through all samples
        if hasattr(train_set, "targets"):
            all_labels = np.array(train_set.targets)
        else:
            # Fallback for wrapped datasets
            all_labels = np.array([train_set[i][1] for i in range(len(train_set))])

        # Create indices for each class
        class_indices = {k: np.where(all_labels == k)[0] for k in range(num_classes)}

        # Sample class proportions from Dirichlet distribution
        class_probs = rng.dirichlet([self.dirichlet_alpha] * num_classes)

        # Determine number of samples per class based on Dirichlet probabilities
        samples_per_class = (class_probs * self.subset_size).astype(int)

        # Adjust to ensure exactly subset_size samples
        diff = self.subset_size - samples_per_class.sum()
        if diff > 0:
            # Add remaining samples to classes with highest probability
            top_classes = np.argsort(class_probs)[-diff:]
            for cls in top_classes:
                samples_per_class[cls] += 1
        elif diff < 0:
            # Remove excess samples from classes with lowest probability
            bottom_classes = np.argsort(class_probs)[: abs(diff)]
            for cls in bottom_classes:
                samples_per_class[cls] = max(0, samples_per_class[cls] - 1)

        # Sample indices from each class
        selected_indices = []
        for cls in range(num_classes):
            n_samples = samples_per_class[cls]
            if n_samples > 0:
                cls_indices = class_indices[cls]
                if len(cls_indices) > 0:
                    # Sample with replacement if needed
                    sampled = rng.choice(cls_indices, size=min(n_samples, len(cls_indices)), replace=False)
                    selected_indices.extend(sampled.tolist())

        # Shuffle the selected indices
        rng.shuffle(selected_indices)

        return selected_indices[: self.subset_size]

    def _pytorch_training(self, global_model, local_bn_state=None):
        """
        Perform local training using FedAvg's training scheme.

        Key aspects (matching codes/fedavg.json):
        - Uses Adam optimizer with lr=0.0003
        - Trains for fixed number of iterations (25)
        - Tracks training metrics
        - Keeps BatchNorm layers local (not shared/aggregated)

        Memory optimized: Recreates dataset on-demand instead of keeping in memory.
        """
        device_id = self.device.device_id if self.device else "unknown"

        # Time: Data loading
        data_load_start = time.time()
        train_set = datasets.CIFAR10(root=self.data_root, train=True, download=False, transform=self.transform)
        train_subset = Subset(train_set, self.train_indices)
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=self.local_batch_size,
            shuffle=True,
            num_workers=0,  # Use main process to avoid subprocess overhead
        )
        data_load_time = time.time() - data_load_start

        # Time: Model initialization
        model_init_start = time.time()
        net = ResNet18Local()

        # Load global model but keep local BatchNorm layers
        model_dict = net.state_dict()

        # Filter out BatchNorm layers from global model
        filtered_global_model = {k: v for k, v in global_model.items() if not is_batch_norm_layer(k)}

        # Update model with non-BN parameters from global model
        model_dict.update(filtered_global_model)

        # If we have local BN state from previous training, use it
        if local_bn_state is not None:
            for k, v in local_bn_state.items():
                if is_batch_norm_layer(k) and k in model_dict:
                    model_dict[k] = v

        net.load_state_dict(model_dict)
        net.to(DEVICE)
        net.train()

        # Use Adam optimizer (FedAvg default from codes)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.local_lr)
        criterion = torch.nn.CrossEntropyLoss()

        model_init_time = time.time() - model_init_start

        # Time: Training loop
        training_start = time.time()
        iter_count = 0
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        while iter_count < self.local_iters:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Forward pass
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                iter_count += 1
                if iter_count >= self.local_iters:
                    break

        training_time = time.time() - training_start

        # Calculate metrics
        avg_loss = total_loss / iter_count if iter_count > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        log.info(
            f"Device {device_id}: Completed {iter_count} iterations in {training_time:.3f}s, "
            f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        # Time: Model diff computation
        diff_compute_start = time.time()
        diff_dict = {}
        local_bn_state = {}

        for key, param in net.state_dict().items():
            if is_batch_norm_layer(key):
                # Store local BN state
                local_bn_state[key] = param.cpu()
            # Send parameter differences to server
            numpy_param = param.cpu().numpy() - global_model[key].cpu().numpy()
            diff_dict[key] = numpy_param

        # Store local BN state for this device (for next round)
        if not hasattr(self, "local_bn_states"):
            self.local_bn_states = {}
        self.local_bn_states[device_id] = local_bn_state

        diff_compute_time = time.time() - diff_compute_start

        # Log timing statistics to TensorBoard
        writer = self._get_tensorboard_writer()
        writer.add_scalar("timing/data_loading", data_load_time, self.task_counter)
        writer.add_scalar("timing/model_initialization", model_init_time, self.task_counter)
        writer.add_scalar("timing/training", training_time, self.task_counter)
        writer.add_scalar("timing/diff_computation", diff_compute_time, self.task_counter)

        # Log training metrics
        writer.add_scalar("training/loss", avg_loss, self.task_counter)
        writer.add_scalar("training/accuracy", accuracy, self.task_counter)

        # Explicitly clean up to free system memory
        del net
        del optimizer
        del train_loader
        del train_subset
        del train_set

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection to free system memory
        gc.collect()

        # Return diff_dict along with timing info
        timing_info = {
            "data_load_time": data_load_time,
            "model_init_time": model_init_time,
            "training_time": training_time,
            "diff_compute_time": diff_compute_time,
        }

        return diff_dict, timing_info

    def process_task(self, task: TaskResponse) -> dict:
        """
        Process a training task.

        This method:
        1. Receives global model from server
        2. Performs local training
        3. Calculates model updates
        4. Simulates delays (communication + computation)
        5. Returns updates to server
        """
        task_start_time = time.time()
        device_id = self.device.device_id if self.device else "unknown"

        # Time: Model receiving and parsing
        model_receive_start = time.time()
        task_data = task.task_data
        assert isinstance(task_data, dict)
        model = from_dict(task_data)
        if not isinstance(model, DXO):
            self.logger.error(f"expect model to be DXO but got {type(model)}")
            raise ValueError("bad model data")

        if model.data_kind != DataKind.WEIGHTS:
            self.logger.error(f"expect model data kind to be {DataKind.WEIGHTS} but got {model.data_kind}")
            raise ValueError("bad model data kind")

        global_model = model.data
        if not isinstance(global_model, dict):
            self.logger.error(f"expect global_model to be dict but got {type(global_model)}")
            raise ValueError("bad global model")

        # Extract round information from DXO metadata
        current_round = model.get_meta_prop(MetaKey.CURRENT_ROUND, default=None)
        total_rounds = model.get_meta_prop(MetaKey.TOTAL_ROUNDS, default=None)

        if current_round is not None:
            log.info(f"Device {device_id}: Processing task for round {current_round}/{total_rounds}")

        # Convert global_model to tensors and move to device
        global_model = {k: torch.tensor(v).to(DEVICE) for k, v in global_model.items()}

        model_receive_time = time.time() - model_receive_start

        # Get local BN state if available from previous training
        local_bn_state = None
        if hasattr(self, "local_bn_states") and device_id in self.local_bn_states:
            local_bn_state = self.local_bn_states[device_id]
            log.info(f"Device {device_id}: Using local BatchNorm state from previous round")

        # Perform local training (BN layers stay local)
        diff_dict, training_timing = self._pytorch_training(global_model, local_bn_state)

        # Time: Create result DXO
        result_create_start = time.time()
        result_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data={"dict": diff_dict})
        result_create_time = time.time() - result_create_start

        # Simulate computational heterogeneity and communication delay
        # Device speed delay (training time)
        mean_speed = self.device_speed.get("mean", [1.0])
        std_speed = self.device_speed.get("std", [0.0])
        if isinstance(mean_speed, list) and len(mean_speed) > self.device_speed_type:
            delay_speed = random.gauss(mean_speed[self.device_speed_type], std_speed[self.device_speed_type])
        else:
            delay_speed = random.gauss(
                mean_speed[0] if isinstance(mean_speed, list) else mean_speed,
                std_speed[0] if isinstance(std_speed, list) else std_speed,
            )
        delay_speed = max(0, delay_speed)

        # Communication delay
        mean_comm = self.communication_delay.get("mean", 0.1)
        std_comm = self.communication_delay.get("std", 0.02)
        delay_comm = random.gauss(mean_comm, std_comm)
        delay_comm = max(0, delay_comm)

        total_simulated_delay = delay_speed + delay_comm

        # Simulate the delay
        time.sleep(total_simulated_delay)

        # Calculate total task time
        total_task_time = time.time() - task_start_time

        # Calculate actual computation time (without simulated delays)
        actual_computation_time = (
            model_receive_time
            + training_timing["data_load_time"]
            + training_timing["model_init_time"]
            + training_timing["training_time"]
            + training_timing["diff_compute_time"]
            + result_create_time
        )

        # Framework overhead (time not accounted for in measured components)
        framework_overhead = total_task_time - actual_computation_time - total_simulated_delay

        # Calculate task end time
        task_end_time = time.time()

        # Log all timing information to TensorBoard
        writer = self._get_tensorboard_writer()
        writer.add_scalar("timing/model_receive", model_receive_time, self.task_counter)
        writer.add_scalar("timing/result_creation", result_create_time, self.task_counter)
        writer.add_scalar("timing/simulated_delay", total_simulated_delay, self.task_counter)
        writer.add_scalar("timing/framework_overhead", framework_overhead, self.task_counter)
        writer.add_scalar("timing/actual_computation", actual_computation_time, self.task_counter)
        writer.add_scalar("timing/total_task_time", total_task_time, self.task_counter)

        # Log timestamps for wall-clock time analysis
        writer.add_scalar("timestamp/task_start", task_start_time, self.task_counter)
        writer.add_scalar("timestamp/task_end", task_end_time, self.task_counter)

        # Log round information
        if current_round is not None:
            writer.add_scalar("metadata/round", current_round, self.task_counter)
        if total_rounds is not None:
            writer.add_scalar("metadata/total_rounds", total_rounds, self.task_counter)

        # Log detailed breakdown
        self.logger.info(f"Device {device_id} Task {self.task_counter} Timing Breakdown:")
        self.logger.info(f"  Model Receive:      {model_receive_time:.3f}s")
        self.logger.info(f"  Data Loading:       {training_timing['data_load_time']:.3f}s")
        self.logger.info(f"  Model Init:         {training_timing['model_init_time']:.3f}s")
        self.logger.info(f"  Training:           {training_timing['training_time']:.3f}s")
        self.logger.info(f"  Diff Computation:   {training_timing['diff_compute_time']:.3f}s")
        self.logger.info(f"  Result Creation:    {result_create_time:.3f}s")
        self.logger.info(
            f"  Simulated Delay:    {total_simulated_delay:.3f}s (speed: {delay_speed:.3f}s, comm: {delay_comm:.3f}s)"
        )
        self.logger.info(f"  Framework Overhead: {framework_overhead:.3f}s")
        self.logger.info(f"  TOTAL:              {total_task_time:.3f}s")
        self.logger.info(
            f"  Pure Computation:   {actual_computation_time:.3f}s ({actual_computation_time/total_task_time*100:.1f}% of total)"
        )

        # Increment task counter for next task
        self.task_counter += 1

        # Flush TensorBoard writer periodically
        if self.task_counter % 10 == 0:
            self._get_tensorboard_writer().flush()

        # Clean up tensors to free system memory
        del global_model
        if local_bn_state is not None:
            del local_bn_state

        # Force garbage collection after each task
        gc.collect()

        return result_dxo.to_dict()
