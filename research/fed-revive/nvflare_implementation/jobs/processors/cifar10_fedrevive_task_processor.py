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
FedRevive Task Processor for CIFAR10.

This task processor implements the FedRevive local training scheme:
- Uses Adam optimizer (instead of SGD)
- Trains for a fixed number of iterations (instead of epochs)
- Supports non-IID data distribution via Dirichlet sampling
- Includes computational heterogeneity simulation
"""

import logging
import os
import random
import time

import filelock
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, from_dict
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse

from .models.cifar10_model import Cifar10ConvNet

log = logging.getLogger(__name__)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def is_batch_norm_layer(layer_name):
    """
    Check if a layer is a BatchNorm layer.
    BatchNorm layers contain running_mean, running_var, num_batches_tracked, weight/bias of BN layers.
    """
    bn_keywords = ["bn", "batch_norm", "batchnorm", "running_mean", "running_var", "num_batches_tracked"]
    layer_name_lower = layer_name.lower()
    # Check for explicit BN indicators or if it's a BN parameter
    return any(keyword in layer_name_lower for keyword in bn_keywords)


class Cifar10FedReviveTaskProcessor(DeviceTaskProcessor):
    """
    Task processor implementing FedRevive's local training scheme for CIFAR10.

    Key differences from standard task processor:
    - Uses Adam optimizer (better for async FL)
    - Fixed number of iterations instead of epochs
    - Supports non-IID data via Dirichlet distribution
    - Simulates computational heterogeneity
    """

    def __init__(
        self,
        data_root: str,
        subset_size: int,
        communication_delay: dict,
        device_speed: dict,
        local_batch_size: int = 32,
        local_iters: int = 20,  # Fixed number of iterations (FedRevive style)
        local_lr: float = 0.01,
        data_distribution: str = "non_iid_dirichlet",  # iid or non_iid_dirichlet
        dirichlet_alpha: float = 0.3,  # Dirichlet concentration parameter (lower = more non-IID)
    ):
        """
        Initialize the FedRevive task processor.

        Args:
            data_root: Root directory for CIFAR10 data
            subset_size: Number of samples per device
            communication_delay: Dict with 'mean' and 'std' for communication delay
            device_speed: Dict with 'mean' (list) and 'std' (list) for device speed heterogeneity
            local_batch_size: Batch size for local training (default: 32)
            local_iters: Number of local training iterations (default: 20)
            local_lr: Learning rate for Adam optimizer (default: 0.01)
            data_distribution: Type of data distribution ('iid' or 'non_iid_dirichlet')
            dirichlet_alpha: Dirichlet concentration parameter for non-IID sampling
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

        # Device speed type for computational heterogeneity
        mean_speed = self.device_speed.get("mean")
        if isinstance(mean_speed, list) and len(mean_speed) > 0:
            self.device_speed_type = random.randint(0, len(mean_speed) - 1)
        else:
            self.device_speed_type = 0

        # Cache for dataset (loaded once per device)
        self.train_subset = None
        self.train_loader = None

    def setup(self, job: JobResponse) -> None:
        """Setup the task processor for a new job."""
        # Pre-load and cache the dataset for this device
        self._prepare_dataset()

    def shutdown(self) -> None:
        """Clean up resources."""
        self.train_subset = None
        self.train_loader = None

    def _prepare_dataset(self):
        """
        Prepare dataset with non-IID distribution if specified.
        This follows the FedRevive data distribution strategy.
        """
        # Data loading with transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR10 normalization
            ]
        )

        # Add file lock to prevent multiple simultaneous downloads
        lock_file = os.path.join(self.data_root, "cifar10.lock")
        with filelock.FileLock(lock_file):
            train_set = datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=transform)

        # Generate device-specific seed for reproducibility
        device_id = self.device.device_id if self.device else "0"
        device_seed = hash(str(device_id)) % (2**32)
        rng = np.random.RandomState(device_seed)

        if self.data_distribution == "non_iid_dirichlet":
            # Non-IID distribution using Dirichlet sampling (FedRevive approach)
            indices = self._sample_dirichlet_indices(train_set, rng)
        else:
            # IID distribution (random sampling)
            indices = list(range(len(train_set)))
            rng.shuffle(indices)
            indices = indices[: self.subset_size]

        # Create subset and dataloader
        self.train_subset = Subset(train_set, indices)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_subset, batch_size=self.local_batch_size, shuffle=True, num_workers=2
        )

        log.info(
            f"Device {device_id}: Prepared dataset with {len(self.train_subset)} samples "
            f"({self.data_distribution} distribution)"
        )

    def _sample_dirichlet_indices(self, train_set, rng):
        """
        Sample indices using Dirichlet distribution for non-IID data.

        This implements the Dirichlet-based non-IID sampling from FedRevive,
        where each device gets a skewed class distribution.
        """
        num_classes = 10  # CIFAR10 has 10 classes

        # Get all labels
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
        Perform local training using FedRevive's training scheme.

        Key aspects:
        - Uses Adam optimizer (better for async FL)
        - Trains for fixed number of iterations
        - Tracks training metrics
        - Keeps BatchNorm layers local (not shared/aggregated)
        """
        # Network loading
        net = Cifar10ConvNet()

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

        # Use Adam optimizer (FedRevive default)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.local_lr)
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop for fixed number of iterations
        iter_count = 0
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        while iter_count < self.local_iters:
            for inputs, labels in self.train_loader:
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

        # Calculate metrics
        avg_loss = total_loss / iter_count if iter_count > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        device_id = self.device.device_id if self.device else "unknown"
        log.info(
            f"Device {device_id}: Completed {iter_count} iterations, " f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        # Calculate model parameter diff, excluding BatchNorm layers
        diff_dict = {}
        local_bn_state = {}

        for key, param in net.state_dict().items():
            if is_batch_norm_layer(key):
                # Store local BN state but don't send to server
                local_bn_state[key] = param.cpu()
                log.debug(f"Keeping BatchNorm layer local: {key}")
            else:
                # Send non-BN parameter differences to server
                numpy_param = param.cpu().numpy() - global_model[key].cpu().numpy()
                # Convert numpy array to list for serialization
                diff_dict[key] = numpy_param.tolist()

        # Store local BN state for this device (for next round)
        if not hasattr(self, "local_bn_states"):
            self.local_bn_states = {}
        self.local_bn_states[device_id] = local_bn_state

        return diff_dict

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
        # Parse task data
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

        # Convert global_model to tensors and move to device
        global_model = {k: torch.tensor(v).to(DEVICE) for k, v in global_model.items()}

        # Get local BN state if available from previous training
        device_id = self.device.device_id if self.device else "unknown"
        local_bn_state = None
        if hasattr(self, "local_bn_states") and device_id in self.local_bn_states:
            local_bn_state = self.local_bn_states[device_id]
            log.info(f"Device {device_id}: Using local BatchNorm state from previous round")

        # Perform local training (BN layers stay local)
        diff_dict = self._pytorch_training(global_model, local_bn_state)

        # Create result DXO
        result_dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data={"dict": diff_dict})

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

        total_delay = delay_speed + delay_comm

        # Log the delay
        device_id = self.device.device_id if self.device else "unknown"
        self.logger.info(
            f"Device {device_id} total delay: {total_delay:.2f}s "
            f"(training: {delay_speed:.2f}s, comm: {delay_comm:.2f}s)"
        )

        # Simulate the delay
        time.sleep(total_delay)

        return result_dxo.to_dict()
