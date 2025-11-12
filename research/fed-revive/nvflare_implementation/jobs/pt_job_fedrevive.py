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
FedRevive Job for Edge Federated Learning with CIFAR10.

This script creates and runs a federated learning job using the FedRevive algorithm,
which addresses challenges in asynchronous federated learning through:
1. Knowledge distillation to handle model staleness
2. Adaptive weighting based on update staleness  
3. Buffered asynchronous aggregation

The implementation follows the FedRevive paper and the reference implementation
in the "research/fed-revive/codes" directory.
"""

import os

from processors.cifar10_fedrevive_task_processor import Cifar10FedReviveTaskProcessor
from processors.models.cifar10_model import Cifar10ConvNet

from nvflare.edge.tools.edge_fed_revive_recipe import (
    DeviceManagerConfig,
    EdgeFedReviveRecipe,
    EvaluatorConfig,
    FedReviveConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.prod_env import ProdEnv


def main():
    """
    Main function to configure and run FedRevive federated learning job.

    Configuration follows FedRevive algorithm specifications with:
    - Asynchronous aggregation with buffering
    - Knowledge distillation for staleness handling
    - Non-IID data distribution via Dirichlet sampling
    - Computational heterogeneity simulation
    """

    # ========================================
    # FL System Configuration
    # ========================================

    # Number of simulated devices
    devices_per_leaf = 10000  # Total device pool per leaf client
    device_selection_size = 200  # Devices selected for training
    num_leaf_nodes = 4  # Number of leaf nodes in the hierarchy

    # NVFlare system paths
    startup_kit_location = "/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com"
    username = "admin@nvidia.com"
    output_dir = "/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/transfer"
    dataset_root = "/tmp/nvflare/datasets/cifar10"

    # ========================================
    # FedRevive Algorithm Configuration
    # ========================================

    # Global aggregation parameters
    global_lr = 0.1  # Global learning rate for model aggregation
    num_updates_for_model = 20  # Buffer size: aggregate after this many updates
    max_model_version = 200  # Maximum number of global model versions (rounds)
    max_model_history = None  # Keep all model versions (None = unlimited)
    min_hole_to_fill = 10  # Wait for 10 updates before dispatching next model
    eval_frequency = 1  # Evaluate every global model version

    # Knowledge Distillation parameters (FedRevive-specific)
    kd_enabled = True  # Enable KD augmentation
    kd_buffer_size = 3  # Number of recent models to use as teachers
    kd_beta = 0.5  # Weight for KD (adaptive based on staleness)
    kd_temperature = 1.0  # Temperature for KD softmax
    kd_learning_rate = 0.01  # Learning rate for KD optimization
    kd_num_iters = 10  # Number of KD optimization iterations

    # ========================================
    # Local Training Configuration (FedRevive style)
    # ========================================

    # Data configuration
    subset_size = 100  # Samples per device
    data_distribution = "non_iid_dirichlet"  # Non-IID distribution
    dirichlet_alpha = 0.3  # Lower = more non-IID (0.3 is moderately non-IID)

    # Local training hyperparameters
    local_batch_size = 32  # Batch size for local training
    local_iters = 20  # Fixed number of training iterations (not epochs)
    local_lr = 0.01  # Learning rate for Adam optimizer

    # ========================================
    # Heterogeneity Simulation
    # ========================================

    # Communication delay (network latency)
    communication_delay = {"mean": 0.0, "std": 0.0}  # Mean communication delay in seconds  # Standard deviation

    # Computational heterogeneity (different device speeds)
    # Each tuple is (probability, mean_training_time)
    # This simulates fast, medium, and slow devices
    device_speed = {
        "mean": [0.25, 0.5, 1.0],  # 25% fast, 50% medium, 100% slow (relative)
        "std": [0.05, 0.1, 0.2],  # Variability within each group
    }

    # ========================================
    # Create FedRevive Recipe
    # ========================================

    print("=" * 80)
    print("Creating FedRevive Federated Learning Recipe")
    print("=" * 80)
    print(f"Algorithm: FedRevive (Async FL with Knowledge Distillation)")
    print(f"Dataset: CIFAR10")
    print(f"Data Distribution: {data_distribution} (alpha={dirichlet_alpha})")
    print(f"Total Devices: {devices_per_leaf * num_leaf_nodes}")
    print(f"Selected Devices per Round: {device_selection_size}")
    print(f"Buffer Size: {num_updates_for_model}")
    print(f"Global Learning Rate: {global_lr}")
    print(f"Local Learning Rate: {local_lr}")
    print(f"Local Iterations: {local_iters}")
    print(f"KD Enabled: {kd_enabled}")
    if kd_enabled:
        print(f"  - KD Buffer Size: {kd_buffer_size}")
        print(f"  - KD Beta: {kd_beta}")
        print(f"  - KD Temperature: {kd_temperature}")
    print("=" * 80)

    # Task processor for device training simulation
    task_processor = Cifar10FedReviveTaskProcessor(
        data_root=dataset_root,
        subset_size=subset_size,
        communication_delay=communication_delay,
        device_speed=device_speed,
        local_batch_size=local_batch_size,
        local_iters=local_iters,
        local_lr=local_lr,
        data_distribution=data_distribution,
        dirichlet_alpha=dirichlet_alpha,
    )

    # Model manager configuration (with FedRevive settings)
    model_manager_config = ModelManagerConfig(
        global_lr=global_lr,
        num_updates_for_model=num_updates_for_model,
        max_model_version=max_model_version,
        max_model_history=max_model_history,
        max_num_active_model_versions=max_model_history,
        update_timeout=500,
        staleness_weight=False,  # FedRevive uses KD instead of simple staleness weighting
    )

    # Device manager configuration
    device_manager_config = DeviceManagerConfig(
        device_selection_size=device_selection_size,
        min_hole_to_fill=min_hole_to_fill,
        device_reuse=False,  # Each device participates only once
    )

    # FedRevive-specific configuration
    fedrevive_config = FedReviveConfig(
        kd_enabled=kd_enabled,
        kd_buffer_size=kd_buffer_size,
        kd_beta=kd_beta,
        kd_temperature=kd_temperature,
        kd_learning_rate=kd_learning_rate,
        kd_num_iters=kd_num_iters,
    )

    # Evaluator configuration
    evaluator_config = EvaluatorConfig(
        torchvision_dataset={"name": "CIFAR10", "path": dataset_root}, eval_frequency=eval_frequency
    )

    # Simulation configuration
    # Calculate optimal number of workers for parallel execution
    # For maximum speed: num_workers = device_selection_size (all parallel)
    # For safety: num_workers = min(cpu_cores, device_selection_size)
    # Current: Using device_selection_size for maximum parallelism
    max_workers = min(os.cpu_count() or 32, device_selection_size * 2)  # Allow up to 2x devices for overlap
    num_workers = max_workers  # Use maximum available workers

    print(f"Parallel workers: {num_workers} (for {device_selection_size} devices)")
    print(f"Expected parallelism: ~{device_selection_size / num_workers:.1f}x batches per round")

    simulation_config = SimulationConfig(
        task_processor=task_processor,
        job_timeout=60.0,  # Increased from 20.0 for safety
        num_workers=num_workers,  # Optimized for parallel execution
        num_devices=devices_per_leaf,
    )

    # Generate the FedRevive recipe
    recipe = EdgeFedReviveRecipe(
        job_name="pt_job_fedrevive",
        model=Cifar10ConvNet(),
        model_manager_config=model_manager_config,
        device_manager_config=device_manager_config,
        fedrevive_config=fedrevive_config,
        evaluator_config=evaluator_config,
        simulation_config=simulation_config,
        custom_source_root=None,
    )

    print(f"\nExporting recipe to {output_dir}")
    recipe.export(output_dir)
    print("Recipe exported successfully!")

    # ========================================
    # Execute the Job
    # ========================================

    print("\n" + "=" * 80)
    print("Executing FedRevive Job")
    print("=" * 80)

    env = ProdEnv(startup_kit_location=startup_kit_location, username=username)
    run = recipe.execute(env)

    # ========================================
    # Display Results
    # ========================================

    print("\n" + "=" * 80)
    print("Job Execution Complete")
    print("=" * 80)
    print(f"Results: {run.get_result()}")
    print(f"Status: {run.get_status()}")
    print("=" * 80)


if __name__ == "__main__":
    exit(main())
