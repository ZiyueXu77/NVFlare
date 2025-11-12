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
FedAvg Job for Edge Federated Learning with CIFAR10.

This script creates and runs a federated learning job using the standard FedAvg algorithm.
The configuration matches the reference implementation in "research/fed-revive/codes/SampleConfigFiles/cifar10/fedavg.json"
to enable reproducible results comparison.

Key FedAvg Parameters (from codes/cifar10/fedavg.json):
- algorithm: sync_fl (synchronous FedAvg)
- num_clients: 1000
- client_data_size: 350 samples per client
- nb_of_active_jobs: 100 (clients selected per round)
- local_batch_size: 32
- local_iters: 25 (fixed iterations, not epochs)
- local_learning_rate: 0.0003
- local_optimizer: adam
- server_lr: 1.4
- dirichlet_alpha: 0.5 (non-IID distribution)
"""

import os

import torch.nn as nn
from processors.cifar10_fedavg_task_processor_with_timing import Cifar10FedAvgTaskProcessor
from processors.models.resnet_18 import ResNet18Global

from nvflare.edge.tools.edge_fed_buff_recipe import (
    DeviceManagerConfig,
    EdgeFedBuffRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.prod_env import ProdEnv


def main():
    """
    Main function to configure and run FedAvg federated learning job.

    Configuration matches codes/SampleConfigFiles/cifar10/fedavg.json:
    - Synchronous aggregation (all selected clients per round)
    - 1000 total devices, 100 devices selected per round
    - 350 samples per device
    - Non-IID data distribution via Dirichlet sampling (alpha=0.5)
    - Computational heterogeneity simulation
    """

    # ========================================
    # FL System Configuration
    # ========================================

    # Number of simulated devices (matching codes configuration)
    device_selection_size = 100  # Devices selected for training per round (nb_of_active_jobs)
    num_leaf_nodes = 20  # Number of leaf nodes in the hierarchy
    devices_per_leaf = int(1000 / num_leaf_nodes)  # Total device pool per leaf client (num_clients from fedavg.json)

    # NVFlare system paths
    startup_kit_location = "/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com"
    username = "admin@nvidia.com"
    output_dir = "/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/transfer"
    dataset_root = "/tmp/nvflare/datasets/cifar10"

    # ========================================
    # FedAvg Algorithm Configuration
    # ========================================

    # Global aggregation parameters (matching fedavg.json)
    global_lr = 1.4  # Global learning rate for model aggregation (server_lr from fedavg.json)

    # For synchronous FedAvg: set buffer size = device selection size
    # This makes the server wait for ALL selected devices before aggregating
    num_updates_for_model = device_selection_size  # Synchronous: wait for all 100 devices

    max_model_version = 10  # Maximum number of global model versions (rounds)
    max_model_history = None  # Keep all model versions (None = unlimited)
    min_hole_to_fill = device_selection_size  # Wait for all devices (synchronous behavior)
    eval_frequency = 1  # Evaluate every global model version

    # ========================================
    # Local Training Configuration (FedAvg from codes)
    # ========================================

    # Data configuration (matching fedavg.json)
    subset_size = 350  # Samples per device (client_data_size from fedavg.json)
    data_distribution = "non_iid_dirichlet"  # Non-IID distribution
    dirichlet_alpha = 0.5  # Lower = more non-IID (0.5 from fedavg.json)

    # Local training hyperparameters (matching fedavg.json)
    # Note: If system memory is still an issue, reduce local_batch_size (e.g., to 16)
    local_batch_size = 32  # Batch size for local training (reduce to 16 if OOM persists)
    local_iters = 25  # Fixed number of training iterations (not epochs)
    local_lr = 0.0003  # Learning rate for Adam optimizer

    # ========================================
    # Heterogeneity Simulation
    # ========================================

    # Communication delay (from fedavg.json: download + upload delays)
    communication_delay = {
        "mean": 0.1,  # Mean communication delay in seconds (from download_delay)
        "std": 0.2,  # Standard deviation
    }

    # Computational heterogeneity (from fedavg.json: localtrain_delay)
    # Each tuple is (probability, mean_training_time)
    # This simulates fast, medium, and slow devices
    device_speed = {
        "mean": [1.0, 1.3, 1.6],  # 25% fast, 50% medium, 25% slow (relative)
        "std": [0.2, 0.2, 0.2],  # Variability within each group
    }

    # For testing: disable delays
    communication_delay = {"mean": 0.0, "std": 0.0}
    device_speed = {"mean": [0.0], "std": [0.0]}

    # ========================================
    # Create FedAvg Recipe
    # ========================================

    print("=" * 80)
    print("Creating FedAvg Federated Learning Recipe")
    print("=" * 80)
    print(f"Algorithm: FedAvg (Synchronous)")
    print(f"Dataset: CIFAR10")
    print(f"Data Distribution: {data_distribution} (alpha={dirichlet_alpha})")
    print(f"Total Devices per Leaf: {devices_per_leaf}")
    print(f"Total Devices Across {num_leaf_nodes} Leaves: {devices_per_leaf * num_leaf_nodes}")
    print(f"Selected Devices per Round: {device_selection_size}")
    print(f"Samples per Device: {subset_size}")
    print(f"Synchronous Aggregation: Wait for all {device_selection_size} devices")
    print(f"Global Learning Rate (Server LR): {global_lr}")
    print(f"Local Learning Rate: {local_lr}")
    print(f"Local Batch Size: {local_batch_size}")
    print(f"Local Iterations: {local_iters}")
    print(f"Max Rounds: {max_model_version}")
    print("=" * 80)

    # Task processor for device training simulation
    task_processor = Cifar10FedAvgTaskProcessor(
        data_root=dataset_root,
        subset_size=subset_size,
        communication_delay=communication_delay,
        device_speed=device_speed,
        local_batch_size=local_batch_size,
        local_iters=local_iters,
        local_lr=local_lr,
        data_distribution=data_distribution,
        dirichlet_alpha=dirichlet_alpha,
        # use pre-computed splits:
        dataset_splits_file="/tmp/nvflare/datasets/cifar10/dataset_splits_non_iid_dirichlet_n1000_s350_a0.5.pkl",
    )

    # Model manager configuration (FedAvg settings)
    model_manager_config = ModelManagerConfig(
        global_lr=global_lr,
        num_updates_for_model=num_updates_for_model,  # Synchronous: wait for all devices
        max_model_version=max_model_version,
        max_model_history=max_model_history,
        max_num_active_model_versions=max_model_history,
        update_timeout=600000,
        staleness_weight=False,  # Standard FedAvg: no staleness weighting
    )

    # Device manager configuration
    device_manager_config = DeviceManagerConfig(
        device_selection_size=device_selection_size,
        initial_min_client_num=num_leaf_nodes,
        min_hole_to_fill=min_hole_to_fill,  # Synchronous: wait for all devices
        device_reuse=False,  # Each device participates only once
    )

    # Evaluator configuration
    evaluator_config = EvaluatorConfig(
        torchvision_dataset={"name": "CIFAR10", "path": dataset_root}, eval_frequency=eval_frequency
    )

    # Simulation configuration
    # Reduce num_workers to avoid OOM issues
    # Each worker loads a model + data into memory
    # Start with fewer workers and increase if memory allows
    num_workers = min(4, num_leaf_nodes)  # Limit to 4 workers to reduce memory pressure

    print(f"Parallel workers: {num_workers} (for {device_selection_size} devices)")
    print(f"Expected parallelism: ~{device_selection_size / num_workers:.1f}x batches per round")
    print(f"Note: Using {num_workers} workers (reduced from {num_leaf_nodes}) to avoid OOM")

    simulation_config = SimulationConfig(
        task_processor=task_processor,
        job_timeout=600000.0,  # Increased from 20.0 for safety
        num_workers=num_workers,  # Optimized for parallel execution
        num_devices=devices_per_leaf,
    )

    # Generate the FedAvg recipe using EdgeFedBuffRecipe
    # Note: We configure it for synchronous behavior by setting:
    #   - num_updates_for_model = device_selection_size
    #   - min_hole_to_fill = device_selection_size
    recipe = EdgeFedBuffRecipe(
        job_name="pt_job_fedavg",
        model=ResNet18Global(),
        model_manager_config=model_manager_config,
        device_manager_config=device_manager_config,
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
    print("Executing FedAvg Job")
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
