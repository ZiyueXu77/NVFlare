#!/usr/bin/env python3
"""
Fast Testing Version of FedAvg Job

This is a reduced-scale version for quick testing and debugging:
- 20 devices instead of 100 (5x faster per round)
- 50 total rounds instead of 200 (4x fewer rounds)
- 10 local iterations instead of 25 (2.5x faster per device)

Expected time: ~2-5 minutes total (vs ~50 minutes for full version)

Use this for:
- Testing configuration changes
- Debugging code
- Quick convergence checks
- Development iteration

For full experiments, use pt_job_fedavg.py
"""

import os

from processors.cifar10_fedavg_task_processor import Cifar10FedAvgTaskProcessor
from processors.models.cifar10_model import Cifar10ConvNet

from nvflare.edge.tools.edge_fed_buff_recipe import (
    DeviceManagerConfig,
    EdgeFedBuffRecipe,
    EvaluatorConfig,
    ModelManagerConfig,
    SimulationConfig,
)
from nvflare.recipe.prod_env import ProdEnv


def main():
    """Fast testing configuration for quick iterations."""

    # ========================================
    # FAST TESTING CONFIGURATION
    # ========================================

    print("=" * 80)
    print("⚡ FAST TESTING MODE ⚡")
    print("=" * 80)
    print("This is a reduced-scale version for quick testing.")
    print("For full experiments, use pt_job_fedavg.py")
    print("=" * 80 + "\n")

    # Reduced scale for fast testing
    devices_per_leaf = 100  # Reduced pool (still sufficient)
    device_selection_size = 20  # Only 20 devices per round (vs 100)
    num_leaf_nodes = 4

    # NVFlare system paths
    startup_kit_location = "/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com"
    username = "admin@nvidia.com"
    output_dir = "/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/transfer"
    dataset_root = "/tmp/nvflare/datasets/cifar10"

    # ========================================
    # Algorithm Configuration (Reduced)
    # ========================================

    global_lr = 1.4
    num_updates_for_model = device_selection_size  # Synchronous
    max_model_version = 50  # Only 50 rounds (vs 200)
    max_model_history = None
    min_hole_to_fill = device_selection_size
    eval_frequency = 5  # Evaluate every 5 rounds (vs every round)

    # ========================================
    # Local Training (Reduced)
    # ========================================

    subset_size = 350  # Keep same data size
    data_distribution = "non_iid_dirichlet"
    dirichlet_alpha = 0.5

    local_batch_size = 32
    local_iters = 10  # Reduced iterations (vs 25)
    local_lr = 0.0003

    # ========================================
    # Delays (Disabled for Speed)
    # ========================================

    # Disable artificial delays for faster testing
    communication_delay = {"mean": 0.0, "std": 0.0}
    device_speed = {"mean": [0.0], "std": [0.0]}

    # ========================================
    # Summary
    # ========================================

    print("Configuration Summary:")
    print(f"  Devices per round: {device_selection_size} (5x faster)")
    print(f"  Total rounds: {max_model_version} (4x fewer)")
    print(f"  Local iterations: {local_iters} (2.5x faster)")
    print(f"  Eval frequency: every {eval_frequency} rounds")
    print(f"  Delays: DISABLED")
    print(f"\n  Expected speedup: ~50x faster than full version!")
    print(f"  Expected time: 2-5 minutes total\n")
    print("=" * 80 + "\n")

    # Task processor
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
    )

    # Model manager configuration
    model_manager_config = ModelManagerConfig(
        global_lr=global_lr,
        num_updates_for_model=num_updates_for_model,
        max_model_version=max_model_version,
        max_model_history=max_model_history,
        max_num_active_model_versions=max_model_history,
        update_timeout=500,
        staleness_weight=False,
    )

    # Device manager configuration
    device_manager_config = DeviceManagerConfig(
        device_selection_size=device_selection_size,
        min_hole_to_fill=min_hole_to_fill,
        device_reuse=False,
    )

    # Evaluator configuration
    evaluator_config = EvaluatorConfig(
        torchvision_dataset={"name": "CIFAR10", "path": dataset_root}, eval_frequency=eval_frequency
    )

    # Simulation configuration - match workers to devices for fast testing
    num_workers = device_selection_size  # 20 workers for 20 devices

    print(f"Parallel workers: {num_workers} (for {device_selection_size} devices)")
    print(f"Expected parallelism: ~1.0x (all parallel - as much as single GPU allows)")
    print()

    simulation_config = SimulationConfig(
        task_processor=task_processor,
        job_timeout=60.0,
        num_workers=num_workers,
        num_devices=devices_per_leaf,
    )

    # Generate recipe
    recipe = EdgeFedBuffRecipe(
        job_name="pt_job_fedavg_fast",
        model=Cifar10ConvNet(),
        model_manager_config=model_manager_config,
        device_manager_config=device_manager_config,
        evaluator_config=evaluator_config,
        simulation_config=simulation_config,
        custom_source_root=None,
    )

    print(f"Exporting recipe to {output_dir}")
    recipe.export(output_dir)
    print("Recipe exported successfully!\n")

    # Execute
    print("=" * 80)
    print("Executing Fast FedAvg Job")
    print("=" * 80)

    import time

    start_time = time.time()

    env = ProdEnv(startup_kit_location=startup_kit_location, username=username)
    run = recipe.execute(env)

    end_time = time.time()
    elapsed = end_time - start_time

    # Results
    print("\n" + "=" * 80)
    print("Job Execution Complete")
    print("=" * 80)
    print(f"Results: {run.get_result()}")
    print(f"Status: {run.get_status()}")
    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Average time per round: {elapsed/max_model_version:.1f} seconds")
    print("=" * 80)

    # Comparison
    print("\nPerformance Comparison:")
    print(f"  This run: {elapsed/60:.1f} minutes for {max_model_version} rounds")
    full_estimate = (elapsed / max_model_version) * 200
    print(f"  Full version estimate: {full_estimate/60:.1f} minutes for 200 rounds")
    print(f"  Speedup: {full_estimate/elapsed:.1f}x faster!")
    print()


if __name__ == "__main__":
    exit(main())
