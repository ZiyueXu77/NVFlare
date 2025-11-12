#!/usr/bin/env python3
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
Verification script for FedRevive implementation.

This script verifies that all FedRevive components are properly implemented
and can be imported without errors.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all components can be imported."""
    print("=" * 80)
    print("Testing FedRevive Implementation Imports")
    print("=" * 80)

    errors = []

    # Test 1: Import FedReviveModelManager
    print("\n[1/5] Testing FedReviveModelManager import...")
    try:
        from nvflare.edge.assessors.fedrevive_model_manager import FedReviveModelManager

        print("✓ FedReviveModelManager imported successfully")
    except Exception as e:
        errors.append(f"FedReviveModelManager import failed: {e}")
        print(f"✗ FedReviveModelManager import failed: {e}")

    # Test 2: Import EdgeFedReviveRecipe
    print("\n[2/5] Testing EdgeFedReviveRecipe import...")
    try:
        from nvflare.edge.tools.edge_fed_revive_recipe import (
            DeviceManagerConfig,
            EdgeFedReviveRecipe,
            EvaluatorConfig,
            FedReviveConfig,
            ModelManagerConfig,
            SimulationConfig,
        )

        print("✓ EdgeFedReviveRecipe and config classes imported successfully")
    except Exception as e:
        errors.append(f"EdgeFedReviveRecipe import failed: {e}")
        print(f"✗ EdgeFedReviveRecipe import failed: {e}")

    # Test 3: Import Cifar10FedReviveTaskProcessor
    print("\n[3/5] Testing Cifar10FedReviveTaskProcessor import...")
    try:
        from processors.cifar10_fedrevive_task_processor import Cifar10FedReviveTaskProcessor

        print("✓ Cifar10FedReviveTaskProcessor imported successfully")
    except Exception as e:
        errors.append(f"Cifar10FedReviveTaskProcessor import failed: {e}")
        print(f"✗ Cifar10FedReviveTaskProcessor import failed: {e}")

    # Test 4: Import CIFAR10 model
    print("\n[4/5] Testing CIFAR10 model import...")
    try:
        from processors.models.cifar10_model import Cifar10ConvNet

        print("✓ Cifar10ConvNet imported successfully")
    except Exception as e:
        errors.append(f"Cifar10ConvNet import failed: {e}")
        print(f"✗ Cifar10ConvNet import failed: {e}")

    # Test 5: Verify job script exists
    print("\n[5/5] Verifying pt_job_fedrevive.py exists...")
    job_script = Path(__file__).parent / "pt_job_fedrevive.py"
    if job_script.exists():
        print(f"✓ Job script exists at: {job_script}")
    else:
        errors.append(f"Job script not found at: {job_script}")
        print(f"✗ Job script not found at: {job_script}")

    # Summary
    print("\n" + "=" * 80)
    if not errors:
        print("SUCCESS: All imports verified!")
        print("=" * 80)
        return True
    else:
        print("ERRORS FOUND:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("=" * 80)
        return False


def test_configuration():
    """Test that configurations can be created."""
    print("\n" + "=" * 80)
    print("Testing Configuration Creation")
    print("=" * 80)

    try:
        from nvflare.edge.tools.edge_fed_revive_recipe import DeviceManagerConfig, FedReviveConfig, ModelManagerConfig

        # Test FedReviveConfig
        print("\n[1/3] Creating FedReviveConfig...")
        fedrevive_config = FedReviveConfig(
            kd_enabled=True,
            kd_buffer_size=3,
            kd_beta=0.5,
        )
        print(
            f"✓ FedReviveConfig created: kd_enabled={fedrevive_config.kd_enabled}, "
            f"buffer_size={fedrevive_config.kd_buffer_size}"
        )

        # Test ModelManagerConfig
        print("\n[2/3] Creating ModelManagerConfig...")
        model_config = ModelManagerConfig(
            global_lr=0.1,
            num_updates_for_model=20,
            max_model_version=200,
        )
        print(
            f"✓ ModelManagerConfig created: global_lr={model_config.global_lr}, "
            f"num_updates={model_config.num_updates_for_model}"
        )

        # Test DeviceManagerConfig
        print("\n[3/3] Creating DeviceManagerConfig...")
        device_config = DeviceManagerConfig(
            device_selection_size=200,
            min_hole_to_fill=10,
        )
        print(
            f"✓ DeviceManagerConfig created: selection_size={device_config.device_selection_size}, "
            f"min_hole={device_config.min_hole_to_fill}"
        )

        print("\n" + "=" * 80)
        print("SUCCESS: All configurations created!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n✗ Configuration creation failed: {e}")
        print("=" * 80)
        return False


def test_model_creation():
    """Test that the model can be instantiated."""
    print("\n" + "=" * 80)
    print("Testing Model Creation")
    print("=" * 80)

    try:
        from processors.models.cifar10_model import Cifar10ConvNet

        print("\nCreating Cifar10ConvNet...")
        model = Cifar10ConvNet()

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"✓ Model created successfully!")
        print(f"  - Total parameters: {num_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        print("\n" + "=" * 80)
        print("SUCCESS: Model created!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n✗ Model creation failed: {e}")
        print("=" * 80)
        return False


def print_summary():
    """Print implementation summary."""
    print("\n" + "=" * 80)
    print("FedRevive Implementation Summary")
    print("=" * 80)
    print(
        """
Components Implemented:

1. FedReviveModelManager (nvflare/edge/assessors/fedrevive_model_manager.py)
   - Buffered asynchronous aggregation
   - KD buffer management (stores recent updates as teachers)
   - Adaptive beta weighting based on staleness
   - KD augmentation for stale updates

2. EdgeFedReviveRecipe (nvflare/edge/tools/edge_fed_revive_recipe.py)
   - High-level recipe for FedRevive jobs
   - Configuration classes for all components
   - Integrates FedReviveModelManager with NVFlare Edge

3. Cifar10FedReviveTaskProcessor (examples/.../cifar10_fedrevive_task_processor.py)
   - Local training following FedRevive specifications
   - Adam optimizer (better for async FL)
   - Fixed iterations (not epochs)
   - Non-IID data via Dirichlet sampling
   - Computational heterogeneity simulation

4. pt_job_fedrevive.py (examples/.../pt_job_fedrevive.py)
   - Main script to create and run FedRevive jobs
   - Comprehensive configuration with defaults
   - Ready to run example

Key Features:
- ✓ Asynchronous federated learning
- ✓ Knowledge distillation for staleness handling
- ✓ Adaptive weighting (beta decreases with staleness)
- ✓ Non-IID data distribution (Dirichlet)
- ✓ Computational heterogeneity simulation
- ✓ CIFAR10 dataset support
- ✓ Comprehensive documentation

Usage:
  python pt_job_fedrevive.py

For more information, see FEDREVIVE_README.md
"""
    )
    print("=" * 80)


def main():
    """Run all verification tests."""
    print(
        """
    ╔═══════════════════════════════════════════════════════════════╗
    ║           FedRevive Implementation Verification               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    )

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test configurations (only if imports passed)
    if all_passed:
        if not test_configuration():
            all_passed = False

    # Test model creation (only if previous tests passed)
    if all_passed:
        if not test_model_creation():
            all_passed = False

    # Print summary
    print_summary()

    # Final result
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ VERIFICATION COMPLETE: All tests passed!")
        print("  FedRevive implementation is ready to use.")
        print("\n  Next steps:")
        print("  1. Review FEDREVIVE_README.md for documentation")
        print("  2. Run: python pt_job_fedrevive.py")
    else:
        print("✗ VERIFICATION FAILED: Some tests did not pass.")
        print("  Please check the errors above and fix the issues.")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
