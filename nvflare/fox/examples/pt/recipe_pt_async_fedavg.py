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
Recipe for Asynchronous FedAvg with Decoupled Update and Dispatch

This recipe demonstrates running async federated averaging in a POC environment
with independent aggregation and dispatch triggers.
"""
import logging

from nvflare.fox.api.app import ServerApp
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.pt.pt_async_fedavg import AsyncPTFedAvg, AsyncPTTrainer
from nvflare.fox.sys.recipe import FoxRecipe
from nvflare.recipe.poc_env import PocEnv

JOB_ROOT_DIR = "/tmp/nvflare/fox_project/prod_00/admin@nvidia.com/transfer"


def main():
    """
    Run async FedAvg recipe with decoupled operations.

    Configuration:
    - 5 clients total
    - M=2: Aggregate when 2 responses received
    - N=2: Dispatch when 2 slots available
    - Max 3 concurrent clients
    - Run until version 10
    """
    simple_logging(logging.INFO)

    server_app = ServerApp(
        strategy_name="async_fedavg_decoupled",
        strategy=AsyncPTFedAvg(
            initial_model={
                "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
            max_version=10,
            aggregation_threshold=2,  # M: aggregate every 2 responses
            dispatch_threshold=2,  # N: dispatch when 2 slots available
            max_concurrent_clients=3,  # Max 3 clients training at once
        ),
    )

    client_app = AsyncPTTrainer(delta=1.0)

    env = PocEnv(num_clients=5)

    recipe = FoxRecipe(
        job_name="pt_async_fedavg_decoupled",
        server_app=server_app,
        client_app=client_app,
    )

    print(f"Job exported to {JOB_ROOT_DIR}")
    recipe.export(JOB_ROOT_DIR)

    run = recipe.execute(env=env)
    run.get_status()
    result = run.get_result()

    print("\nAsync FedAvg completed!")
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()
