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
import logging

from nvflare.fox.api.app import ServerApp
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.pt.pt_hf_avg_stream import HFFedAvgStream, HFTrainer
from nvflare.fox.sys.recipe import FoxRecipe
from nvflare.recipe.poc_env import PocEnv

JOB_ROOT_DIR = "/tmp/nvflare/fox_project/prod_00/admin@nvidia.com/transfer"

env = PocEnv(num_clients=2)


def main():
    simple_logging(logging.INFO)

    # Model configuration
    model_name_or_path = "facebook/opt-125m"

    # Model transmission configuration
    num_tensors_per_chunk = 10

    # Federated learning configuration
    num_rounds = 3
    local_epoch = 1

    # Server app with HF FedAvg strategy
    server_app = ServerApp(
        strategy_name="hf_fedavg_stream",
        strategy=HFFedAvgStream(
            model_name_or_path=model_name_or_path,
            num_rounds=num_rounds,
            timeout=10.0,
            num_tensors_per_chunk=num_tensors_per_chunk,
            min_clients=None,  # Require all clients
        ),
    )

    # Client app with HF trainer
    client_app = HFTrainer(
        model_name_or_path=model_name_or_path,
        output_path="dolly-sft",
        data_path_train="/media/ziyuexu/Data/FL_Dataset/LLM/dolly/training.jsonl",
        data_path_valid="/media/ziyuexu/Data/FL_Dataset/LLM/dolly/validation.jsonl",
        num_tensors_per_chunk=num_tensors_per_chunk,
        lr_scheduler="constant",
        local_epoch=local_epoch,
        num_rounds=num_rounds,
        batch_size=4,
        gradient_accumulation_steps=10,
        learning_rate=5e-4,
        max_length=1024,
        device="cuda:0",  # Use GPU for production deployment
    )

    recipe = FoxRecipe(
        job_name="hf_fedavg_stream",
        server_app=server_app,
        client_app=client_app,
    )
    print(f"Job exported to {JOB_ROOT_DIR}")
    recipe.export(JOB_ROOT_DIR)
    run = recipe.execute(env=env)
    run.get_status()
    run.get_result()


if __name__ == "__main__":
    main()
