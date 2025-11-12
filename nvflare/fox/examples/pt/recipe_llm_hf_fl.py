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

import sys

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.recipe.poc_env import PocEnv

# Add src directory to path so we can import the model
sys.path.append("src")
from hf_sft_model import CausalLMModel


def main():
    # Model configuration
    model_name_or_path = "facebook/opt-125m"

    # Federated learning configuration
    num_rounds = 3
    num_clients = 2

    # Training script configuration
    train_script = "src/hf_sft_peft_fl.py"
    train_args = (
        f"--model_name_or_path {model_name_or_path} "
        f"--num_rounds {num_rounds} "
        f"--local_epoch 1 "
        f"--data_path_train /media/ziyuexu/Data/FL_Dataset/LLM/dolly/training.jsonl "
        f"--data_path_valid /media/ziyuexu/Data/FL_Dataset/LLM/dolly/validation.jsonl "
        f"--output_path opt-125m-sft-fl "
        f"--lr_scheduler constant"
    )

    # Create initial model
    initial_model = CausalLMModel(model_name_or_path=model_name_or_path)

    # Create FedAvg recipe
    recipe = FedAvgRecipe(
        name="llm_hf_sft",
        initial_model=initial_model,
        min_clients=num_clients,
        num_rounds=num_rounds,
        train_script=train_script,
        train_args=train_args,
        launch_external_process=True,
        command="python3 -u",
        server_expected_format=ExchangeFormat.PYTORCH,
        params_transfer_type=TransferType.FULL,
    )

    print("Recipe created successfully!")
    print(f"Recipe name: {recipe.name}")
    print(f"Min clients: {recipe.min_clients}")
    print(f"Number of rounds: {recipe.num_rounds}")

    # Create POC environment
    env = PocEnv(num_clients=num_clients)

    # Execute the recipe
    print(f"\nRunning with POC environment (num_clients={num_clients})")
    run = recipe.execute(env=env)
    run.get_status()
    run.get_result()


if __name__ == "__main__":
    main()
