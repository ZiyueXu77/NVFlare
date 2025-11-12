# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse

from nvflare import FedJob
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.script_runner import ScriptRunner


def main():
    args = define_parser()
    train_script = "src/hf_sft_peft_fl.py"
    client_ids = args.client_ids
    num_clients = len(client_ids)

    if args.threads:
        num_threads = args.threads
    else:
        num_threads = num_clients

    num_rounds = args.num_rounds
    workspace_dir = args.workspace_dir
    job_dir = args.job_dir
    model_name_or_path = args.model_name_or_path

    # Create the FedJob
    job = FedJob(name="llm_hf_sft", min_clients=num_clients)

    # Define the FedAvg controller workflow and send to server
    controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    # First send the model to the server
    job.to("src/hf_sft_model.py", "server")
    # Then send the model persistor to the server
    model_args = {"path": "src.hf_sft_model.CausalLMModel", "args": {"model_name_or_path": model_name_or_path}}
    job.to(PTFileModelPersistor(model=model_args, allow_numpy_conversion=False), "server", id="persistor")

    # Add model selection widget and send to server
    job.to(IntimeModelSelector(key_metric="eval_loss", negate_key_metric=True), "server", id="model_selector")

    # Send ScriptRunner to all clients
    for i in range(num_clients):
        client_id = client_ids[i]
        site_name = f"site-{client_id}"
        script_args = f"--model_name_or_path {model_name_or_path} --num_rounds {num_rounds}"
        server_expected_format = "pytorch"

        runner = ScriptRunner(
            script=train_script,
            script_args=script_args,
            server_expected_format=server_expected_format,
            launch_external_process=True,
        )
        job.to(runner, site_name, tasks=["train"])

        # Add additional parameters to clients
        client_params = {"get_task_timeout": 300, "submit_task_result_timeout": 300}
        job.to(client_params, site_name)

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", workspace_dir)
    print("num_threads=", num_threads)
    job.simulator_run(workspace_dir, threads=num_threads, gpu=args.gpu)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_ids",
        nargs="+",
        type=str,
        default="",
        help="Clinet IDs, used to get the data path for each client",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of rounds, default to 3",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="/tmp/nvflare/base_hf/workdir",
        help="work directory, default to '/tmp/nvflare/llm_hf/workdir'",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="/tmp/nvflare/base_hf/jobdir",
        help="directory for job export, default to '/tmp/nvflare/llm_hf/jobdir'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-125m",
        help="model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="root directory for training and validation data",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads to use for FL simulation, default to the number of clients",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="gpu assignments for simulating clients, comma separated, default to single gpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
