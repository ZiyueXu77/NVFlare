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

from src.cifar10_nets import ModerateCNN
from src.fedavg_advanced import FedAvgSync
from src.utils.cifar10_data_splitter import Cifar10DataSplitter

from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

if __name__ == "__main__":
    n_clients = 20
    num_rounds = 100
    alpha = 0.5
    train_idx_root = "/tmp/nvflare/dataset/cifar10_idx"
    train_script = "src/cifar10_fl.py"

    job = FedJob(name="cifar10_fedavg_sync")

    # Define the controller workflow and send to server
    controller = FedAvgSync(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to_server(controller)

    # Define the initial global model and send to server
    job.to_server(PTModel(ModerateCNN()))
    job.to(IntimeModelSelector(key_metric="eval_acc"), "server")

    # Add data splitter to server
    data_spliter = Cifar10DataSplitter(split_dir=train_idx_root, num_sites=n_clients, alpha=alpha)
    job.to_server(data_spliter, id="data_splitter")

    # Add executor to clients
    executor = ScriptRunner(script=train_script, script_args="")
    job.to_clients(executor)

    job.export_job("/tmp/nvflare/workspace/jobs")
    job.simulator_run("/tmp/nvflare/workspace/works", n_clients=n_clients, gpu="0")
