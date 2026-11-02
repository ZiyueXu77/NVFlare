# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Synchronous rounds with replaceable weight averaging or KD aggregation."""

import json
import time
from functools import partial
from pathlib import Path

import torch

from nvflare.app_opt.pt.distillation import DistillationTrainer, kl_divergence
from nvflare.collab import collab

from ..aggregation import aggregate_result
from ..fedavg.client import BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS
from ..fedavg.server import TRAIN_TIMEOUT, FedAvgServer
from ..loader import make_data_loader
from ..model import SimpleNetwork, get_model_state


def _teacher(weights):
    model = SimpleNetwork()
    model.load_state_dict(weights)
    return model


class FedKDServer:
    def __init__(
        self,
        num_rounds=20,
        aggregation="kd",
        split_root="/tmp/cifar10_splits/pt_cifar10_fedkd",
        seed=10,
        kd_epochs=1,
        kd_lr=0.001,
        temperature=2.0,
    ):
        if aggregation not in ("fedavg", "kd"):
            raise ValueError("aggregation must be fedavg or kd")
        if num_rounds < 1 or kd_epochs < 1 or kd_lr <= 0 or temperature <= 0:
            raise ValueError("Round count, KD epochs, learning rate and temperature must be positive")
        self.num_rounds = num_rounds
        self.aggregation = aggregation
        self.split_root = split_root
        self.seed = seed
        self.kd_epochs = kd_epochs
        self.kd_lr = kd_lr
        self.temperature = temperature

    def aggregate(self, model, client_results, proxy_loader, round_number):
        """Retain round semantics while replacing only server aggregation.

        KD starts from the previous global model, without parameter averaging.
        Each current-round teacher contributes equally to the mean KL loss.
        The FedAvg baseline retains the existing local-step weighting.
        """
        if client_results.failures:
            raise next(iter(client_results.failures.values()))
        # Stable teacher order avoids completion timing changing reduction order.
        results = sorted(client_results)
        if not results:
            raise ValueError("No client models were returned")
        if self.aggregation == "fedavg":
            model.load_state_dict(aggregate_result(client_results, "weights", round_number))
            return {}

        trainer = DistillationTrainer(
            student=model,
            teachers=[partial(_teacher, result["weights"]) for _, result in results],
            optimizer_factory=partial(torch.optim.Adam, lr=self.kd_lr),
            device=next(model.parameters()).device,
            batch_adapter=lambda batch: batch[0],  # Proxy targets never enter the loss.
            loss_fn=partial(kl_divergence, temperature=self.temperature),
        )
        total_loss, samples, steps = 0.0, 0, 0
        for _ in range(self.kd_epochs):
            if collab.is_aborted:
                raise RuntimeError("Distillation aborted")
            metrics = trainer.fit(proxy_loader)
            total_loss += metrics["loss"] * metrics["samples"]
            samples += metrics["samples"]
            steps += metrics["steps"]
        return {"kd_loss": total_loss / samples, "kd_steps": steps, "kd_samples": samples}

    @collab.main
    def run(self):
        manifest = json.loads((Path(self.split_root) / "manifest.json").read_text())
        if manifest["num_clients"] != len(collab.clients):
            raise ValueError("The client count must match the prepared manifest")
        torch.manual_seed(self.seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SimpleNetwork().to(device)
        test_loader = make_data_loader(train=False, batch_size=BATCH_SIZE)
        proxy_loader = None
        if self.aggregation == "kd":
            # An independent loader generator prevents teacher construction from
            # changing proxy order. The same prepared transform is used by clients.
            proxy_loader = make_data_loader(
                train=True,
                batch_size=BATCH_SIZE,
                site_name="proxy",
                split_root=self.split_root,
                generator=torch.Generator(device="cpu").manual_seed(self.seed),
            )
        run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        run_dir.mkdir(parents=True, exist_ok=True)
        config = dict(
            vars(self),
            data=manifest,
            local_batch_size=BATCH_SIZE,
            local_lr=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS,
            kd_teacher_weighting="uniform",
            fedavg_weighting="local_steps",
        )
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
        torch.save(get_model_state(model), run_dir / "initial_model.pt")
        history = []
        for round_number in range(self.num_rounds):
            if collab.is_aborted:
                raise RuntimeError("Training aborted")
            results = collab.clients(timeout=TRAIN_TIMEOUT).train(get_model_state(model))
            started = time.perf_counter()
            metrics = self.aggregate(model, results, proxy_loader, round_number)
            metrics["aggregation_seconds"] = time.perf_counter() - started
            accuracy = FedAvgServer.evaluate(model, test_loader)
            history.append(
                dict(
                    round=round_number + 1,
                    accuracy=accuracy,
                    clients=sorted(name for name, _ in results),
                    local_steps=sum(result["num_steps"] for _, result in results),
                    **metrics,
                )
            )
            (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
            print(f"Round {round_number + 1}: {self.aggregation}, test accuracy={accuracy:.4f}")
        weights = get_model_state(model)
        torch.save(weights, run_dir / "final_model.pt")
        return weights
