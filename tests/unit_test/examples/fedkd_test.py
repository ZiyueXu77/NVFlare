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

"""Essential example contracts: proxy isolation and KD aggregation without averaging."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def fedkd(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "advanced"))
    state = np.random.get_state()
    try:
        yield lambda name: importlib.import_module(f"collab.pt_cifar10.fedkd.{name}")
    finally:
        np.random.set_state(state)


def test_proxy_and_client_shards_are_disjoint_complete_and_repeatable(fedkd, tmp_path):
    prepare = fedkd("prepare_data").prepare_splits
    labels = np.tile(np.arange(10), 24)
    first, second = tmp_path / "first", tmp_path / "second"
    assert prepare(labels, first, 4, 0.1, 40, 7) == prepare(labels, second, 4, 0.1, 40, 7)
    proxy = np.load(first / "proxy.npy")
    shards = [np.load(first / f"site-{i}.npy") for i in range(1, 5)]
    assert len(proxy) == 40 and all(len(shard) >= 10 for shard in shards)
    combined = np.concatenate([proxy, *shards])
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(labels)))
    assert all(not np.intersect1d(proxy, shard).size for shard in shards)
    for path in first.glob("*.npy"):
        np.testing.assert_array_equal(np.load(path), np.load(second / path.name))


def test_kd_improves_teacher_agreement_without_averaging_or_teacher_mutation(fedkd, monkeypatch):
    import torch
    from torch import nn

    from nvflare.app_opt.pt.distillation import kl_divergence
    from nvflare.collab.api.group_call_context import FrozenResultQueue

    server = fedkd("server")
    monkeypatch.setattr(server, "SimpleNetwork", lambda: nn.Linear(2, 2, bias=False))
    monkeypatch.setattr(server, "collab", SimpleNamespace(is_aborted=False))
    weights = [torch.tensor([[3.0, 0.0], [-3.0, 0.0]]), torch.tensor([[1.0, 0.0], [-1.0, 0.0]])]
    results = FrozenResultQueue(
        [(f"site-{i + 1}", {"weights": {"weight": w.clone()}, "num_steps": i + 1}) for i, w in enumerate(weights)],
        {},
        2,
    )
    model = nn.Linear(2, 2, bias=False)
    nn.init.zeros_(model.weight)
    inputs = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])

    def loss():
        return sum(kl_divergence(model(inputs), inputs @ w.T).item() for w in weights) / len(weights)

    before = loss()
    metrics = server.FedKDServer(kd_lr=0.1, temperature=1.0).aggregate(
        model, results, [(inputs, torch.full((4,), -999))] * 8, 0
    )
    assert metrics["kd_steps"] == 8 and metrics["kd_samples"] == 32
    assert loss() < before
    assert not torch.allclose(model.weight, torch.stack(weights).mean(0))
    for (_, result), original in zip(results, weights):
        torch.testing.assert_close(result["weights"]["weight"], original, rtol=0, atol=0)
    # The alternate mode retains the existing step-weighted FedAvg behavior.
    server.FedKDServer(aggregation="fedavg").aggregate(model, results, None, 0)
    torch.testing.assert_close(model.weight, (weights[0] + 2 * weights[1]) / 3)
