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

"""Focused research contracts for revival, phase eligibility and FedAvg bases."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def research(monkeypatch):
    folder = Path(__file__).resolve().parents[3] / "research" / "fedrevive"
    monkeypatch.syspath_prepend(str(folder))
    names = ("fedrevive", "update", "schedule")
    saved = {name: sys.modules.pop(name, None) for name in names}
    try:
        yield lambda name: importlib.import_module(name)
    finally:
        for name, previous in saved.items():
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous


def test_revival_uses_current_global_delta_and_refreshes_arriving_teacher(research, tmp_path):
    torch = pytest.importorskip("torch")
    from nvflare.collab.fedbuff_recipe import AcceptedUpdate, Assignment

    policy_module = research("update")
    policy = policy_module.FedReviveUpdate("unused")
    policy.run_dir = tmp_path
    policy.config = SimpleNamespace(freeze_versions=50)
    policy.teachers = research("fedrevive").TeacherBuffer(2)
    policy.estimator = SimpleNamespace(estimate=lambda name, model: torch.tensor([0.25, 0.75]))
    seen = []

    def revive(global_model, teachers, version):
        seen.append((global_model["w"].item(), [t.model["w"].item() for t in teachers], version))
        assert all(torch.equal(t.class_proxy, torch.tensor([0.25, 0.75])) for t in teachers)
        return {"w": torch.tensor([8.0])}, {}

    policy.reviver = SimpleNamespace(revive=revive)
    policy.teachers.add({"w": torch.tensor([1.0])}, torch.tensor([1.0, 0.0]), "alice", 0)
    assignment = Assignment(7, "site-1", "alice", 0, {"w": torch.tensor([0.0])})
    update = AcceptedUpdate(
        assignment, 75, {"w": torch.tensor([20.0])}, {"w": torch.tensor([2.0])}, 1.0, {}, {"w": torch.tensor([2.0])}, {}
    )
    result = policy(update)
    assert seen == [(20.0, [2.0, 1.0], 75)]
    torch.testing.assert_close(result["w"], torch.tensor([5.0]))  # beta(75)=0.5
    assert update.current_model["w"].item() == 20 and update.delta["w"].item() == 2
    # An on-time upload still enters history but does not trigger revival.
    current = Assignment(8, "site-1", "alice", 75, update.current_model)
    policy(AcceptedUpdate(current, 75, update.current_model, update.delta, 1.0, {}, update.returned_model, {}))
    assert len(seen) == 1 and len(policy.teachers) == 2


def test_phase_release_precedes_upload_and_horizon_rejects_due_result(research, tmp_path):
    schedule = research("schedule").PhaseSchedule(
        {"client-0": {"download_mean": 0.1, "train_mean": 1.0, "upload_mean": 0.2}},
        run_seed=10,
        max_time=0.1 + np.random.RandomState(10).exponential(1.0) + 0.1,
    )

    class Coordinator:
        run_dir = tmp_path
        active = {}
        finished = False
        available_slots = 1

        def reserve(self, names, run_seed):
            assignment = SimpleNamespace(assignment_id=0, logical_name=names[0])
            self.active[0] = assignment
            return [assignment]

    schedule.initialize(Coordinator(), ["site-1"])
    calls = schedule.dispatch()
    assert len(calls) == 1 and schedule.logical == []
    assert schedule.next_result() is None  # download/train occur before physical completion
    assert schedule.logical == ["client-0"] and not schedule.stopped
    schedule.on_result("site-1", 0)
    assert schedule.next_result() is None and schedule.stopped  # upload exceeds horizon
    phases = (tmp_path / "phases.jsonl").read_text()
    assert all(f'"phase": "{name}"' in phases for name in ("download", "train", "upload"))
    schedule.close()


def test_fedavg_rejects_mixed_assignment_bases_within_a_publication(research):
    pytest.importorskip("torch")
    guard = research("update").FedAvgUpdate(2)
    delta = {"w": 1.0}

    def update(version):
        return SimpleNamespace(assignment=SimpleNamespace(base_version=version), delta=delta)

    assert guard(update(0)) is delta
    assert guard(update(0)) is delta
    guard(update(1))
    with pytest.raises(RuntimeError, match="different global versions"):
        guard(update(2))
