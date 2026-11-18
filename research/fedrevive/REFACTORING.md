# FedRevive component migration

This work covers Stage 5 and the first Stage 6 consumer in
[the FedRevive roadmap](https://github.com/NVIDIA/NVFlare/issues/5572).
This branch composes FedBuff, distillation and the reproducible simulator.
The research application owns the CIFAR architecture, augmentation,
paper scheduler and algorithm presets.

## Branches and pinned inputs

| Branch | Contents | Review target |
| --- | --- | --- |
| `codex/fedrevive-dependencies` | Upstream main with merged FedBuff, plus KD and replay PR heads | Dependency baseline |
| `codex/fedrevive` | FedRevive changes above that baseline | Compare against `codex/fedrevive-dependencies` while KD and replay are pending |

The current dependency baseline is `e22a54fc67dafa11f86f94dd381f602e4d8bd8e2`.
It uses the following source revisions:

| Dependency | Branch | Pinned head |
| --- | --- | --- |
| Upstream main | `main` | `703d3b07dec45ea020ea706d32f29fb1ee2771b2` |
| [Merged FedBuff recipe, PR #5573](https://github.com/NVIDIA/NVFlare/pull/5573) | `main` | `e19bda577d438243374a5f00be8c41f4f10d0fde` |
| [Standalone KD and synthesis, PR #5613](https://github.com/NVIDIA/NVFlare/pull/5613) | `codex/standalone-kd` | `97639bc777e28eb575653964caf7eaad7e3a9605` |
| [Reproducible SimEnv, PR #5780](https://github.com/NVIDIA/NVFlare/pull/5780) | `codex/reproducible-env` | `4e62daded67db10a46da70dbdf67deb353aa118c` |

Runtime code imports the shared components directly. The integration baseline
keeps a merge parent for the previous baseline so existing branch users can
fast-forward. Its tree comes from the current source revisions above.

## Coordinating dependency changes

Use a dedicated clean worktree for these branches. Fix shared components in their
own PR branches with focused tests. Bring reviewed updates into the dependency
branch, then merge that baseline into the FedRevive branch.

For ordinary dependency updates that preserve the pinned history:

```bash
git fetch origin codex/standalone-kd codex/reproducible-env
git switch codex/fedrevive-dependencies
git merge --no-ff origin/codex/standalone-kd
git merge --no-ff origin/codex/reproducible-env
git switch codex/fedrevive
git merge --no-ff codex/fedrevive-dependencies
```

Before merging an updated source head, verify that its previously pinned SHA is
still an ancestor (`git merge-base --is-ancestor OLD_SHA NEW_SHA`). After a
squash merge or a rewritten source branch, rebuild the baseline from current
upstream main and the pending PR heads. Reapply the FedRevive diff from the
previous baseline. Preserve the previous branch tips as merge parents if the
refreshed branches need to fast-forward.

After each dependency update, record the new baseline/head SHAs, run the
shared tests plus the relevant FedRevive comparison, and push
the dependency baseline before the feature branch. Compare FedRevive changes
against the dependency branch; comparison against `main` also includes both
pending PRs. These are deliberate update checkpoints, not automatic branch polling.

These branches live in the contributor fork. Use its compare view or a fork-local
draft PR for a focused intermediate review; NVIDIA's repository cannot use a
fork-only dependency branch as its PR base. The final upstream PR targets NVIDIA
`main` after the source PRs land.

After KD and replay merge, transplant the FedRevive branch diff onto current
NVIDIA `main` on a fresh branch. Use the two branch trees to capture changes
introduced by the composition merge:

```bash
git diff --binary codex/fedrevive-dependencies codex/fedrevive > /tmp/fedrevive-composition.patch
git fetch https://github.com/NVIDIA/NVFlare.git main
git switch -c codex/fedrevive-main FETCH_HEAD
git apply --index --3way /tmp/fedrevive-composition.patch
```

Review the applied diff and retest before committing and opening the final PR.

## Implementation slices

1. **Shared math, then fixed-input comparisons.** Route ordinary KD through
   `distillation.losses.kl_divergence`. Keep synthesis gradients distinct. Capture
   the reference behavior before changing teacher/pool storage or RNG ordering.
2. **FedRevive update composition.** Use the shared teacher history and an optional
   async `update_transform`. The client delta references its assignment base;
   the distilled delta references the current global model. Preserve insertion
   of the arriving teacher before revival and update existing teacher proxies
   when a client's second upload finalizes its estimate.
3. **Research DFKD composition.** Reuse synthesis/pool/KD mechanics where their
   contracts match. Keep the conditional generator, optimized latent/embedding
   tensors, Reptile meta-update, per-teacher sampling/weighting, augmentation,
   loss stabilization and tracker policy in the research implementation.
4. **Async workflows.** Compose actual-arrival training from `FedBuffRecipe`.
   Migrate the seeded download/train/upload scheduler to the research workflow,
   reusing `FedBuffTrainingCoordinator`, Collab calls and disk-first artifacts.
   Preserve logical-client release after training, before upload, and the
   simulated-time cutoff. Fixed-order replay alone does not implement this policy.
5. **Independent reuse and qualification.** Compare fixed updates and a small
   seeded run, validate export and one/two-worker equivalence, then demonstrate
   the async and KD backbones in consumers outside FedRevive. Record intentional
   numerical changes explicitly.

## Preservation checks and current progress

- [x] Combine the two pinned PR heads without changes to either dependency.
- [x] Route ordinary student KD through the shared loss; retain synthesis KL
  with teacher-input gradients.
- [x] Compose the FedRevive update transformation and shared teacher history.
- [x] Migrate research pool/probe/feature math and phase scheduling; retain the
  task-specific synthesis and per-teacher KD loops where generic contracts differ.
- [x] Add actual-arrival composition and compare small seeded runs.

Preserve the reference `run_seed * 1_000_000 + assignment_id + 1` training seeds
and NumPy `RandomState` scheduler draws. The generic example seed hash and
vectorized class sampler have different draw orders; substituting them is not
an exact replay. Preserve the version-50 synthesis gate, version-100 KD gate,
generation interval, tracker resets and method concurrency/buffer/refill/server-LR presets.

Basic KD detaches teacher outputs. Adversarial synthesis must retain gradients
through both teacher and student outputs to generated inputs. A single shared
KL wrapper for both paths would change the experiment silently.

At each checkpoint run the focused FedBuff, replay and KD cases, relevant FedRevive
fixed-input/seeded comparisons, project style and license checks. Keep tests
focused on these contracts; do not add a broad research unit-test matrix.
CPU smoke checks do not qualify CUDA or reproduce the paper's accuracy curves.

Initial checkpoint validation: all 36 async/KD helper cases pass in the combined
checkout. A two-teacher CPU comparison against the dependency baseline ran one
synthesis step and two ordinary KD steps using the actual DFKD reviver with a
small classifier. Model deltas, generator state and loss metrics were bit-exact;
synthesis retained teacher-input gradients and ordinary KD detached its teacher
targets. This used PyTorch 2.15 and Kornia 0.9, without downloading a dataset.
Full project style, explicit style checks on the changed research module, license
headers and changed Markdown links pass.

## Stage 5 composition

`FedReviveUpdate` attaches to the core coordinator in both arrival modes.
It inserts each arriving teacher before revival, refreshes retained client
proxies, preserves the version-50/version-100 gates, and blends the exact-base
client delta with the current-global KD delta. `FedAvgUpdate` preserves the
same-base-per-buffer check when users override method presets.

`FedReviveWorkflow` reuses `FedBuffWorkflow` calls and lifecycle. Its phase scheduler
retains the reference `RandomState`, heap tie rule, logical release point and
linear training seeds. Research reservations retain speculative assignments near
the horizon, as the reference did. `TensorStore` retains bases/results and all
publications on disk without reference counting or reclamation. The former
`--no-in-time` inspection path is rejected; raw result files serve inspection.

The shared `TeacherBuffer`, `SyntheticPool`, ordinary KD loss, feature-statistics
math and mean-probability probe are used directly. The research class sampler
preserves the original RNG draws, including absent-class fallback. Generic
`DataFreeSynthesizer` fixes a latent batch and optimizes generator parameters;
FedRevive also optimizes latent/embedding tensors and meta-updates a persistent
generator. Generic `DistillationTrainer` evaluates teachers on the same batch;
FedRevive samples/augments a separate batch per teacher before one optimizer
step. Those research loops intentionally remain explicit.

Validation includes all three method schedules through 110 small-model
publications and a time-horizon cutoff, with one/two workers. Assignment identity,
logical identity, bases, timestamps, seeds and final parameters match the
pre-migration implementation exactly. A two-teacher synthesis/KD comparison
matches model deltas, generator state and losses exactly; pool samples/RNG
continuation and two-upload probe freezing match exactly. The shared feature
helper groups each layer's mean/variance terms before summation: one ResNet CPU
comparison changes the scalar from 299.679016 to 299.679047 with identical input
gradients. This floating-point reduction change is explicit; complete CIFAR
trajectories are not claimed bit-identical.

Dataset-free jobs through the real Collab simulator complete in both modes.
The one/two-worker phase runs have identical accepted IDs/bases/seeds and final
model SHA-256 `cf700754f7729a6aeb6dac0eac7c50b9f9ff7f151e9bc92ac7017b7bdf889ff4`.
All six method/mode server configurations reconstruct from isolated exports.
Only three focused research unit cases are added; full dataset accuracy,
CUDA execution and distributed workflow recovery remain outside this validation.

The combined focused suite has 39 cases; the affected Collab/catalog/KD/research
and documentation-link suite has 295 passing cases. Full project style, explicit
research style, license headers and unit-test timing contracts pass. Fresh
independent review against the dependency baseline reports zero findings and
independently reruns the reference schedule and DFKD comparisons.

Failure handling now follows the core fail-fast contract: physical RPC errors
are surfaced when received, including during horizon draining. The old harness
could defer an error until its logical upload event or discard it at shutdown.
No retry or distributed recovery is introduced.

### Dependency refresh validation

Live and plan-driven runs now use the merged `FedBuffRecipe` API, with `name`
and `num_rounds`. The CLI passes explicit plans to `SimEnv` and keeps the
existing paper scheduler as the default. The shared replay runtime keeps a
logical client active through acceptance. The paper harness permits a new
assignment after the previous assignment's training phase, before its upload
arrives, and preserves its own seeded timing and reservation policies.

The refreshed checkout passed 83 focused FedBuff, replay, KD and application
tests across the two local Python environments, plus scoped style, license
and relative link checks. Six runs through the local simulator used synthetic
CIFAR-shaped data and a small classifier. Live mode published four versions.
Fixed replay accepted `[1, 0, 2, 3]` with matching seeds, bases and exact final
weights across one and two physical workers. Timed replay stopped at virtual
time 5 after two publications. The paper harness also matched its accepted
trace and exact final weights across one and two workers. The smoke runs
exercise the composition before the synthesis/KD warmup gates.

## Stage 6: independent consumers

Stage 5 is stable for the tested seeded simulated-arrival configuration. The
subsequent bounded real-data CUDA comparison recorded in
[the roadmap](https://github.com/NVIDIA/NVFlare/issues/5572) covers 250
publications through synthesis/KD warmup, with aligned curves and accepted
schedules. It does not establish full paper-length or multi-seed equivalence.

Stage 6 demonstrates reusable components in other applications:

- The existing [async CIFAR-10 example](../../examples/advanced/collab/pt_async_cifar10/README.md)
  already supplies the independent FedBuff consumer (`FedBuffRecipe`, concurrency=8, buffer_size=4, min_open_slots=1).
- The new [synchronous CIFAR-10 KD example](../../examples/advanced/collab/pt_cifar10/fedkd/README.md)
  uses the native `DistillationTrainer` through the `Distiller` session contract
  in ordinary `CollabRecipe` rounds. It reuses the FedAvg client and compares weight averaging with KD on
  identical private shards. A server proxy subset is reserved before client
  partitioning; the official test set is evaluation-only. Neither async nor
  FedRevive is required by that consumer.

Start with supplied proxy batches and the same model architecture. Optional
synthetic-batch consumers and broader execution/accuracy qualification can
follow. FedRevive ablation controls are not the acceptance criterion for this
stage. Keep task-specific preparation, models and defaults in examples, and add
shared helper behavior only when an independent consumer needs it.

The first KD consumer adds two focused tests: complete, disjoint and repeatable
proxy/client partitions; and KD that improves teacher agreement without
averaging parameters or mutating teachers. The 297-case affected suite passes.
Both aggregation modes complete two rounds through the real Collab simulator
using small synthetic CIFAR-shaped data, with identical initial models. Both
exported configurations reconstruct and aggregate in isolated interpreters
against a freshly built/installed NVFlare wheel, without source-checkout or
FedRevive imports. Project style, license and documentation-link checks pass.
These checks establish execution and reuse, not real-data accuracy improvement;
the heterogeneous CIFAR-10 comparison remains to be run.
