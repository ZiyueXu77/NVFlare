# FedRevive component migration

This work covers Stage 4 and the first Stage 5 consumer in
[the FedRevive roadmap](https://github.com/NVIDIA/NVFlare/issues/5572).
The async and KD backbones remain independent. This branch composes them and
keeps CIFAR architecture, augmentation, experimental schedules and algorithm
presets in the research application.

## Branches and pinned inputs

| Branch | Contents | Review target |
| --- | --- | --- |
| `codex/fedrevive-dependencies` | Combined async and KD PR heads, without FedRevive edits | Dependency baseline only |
| `codex/fedrevive` | FedRevive changes above that baseline | Compare against `codex/fedrevive-dependencies` until both PRs merge |

Current dependency baseline: `109b6da8dfb4933ffcd4efedf0e08812aa3458b2`.
The initial baseline was `474c0e27608ebda5457b92ca58a25ce565c1cf92`; the current
baseline adds the FedBuff naming and model-component split from the async PR. Its pinned
source heads are:

| Dependency | Branch | Pinned head |
| --- | --- | --- |
| [FedBuff recipe, PR #5573](https://github.com/NVIDIA/NVFlare/pull/5573) | `codex/generalizable-async-recipe` | `a98aa542a799eedd2cf79fe5461e3578e9a0b076` |
| [Standalone KD and synthesis, PR #5613](https://github.com/NVIDIA/NVFlare/pull/5613) | `codex/standalone-kd` | `f021a57b45c60b088988c3cafc3cdb30e5d4207e` |

The recorded commits are source-history pins, not package-version dependencies.
Runtime code imports the shared components directly. Source PR review and merge
approval remain independent; neither dependency is merged by creating these branches.

## Coordinating dependency changes

Use a dedicated clean worktree for these branches. Keep backbone fixes in their
own PR branches, with their focused tests. Bring the reviewed changes into the
dependency branch, then integrate that baseline into the FedRevive branch.
Do not maintain a different copy of a backbone inside this research directory.

For ordinary dependency updates that preserve the pinned history:

```bash
git fetch origin codex/generalizable-async-recipe codex/standalone-kd
git switch codex/fedrevive-dependencies
git merge --no-ff origin/codex/generalizable-async-recipe
git merge --no-ff origin/codex/standalone-kd
git switch codex/fedrevive
git merge --no-ff codex/fedrevive-dependencies
```

Before merging an updated source head, verify that its previously pinned SHA is
still an ancestor (`git merge-base --is-ancestor OLD_SHA NEW_SHA`). If a source
PR is rebased or force-pushed, rebuild the dependency baseline from its new
heads and replay the FedRevive-only commits. Merging rewritten history can keep
implementation that reviewers deliberately removed. Coordinate any resulting
feature-branch history rewrite and use `--force-with-lease` only after inspecting
the current remote head.

After each dependency update, record the new baseline/head SHAs here and in the
roadmap, run the shared tests plus the relevant FedRevive comparison, and push
the dependency baseline before the feature branch. Compare FedRevive changes
against the dependency branch; comparison against `main` also includes both
pending PRs. These are deliberate update checkpoints, not automatic branch polling.

These branches live in the contributor fork. Use its compare view or a fork-local
draft PR for a focused intermediate review; NVIDIA's repository cannot use a
fork-only dependency branch as its PR base. The final upstream PR targets NVIDIA
`main` after the source PRs land.

After both source PRs merge, transplant only the FedRevive changes onto current
NVIDIA `main` and target the final PR there. This is necessary if the dependencies
were squash-merged: simply merging `main` can leave their original commit history
in the feature branch. Starting with a clean, synchronized feature branch:

```bash
dependency_tip=$(git rev-parse codex/fedrevive-dependencies)
git fetch https://github.com/NVIDIA/NVFlare.git main
main_commit=$(git rev-parse FETCH_HEAD)
git rebase --onto "$main_commit" "$dependency_tip" codex/fedrevive
```

Check the resulting diff, retest, and coordinate the history rewrite before
pushing. Retain this migration record until the final review is complete.

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

At each checkpoint run the 22 async and 14 KD helper cases, relevant FedRevive
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

## Stage 4 composition

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

## Stage 5: independent consumers

Stage 4 is stable for the tested seeded simulated-arrival configuration. The
subsequent bounded real-data CUDA comparison recorded in
[the roadmap](https://github.com/NVIDIA/NVFlare/issues/5572) covers 250
publications through synthesis/KD warmup, with aligned curves and accepted
schedules. It does not establish full paper-length or multi-seed equivalence.

Stage 5 demonstrates reusable components in other applications:

- The existing [async CIFAR-10 example](../../examples/advanced/collab/pt_async_cifar10/README.md)
  already supplies the independent FedBuff consumer (`FedBuffRecipe`, concurrency=8, buffer_size=4, min_open_slots=1).
- The new [synchronous CIFAR-10 KD example](../../examples/advanced/collab/pt_cifar10/fedkd/README.md)
  imports `DistillationTrainer` and `kl_divergence` into ordinary `CollabRecipe`
  rounds. It reuses the FedAvg client and compares weight averaging with KD on
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
