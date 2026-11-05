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

Initial dependency baseline: `474c0e27608ebda5457b92ca58a25ce565c1cf92`.
The baseline merges these commits without changing their source branches:

| Dependency | Branch | Pinned head |
| --- | --- | --- |
| [Async recipe, PR #5573](https://github.com/NVIDIA/NVFlare/pull/5573) | `codex/generalizable-async-recipe` | `03ac789dabf4303a44fcd4238db3674ad7d2c6b5` |
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
4. **Async workflows.** Compose actual-arrival training from `AsyncRecipe`.
   Migrate the seeded download/train/upload scheduler to the research workflow,
   reusing `AsyncTrainingCoordinator`, Collab calls and disk-first artifacts.
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
generation interval, tracker resets and method K/B/O/server-LR presets.

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

`FedReviveWorkflow` reuses `AsyncWorkflow` calls and lifecycle. Its phase scheduler
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

Stage 5 is delivered through an independent FedDF example branch,
`codex/feddf-example-curves`. Wait for #5573 and #5613 to merge before opening
its upstream PR; rebase onto the merged dependencies and remove inherited
Stage 4 changes from that PR's diff.

- The existing [async CIFAR-10 example](../../examples/advanced/collab/pt_async_cifar10/README.md)
  supplies the independent FedBuff consumer (`AsyncRecipe`, K=8/B=4/O=1).
- The [synchronous FedDF example](../../examples/advanced/collab/pt_cifar10/feddf/README.md)
  replaces the earlier KD demonstration. It reuses the FedAvg client and model,
  initializes the student from uniformly averaged current client parameters,
  and supplies a mean-logit ensemble to `DistillationTrainer`. Its distilled
  student becomes the next round's global model. It requires neither async
  execution nor FedRevive imports.

The example includes measured FedAvg/FedDF round 0–100 curves, CSV and protocol
metadata from a separate, code-aligned ResNet-8-BN reference experiment. The
compact runnable demo uses different data/training defaults. The single-seed
curves show 39.88% versus 69.28% accuracy at round 100; neither reaches the
75% target, so paper target-round counts are not claimed reproduced.

Final validation and any execution/recovery limitations are tracked in
[issue #5572](https://github.com/NVIDIA/NVFlare/issues/5572). Earlier generic-KD
validation does not qualify the replacement FedDF implementation. No SDK or
FedRevive algorithm changes are needed for this consumer.

Replacement validation (2026-11-04): the two focused example cases pass,
including a direct one-step Adam comparison that distinguishes mean-logit KL
from mean per-teacher KL and verifies average initialization, buffer policy
and teacher immutability. Both modes complete two rounds with two real Collab
simulator clients on 80 private CIFAR-10 images and 16 proxy images, with equal
initial states and finite final states; evaluation uses the official test set.
Recipe CLI export and imports from the exported source tree pass. Scoped
Black/isort/flake8, Markdown asset links and unit-test timing contracts pass.
The affected example/export/KD-helper run reports 18 passing cases and the
previously recorded PyTorch 2.11 optional synthesis meta-device failure
(`Tensor.item() cannot be called on meta tensors`). That synthesis path is not
used by FedDF. These smoke checks do not rerun the long ResNet reference pair
or establish installed-wheel/distributed recovery qualification.

Replaceable-distiller follow-up (2026-11-05): the working branch includes the
pending KD dependency update from #5613 (`20fafba47`, cherry-picked here).
The FedDF consumer now injects an exportable `Cifar10DistillerFactory`, with
native/ModelOpt/torchdistill selection. Its round loop uses only `fit`,
`export_student_state` and `close`, loads the student snapshot for broadcast,
and closes in `finally`. The 0–100 reference curves are unchanged.

The SDK affected suite passes 100 cases, and 17 new session contracts pass
against an isolated wheel. Optional imports are lazy; native training works
when both external libraries are unavailable. The meta-device allocation test
now uses SGD, resolving the previously recorded Adam compatibility issue.
All three distillers complete the same bounded two-client/two-round CPU Collab
run on the existing 80-private/16-proxy CIFAR split, with identical initial and
final model tensors. The example/export checks pass seven cases. These are
smoke/numerical checks, not new paper-accuracy, GPU or recovery qualification.
The eventual Stage 5-only PR must exclude this pending SDK dependency update
as well as the inherited Stage 4 changes after dependencies merge.
