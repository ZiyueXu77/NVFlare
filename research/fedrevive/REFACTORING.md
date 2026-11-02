# FedRevive component migration

This work implements Stage 4 of [the FedRevive roadmap](https://github.com/NVIDIA/NVFlare/issues/5572).
The async and KD backbones remain independent. This branch composes them and
keeps CIFAR architecture, augmentation, experimental schedules and algorithm
presets in the research application.

## Branches and pinned inputs

| Branch | Contents | Review target |
| --- | --- | --- |
| `codex/fedrevive-dependencies` | Combined async and KD PR heads, without FedRevive edits | Dependency baseline only |
| `codex/fedrevive` | FedRevive changes above that baseline | `codex/fedrevive-dependencies` until both PRs merge |

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
5. **Qualification and ablations.** Compare fixed updates and a small seeded run,
   validate export and one/two-worker equivalence, then demonstrate component
   omission/replacement. Record intentional numerical changes explicitly.

## Preservation checks and current progress

- [x] Combine the two pinned PR heads without changes to either dependency.
- [x] Route ordinary student KD through the shared loss; retain synthesis KL
  with teacher-input gradients.
- [ ] Compose the FedRevive update transformation and shared teacher history.
- [ ] Migrate research synthesis/pool usage and phase scheduling.
- [ ] Add actual-arrival composition and compare seeded end-to-end behavior.

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
headers and changed Markdown links pass. The phase-scheduler and complete
teacher/synthesis migration remain unchecked above.
