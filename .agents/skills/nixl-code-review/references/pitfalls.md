# Code Review Pitfalls

Use this reference during the final finding-quality pass.

## Common Pitfalls

- Reviewing broad unchanged context instead of changed hunks and the smallest
  source needed to understand them.
- Treating current NIXL `main` behavior as proof for an older package, image, or
  framework integration under review.
- Raising a backend, CUDA, connector, or telemetry concern without a source path
  or changed-code link.
- Approving tests that mock away the NIXL surface changed by the diff.
- Writing style or architecture preferences as findings when they are not tied
  to a concrete failure mode.
- Recommending a fix that requires running installers, mutating clusters, or
  changing runtime config when the task was review-only.
- Leaving speculative findings in the final output after the evidence pass.

## Recovery Moves

- Re-read every finding against the diff and delete it if it is not actionable,
  source-backed, and tied to changed code.
- Downgrade missing evidence to `Question` when it could change the conclusion.
- If runtime readiness blocks review, stop and request an install-readiness
  hand-off instead of inventing environment behavior.
