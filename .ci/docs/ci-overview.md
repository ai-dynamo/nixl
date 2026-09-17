# NIXL CI Overview

This document catalogs every CI job in the NIXL repository — GitHub Actions
workflows (`.github/workflows/`) and Jenkins jobs triggered through
Blossom-CI (`.ci/jenkins/`) — and states whether each one runs
**automatically on every PR**, or is a **standalone/manual** job that only
runs on-demand (`workflow_dispatch`, a PR comment, or a cron schedule).

## Quick reference

| Job | System | Trigger | Automatic on every PR? |
|---|---|---|---|
| [NVIDIA NIXL Validation](#nvidia-nixl-validation-build_validationyml) (`mirror_repo`, `trigger-ci`) | GitHub Actions | `push` to `main` / `pull-request/<n>` | Yes |
| [AWS NIXL Validation](#aws-nixl-validation-aws_efa_validationyml) | GitHub Actions | `push` to `main` / `pull-request/<n>` | Yes |
| [Clang Format Check](#clang-format-check-clang-formatyml) | GitHub Actions | `pull_request` | Yes |
| [Copyright Checks](#copyright-checks-copyright-checksyml) | GitHub Actions | `pull_request` | Yes |
| [PR Size Check](#pr-size-check-pr-size-checkyml) | GitHub Actions | `pull_request` | Yes |
| [Python Checks](#python-checks-python-checksyml) | GitHub Actions | `pull_request` | Yes |
| [Run Pre-Commit Hooks](#run-pre-commit-hooks-pre-commityml) | GitHub Actions | `push`, `pull_request` | Yes |
| [Claude Code Review](#claude-code-review-claude-reviewyml) | GitHub Actions | `pull_request` (opened/synchronize/reopened) | Yes |
| [External Contributor](#external-contributor-external_contributoryaml) | GitHub Actions | `pull_request_target` (opened, fork only) | Yes (fork PRs only) |
| [Blossom-CI](#blossom-ci-blossom-ciyml) | GitHub Actions | `/build` PR comment, or `workflow_dispatch` | No — manual |
| `nixl-ci-dispatcher` → `non-gpu`, `gpu`, `dl-gpu`, `dl-gpu-ep`, `build-wheel`, `test-sanitizers`, `build-container-pr` | Jenkins (dispatcher-triggered) | Fan-out from Blossom-CI `Job-trigger` | No — only after `/build`, but these 7 are the *only* Jenkins jobs in the PR CI path |
| `nixl-ci-build-container` | Jenkins (standalone) | Nightly cron + manual | No — never runs as part of PR CI |
| `nixl-ci-build-wheel-nightly` | Jenkins (standalone) | Nightly cron, triggered by `build-wheel-release-poller`, or manual | No — never runs as part of PR CI |
| `nixl-build-wheel-release-poller` | Jenkins (standalone) | 4-hourly cron + manual | No — never runs as part of PR CI |
| `nixl-ci-build-llm-container` | Jenkins (standalone) | Manual only | No — never runs as part of PR CI |
| `nixl-ci-test-llm-container` | Jenkins (standalone) | Manual, or chained from `build-llm-container` via `RUN_TEST` | No — never runs as part of PR CI |
| `nixl-ci-cleanup-artifacts` | Jenkins (standalone) | Daily cron (6 AM) + manual | No — never runs as part of PR CI |
| `nixl-ci-nightly` | Jenkins (standalone) | Nightly cron (`H 0`) + manual | No — orchestrates the nightly UCX-`master` run of the 3 GPU jobs |

> **Note on Jenkins jobs:** `proj-jjb.yaml` defines 15 Jenkins jobs: the
> dispatcher, the 7 jobs it fans out to (the PR CI flow), and 7 standalone jobs
> (`build-container`, `build-wheel-nightly`, `build-wheel-release-poller`,
> `build-llm-container`, `test-llm-container`, `cleanup-artifacts`, `nightly`). The
> standalone ones run only on their own cron, when someone triggers them manually
> from the Jenkins UI, or when chained from another standalone job,
> and are never invoked by the dispatcher or by a PR event.

## GitHub Actions workflows

All files below live in `.github/workflows/`.

### NVIDIA NIXL Validation (`build_validation.yml`)
- **Trigger:** `push` to `main` or to a `pull-request/<n>` ref (GitHub auto-creates this ref for open PRs).
- **What it does:** `mirror_repo` syncs the repo to an internal GitLab mirror; `trigger-ci` then POSTs to a GitLab pipeline (`PIPELINE_URL`) using the same ref, kicking off the internal GitLab CI pipeline.
- **Automatic on every PR:** Yes — this is the trigger that starts the internal GitLab-side pipeline for every push/PR update.

### AWS NIXL Validation (`aws_efa_validation.yml`)
- **Trigger:** same as above — `push` to `main` / `pull-request/<n>`.
- **What it does:** Runs the AWS EFA validation test suite (`run_aws_tests`) against AWS infrastructure.
- **Automatic on every PR:** Yes.

### Clang Format Check (`clang-format.yml`)
- **Trigger:** `pull_request`.
- **What it does:** Runs `clang-format-19` over the C/C++ lines changed in the PR (excluding `examples/device/ep`) and fails on formatting violations.
- **Automatic on every PR:** Yes.

### Copyright Checks (`copyright-checks.yml`)
- **Trigger:** `pull_request`.
- **What it does:** Runs `.github/workflows/copyright-check.sh` inside the `dynamo/helm-tester` container to verify SPDX/copyright headers.
- **Automatic on every PR:** Yes.

### PR Size Check (`pr-size-check.yml`)
- **Trigger:** `pull_request`.
- **What it does:** Fails the PR if it changes more than 500 lines (excluding `subprojects/*`).
- **Automatic on every PR:** Yes.

### Python Checks (`python-checks.yml`)
- **Trigger:** `pull_request`.
- **What it does:** Runs `.ci/scripts/check_prints.sh` against `./src`, `./test`, `./benchmark`, `./examples/python` to flag stray `print()` calls.
- **Automatic on every PR:** Yes.

### Run Pre-Commit Hooks (`pre-commit.yml`)
- **Trigger:** `push`, `pull_request`.
- **What it does:** Runs the repo's configured `pre-commit` hooks against files modified in the PR/push.
- **Automatic on every PR:** Yes.

### Claude Code Review (`claude-review.yml`)
- **Trigger:** `pull_request` (`opened`, `synchronize`, `reopened`).
- **What it does:** Generates the PR diff and runs an automated Claude-based code review, posting results as PR feedback.
- **Automatic on every PR:** Yes.

### External Contributor (`external_contributor.yaml`)
- **Trigger:** `pull_request_target`, `opened`.
- **What it does:** Posts a reminder comment and adds the `external-contribution` label when the PR author is not a member of the `ai-dynamo` organization.
- **Automatic on every PR:** Only fires for PRs from non-members — a no-op job condition otherwise.

### Blossom-CI (`blossom-ci.yml`)
- **Trigger:** `issue_comment` (only proceeds `if` the comment body is exactly `/build`), or manual `workflow_dispatch`.
- **What it does:** `Authorization` validates the commenter is authorized; `Vulnerability-scan` checks out the PR code and runs the Blossom vulnerability scan action; `Job-trigger` calls `blossom-ci` with `OPERATION: START-CI-JOB`, which kicks off the Jenkins `nixl-ci-dispatcher` job (see below); `Upload-Log` (on `workflow_dispatch` only) links the Jenkins build log back to the PR.
- **Automatic on every PR:** **No.** This is a maintainer-gated, manual step — someone with commit access must comment `/build` on the PR to start the full Jenkins pipeline.

#### Blossom-CI flow

```mermaid
sequenceDiagram
    participant User
    participant GH as GitHub PR
    participant Blossom as Blossom-CI (GH Action)
    participant Auth as Authorization
    participant Scan as Vulnerability-scan (Black Duck)
    participant Trigger as Job-trigger
    participant Jenkins as nixl-ci-dispatcher
    participant Children as Child jobs<br/>(non-gpu, gpu, dl-gpu,<br/>dl-gpu-ep, build-wheel,<br/>test-sanitizers, build-container-pr)

    User->>GH: comment "/build"
    GH->>Blossom: issue_comment event
    Blossom->>Auth: run Authorization job
    Auth->>Auth: is commenter NVIDIA + Blossom-SRE authorized?
    alt not authorized
        Auth-->>GH: fail / stop (no CI run)
    else authorized
        Auth->>Scan: proceed to Vulnerability-scan
        Scan->>Scan: checkout PR code, run Black Duck scan
        Scan->>Trigger: proceed to Job-trigger
        Trigger->>Jenkins: START-CI-JOB (trigger dispatcher)
        Jenkins->>Children: fan out in parallel, wait for all
        Children-->>GH: each job posts its own check status
        Jenkins-->>GH: dispatcher posts overall commit status
    end
```

Step by step, matching the jobs in `blossom-ci.yml`:

1. A user comments `/build` on the PR.
2. The `Blossom-CI` GitHub Action wakes up on the `issue_comment` event.
3. **Authorization** checks the commenter is allowed to trigger CI: non-NVIDIA
   users cannot trigger it at all, and NVIDIA employees need prior
   authorization from the Blossom SRE team. Unauthorized comments stop here.
4. **Vulnerability-scan** checks out the PR's code and runs a Black
   Duck-based vulnerability scan via the `NVIDIA/blossom-action`.
5. **Job-trigger** calls `blossom-ci` with `OPERATION: START-CI-JOB`, which
   triggers the Jenkins `nixl-ci-dispatcher` job.
6. `nixl-ci-dispatcher` fans out in parallel to its seven child jobs
   (`non-gpu`, `gpu`, `dl-gpu`, `dl-gpu-ep`, `build-wheel`,
   `test-sanitizers`, `build-container-pr` — see [Jenkins jobs](#jenkins-jobs) below).
7. Each child job reports its own status back as an individual GitHub PR
   check, so the PR shows per-job pass/fail rather than one aggregate check.

## Jenkins jobs

All 14 Jenkins jobs are defined in `.ci/jenkins/pipeline/proj-jjb.yaml`
(Jenkins Job Builder config). The dispatcher runs its own pipeline,
`.ci/jenkins/pipeline/Jenkinsfile.dispatcher`, checked out from the PR merge
ref (`refs/pull/<n>/merge`) on webhook runs, or from any branch/commit passed
in `sha1` on manual runs; all other jobs run through the shared pipeline
entry point `.ci/jenkins/pipeline/Jenkinsfile`. None of them run directly off GitHub
events — they only start via the Jenkins webhook fired by Blossom-CI, or via
their own nightly/manual trigger. They split into two groups:

- **Dispatcher-triggered (part of the PR CI path):** `nixl-ci-dispatcher` and
  the 7 jobs it fans out to. This is the *only* way any Jenkins job runs
  against a PR, and only after a `/build` comment.
- **Standalone (never run against a PR):** `nixl-ci-build-container`,
  `nixl-ci-build-wheel-nightly`, `nixl-build-wheel-release-poller`,
  `nixl-ci-build-llm-container`, `nixl-ci-test-llm-container`,
  `nixl-ci-cleanup-artifacts`, `nixl-ci-nightly` — each has its own cron, manual
  trigger, and/or upstream standalone job, and is invoked independently of PRs
  and of the dispatcher.

### `nixl-ci-dispatcher` (dispatcher-triggered)

- **Trigger:** GitHub webhook payload forwarded by Blossom-CI's `Job-trigger` step (`OPERATION: START-CI-JOB`). Not a raw GitHub Actions event.
- **What it does:** Fans out in parallel to seven downstream Jenkins jobs, waiting on all of them:
  - `nixl-ci-non-gpu` — `.ci/jenkins/lib/build-matrix.yaml`
  - `nixl-ci-gpu` — `.ci/jenkins/lib/test-matrix.yaml`
  - `nixl-ci-dl-gpu` — `.ci/jenkins/lib/test-dl-matrix.yaml` (dlcluster.nvidia.com)
  - `nixl-ci-dl-gpu-ep` — `.ci/jenkins/lib/test-dl-ep-matrix.yaml` (NIXL EP tests on dlcluster.nvidia.com)
  - `nixl-ci-build-wheel` — `.ci/jenkins/lib/build-wheel-matrix.yaml`
  - `nixl-ci-test-sanitizers` — `.ci/jenkins/lib/test-sanitizer-matrix.yaml` (ASan/UBSan + TSan)
  - `nixl-ci-build-container-pr` — `.ci/jenkins/lib/build-container-pr-matrix.yaml`
- **UCX version:** The three GPU test jobs (`nixl-ci-gpu`, `nixl-ci-dl-gpu`, `nixl-ci-dl-gpu-ep`) build and test against a single UCX version per run — the `UCX_VER` parameter, which defaults to empty and falls back to the `Dockerfile` `ARG UCX_VERSION` default (`v1.23.x`). UCX `master` is validated nightly, not per PR: the standalone `nixl-ci-nightly` job (see below) fans out to all three with `UCX_VER=master` and emails one consolidated report, so UCX regressions surface outside the PR path instead of blocking PRs.
- **Automatic on every PR:** No — only runs after a `/build` comment triggers Blossom-CI. The dispatcher also aborts any stale in-flight dispatcher run for the same PR (and the leaf builds it started) before starting.

### `nixl-ci-build-container-pr` (dispatcher-triggered)

- **Trigger:** Fan-out from `nixl-ci-dispatcher` (same as the other PR CI jobs).
- **What it does:** Build-only verification (no push) of the `nixl` (EP + debug) and `nixlbench` container images — one matrix cell per target/arch (x86_64 and aarch64), all in parallel. It runs the same `contrib/build-container.sh` / `benchmark/nixlbench/contrib/build.sh` the standalone `nixl-ci-build-container` job runs, so container/packaging breakage (e.g. a missing `--torch-versions`, or an EP nvlink register-count failure) is caught on the PR instead of only by the nightly job. Like the other leaf jobs it runs whenever the dispatcher fans out. It passes no base-image flags, so each target builds on the base image pinned in its own Dockerfile (`nvcr.io/nvidia/cuda-dl-base`) — the gate therefore exercises the base the shipped images actually use. It previously pinned a DLFW PyTorch base, which skipped the torch and DOCA SDK installs that `cuda-dl-base` requires, so the build path under test diverged from the released one.
- **Automatic on every PR:** No — only after a `/build` comment (like the other dispatcher jobs).

### `nixl-ci-build-wheel` (dispatcher-triggered)
- **Config:** `.ci/jenkins/lib/build-wheel-matrix.yaml`
- **What it does:** Builds NIXL Python wheels for each Python version × architecture combination. Uses a two-stage `contrib/Dockerfile.manylinux` build: the `wheel_base` stage (all slow deps: UCX, gRPC, Rust, etc.) is pre-built and cached in Artifactory under `CI_IMAGE_TAG`; PR builds pull this pre-built image and run only the `wheel` stage (~16 steps). PR builds also pass `--torch-versions` (set via the `TORCH_VERSIONS` env in the matrix file) to limit the torch extension builds to the latest version. x86_64 wheels build in the `manylinux` (podman-in-container) runner.
- **vLLM/SGLang NIXL sanity (aarch64):** the same job also gates a 1-prefill/1-decode NIXL KV-transfer sanity for vLLM and SGLang. Each framework runs as its own aarch64 branch (`build_helper_vllm` / `build_helper_sglang`): build the aarch64 wheels (reusing the cached `wheel_base`), layer them onto the pinned framework base image (`.ci/dockerfiles/Dockerfile.{vllm,sglang}-base`, built as `category: tool` by ci-demo), then run the sanity on the `gb200nvl72_ci` dlcluster SLURM partition over SSH (`.gitlab/test_vllm_sglang_sanity.sh`). SGLang additionally asserts a gsm8k accuracy floor on `Qwen/Qwen3-8B`. The aarch64 wheels are built once per framework branch (duplicated) so the two flows stay separate, labeled branches in the Jenkins UI. The sanity model is prefetched through the internal HuggingFace mirror in `SANITY_HF_ENDPOINT` rather than `huggingface.co` (which rate-limits the shared CI egress IP); each sanity step passes it into the SLURM container as `HF_ENDPOINT`.
- **`CI_IMAGE_TAG`:** Same convention as the other five matrix files (also tags `build_helper_*` and the sanity `*-nixl-base` images). `contrib/Dockerfile.manylinux` is part of the `CI_FILES` list in `cidemo-init.sh`, so changing it (or any other CI file) automatically derives a new `CI_IMAGE_TAG` and rebuilds the cached `wheel_base` image (see [CI_IMAGE_TAG management](#ci_image_tag-management)).

### `nixl-ci-build-container` (standalone)
- **Trigger:** Nightly cron (builds `nixlbench` and `nixl` targets, on both the default CUDA base image and the PyTorch release image `nvcr.io/nvidia/pytorch:26.06-py3`, ~22:00), or manual run with parameters (`BUILD_TARGET`, `NIXL_VERSION`, `UCX_VERSION`, base image overrides, etc.).
- **What it does:** Builds and pushes x86_64/aarch64 NIXL/NIXLBench container images to Artifactory, then sets build metadata properties on each image via the Artifactory REST API. The Push step runs with `set -eo pipefail` so auth or API failures abort the build immediately. `BUILD_UCX_SPCX_PLUGIN` (default on, `nixl` target only) builds the internal UCX spcx plugin against the image's UCX and installs it into the UCX plugin dir; it needs the `svc-nixl-gitlab-token` + `ucx-plugin-gitlab-url` credentials, bound on the Build NIXL step. `BUILD_NIXL_EP` is on by default and has no off-switch — `contrib/build-container.sh` builds EP regardless. `BUILD_INFINIA` (default off, `nixl` target only) stages the DDN libs from harbor.mellanox.com and builds `libplugin_INFINIA.so` into the image; it is off by default because it adds an external registry dependency.
- **Image tag:** `pipeline_start` resolves the NIXL and UCX commits once (`NIXL_SHA`, `UCX_SHA`) and exports `IMAGE_TAG_BASE`; both arches build from those exact commits and tag from that base, and `pipeline_stop` reuses it to name the images. Previously each arch re-resolved UCX independently, so a branch moving mid-build could ship two arches from different UCX commits under one tag.
- **Completion mail:** When `MAIL_TO` is set, `pipeline_stop` mails the build result, listing the pushed image URLs (and the `-latest` tags when `UPDATE_LATEST`) so the verification team can pull them directly. The list is included only on `SUCCESS` — a failed run may not have pushed, and a link to a missing image is worse than no link.
- **Automatic on every PR:** No — standalone/nightly + manual only.

### `nixl-ci-build-wheel-nightly` (standalone)
- **Trigger:** Nightly cron (two runs, `CUDA_MAJOR=13` and `CUDA_MAJOR=12`), triggered by [`nixl-build-wheel-release-poller`](#nixl-build-wheel-release-poller-standalone) for release publishing, or manual run. The pipeline and matrix config run from `ci_refspec` (default `main`; pass `refs/pull/<n>/head` to test CI changes end to end before merge); the NIXL source is cloned inside the build from the `NIXL_VERSION` parameter (branch/tag/PR ref/sha), so any ref is buildable without CI files on it.
- **What it does:** Reuses the per-PR wheel build path (`contrib/build-container.sh` + `Dockerfile.manylinux`) and publishes wheels to `sw-nbu-swx-nixl-pypi-local`. With `PUBLISH_DIR` empty (the default, and what the nightly cron uses) wheels land under `verification/g<nixl-sha8>.ucx<ucx-sha8>/` — the long-standing schema the `build-llm-container` verification pipeline consumes; the poller and manual release runs pass `release/<ver>`, which co-locates cu12/cu13 under `release/<ver>/<nixl-sha8>/` (unkeyed to UCX). `CUDA_MAJOR` selects the CUDA line to build: `13` (default) passes no base-image flags to `build-container.sh` and relies on its own defaults; `12` passes the pinned CUDA 12 base image/tag from the matrix env. The UCX spcx and Infinia DDN plugins are always bundled (`--build-ucx-spcx-plugin --build-infinia`, unconditional — not job parameters); the source ref's `contrib/build-container.sh` must carry both flags and pin their versions. The resolved build options (base image, UCX ref/sha, plugin flags, etc.) are written to `build_options.env` via `--build-options-file`; Publish reads it and attaches `UCX_REF`, `UCX_SHA`, `CUDA_VERSION`, and (when built) `UCX_SPCX_PLUGIN_REF`/`INFINIA_LIBS_IMAGE` as Artifactory properties on each uploaded wheel (skipped if the built ref's `build-container.sh` predates `--build-options-file`).
- **Automatic on every PR:** No — standalone nightly/poller-triggered + manual only.

### `nixl-build-wheel-release-poller` (standalone)

- **Trigger:** 4-hourly cron (`H H/4 * * *`) or manual run. The pipeline and matrix config (`.ci/jenkins/lib/build-wheel-release-poller-matrix.yaml`) run from `ci_refspec` (default `main`); the poller forwards `ci_refspec` to the builds it triggers, so a pre-merge test run drives the whole chain from one PR ref.
- **What it does:** Builds release wheels for every `release/*` branch with version >= 1.4.0 whose `contrib/build-container.sh` accepts `--build-options-file` (the nightly always passes it, so older refs would fail at option parsing) - new release branches are picked up automatically, with no CI config anywhere. The CUDA variants are the matrix axis (`cuda_major`): each cell runs `.ci/scripts/scan-missing-release-wheels.sh` for its variant and triggers the missing builds. Per release the scan takes the newest 10 first-parent commits past the merge-base with `main`, checks the Artifactory folder `release/<ver>/<nixl-sha8>/` for that variant's wheel presence, and triggers [`nixl-ci-build-wheel-nightly`](#nixl-ci-build-wheel-nightly-standalone) once per missing build, passing the commit sha, `CUDA_MAJOR`, and `PUBLISH_DIR=release/<ver>`. UCX and the bundled plugins are not passed: each release builds against the `UCX_REF` and plugin versions pinned in its own `contrib/build-container.sh`. No marker files: a failed build is retried on later cycles while its commit stays within the newest-10 window, and a partially-uploaded variant looks complete; delete the folder in Artifactory to force a rebuild.
- **Automatic on every PR:** No — standalone cron + manual only, never part of the PR CI path.

### `nixl-ci-build-llm-container` (standalone)
- **Trigger:** Manual only (no cron, no webhook).
- **What it does:** Builds the 4 LLM inference container variants (`vllm-nixl`, `vllm-cu12-nixl`, `sglang-nixl`, `sglang-cu13-nixl`) for x86_64/aarch64 from a published NIXL wheel set, publishes multi-arch manifests, and optionally (`RUN_TEST`) fires `nixl-ci-test-llm-container` per built variant.
- **Automatic on every PR:** No — standalone/manual only, used for release verification.

### `nixl-ci-cleanup-artifacts` (standalone)
- **Trigger:** Daily cron at 6 AM, or manual run with optional `DRY_RUN=true` parameter.
- **What it does:** Deletes stale Artifactory artifacts based on `.ci/cleanup-spec.json` — PR Docker images older than 1 day, CI base images not pulled in 2 weeks, verification Docker images and PyPI wheels older than 3 months.
- **Matrix:** `.ci/jenkins/lib/cleanup-matrix.yaml`. Runs on an Ubuntu 24.04 container with `jf` and `jq` installed at runtime.
- **Automatic on every PR:** No — standalone/scheduled + manual only.

### `nixl-ci-test-llm-container` (standalone)
- **Trigger:** Manual, or chained asynchronously from `nixl-ci-build-llm-container` when `RUN_TEST` is enabled.
- **What it does:** Runs smoke/perf/accuracy tests for one published LLM inference container image on the `mizu` SLURM partition (2-GPU node); framework (vllm/sglang) is auto-detected from the image URL.
- **Automatic on every PR:** No — standalone/manual only.

### `nixl-ci-nightly` (standalone)

- **Trigger:** Nightly cron (`H 0 * * *`), or manual run (`UCX_REF`, `MAIL_TO` parameters).
- **What it does:** Fans out to `nixl-ci-gpu`, `nixl-ci-dl-gpu`, `nixl-ci-dl-gpu-ep` with `UCX_VER=${UCX_REF}` (default `master`), waits for all three, and emails one consolidated report to `MAIL_TO` (default `nixl-ci-alerts@exchange.nvidia.com`) **only when a leg fails** — a green night is silent. This is the single place nightly UCX-`master` results are collected and sent from; per-PR runs of the GPU jobs cover only the release UCX version.
- **Matrix:** `.ci/jenkins/lib/nightly-matrix.yaml` — a lightweight ci-demo orchestrator (one groovy step: fan out, wait, mail), no GPU of its own.
- **Automatic on every PR:** No — standalone/scheduled + manual only.

## Slurm job naming

Jobs submitted via the `slurmCI` module are named `${JOB_BASE_NAME}-${BUILD_NUMBER}`; jobs that run multiple parallel allocations within a build insert a `<variant>` before the build number to disambiguate:

| Pipeline | Slurm job name pattern |
|---|---|
| `nixl-ci-gpu` | `nixl-ci-gpu-<build>` |
| `nixl-ci-dl-gpu` | `nixl-ci-dl-gpu-<build>` |
| `nixl-ci-dl-gpu-ep` | `nixl-ci-dl-gpu-ep-<build>` |
| `nixl-ci-build-wheel` | `nixl-ci-build-wheel-<fw>-<build>` (`fw`: `vllm` or `sglang`) |
| `nixl-ci-test-llm-container` | `nixl-ci-test-llm-container-<build>` |

Use `squeue --name <pattern>` or `squeue -u <user>` to identify which pipeline owns a running job.

## How to trigger CI manually

- **Full Jenkins pipeline for a PR:** comment `/build` on the PR (requires authorization — see `Authorization` step in `blossom-ci.yml`).
- **Re-run a single Jenkins job with different parameters:** use `workflow_dispatch` on `blossom-ci.yml`, or trigger the Jenkins job directly if you have Jenkins access.
- **Container/wheel builds outside the nightly schedule:** run `nixl-ci-build-container` or `nixl-ci-build-wheel-nightly` manually from the Jenkins UI with custom parameters.

## CI_IMAGE_TAG management

`CI_IMAGE_TAG` is the Docker image tag used by all matrix jobs to identify the
base images they build and pull. It appears as `CI_IMAGE_TAG: "CI_MANAGED"` in
the six matrix YAML files — this placeholder is intentional and must not be
replaced with a static value.

At the start of every Jenkins run, `cidemo-init.sh` derives the real tag
automatically:

```bash
NEW_TAG=$(git log -1 --format=%h -- "${CI_FILES[@]}")
```

This returns the short git commit hash of the most recent commit that touched
any of the CI source files (`Dockerfile.base`, `Dockerfile.gpu-test`,
`Dockerfile.build_helper`, `nixl_ep_vllm_release_test.patch`, `build.sh`, `common.sh`, `Dockerfile.manylinux`). It
then patches all six YAML files in the Jenkins workspace with `sed` before the
matrix library reads them. No commit or push is made — the patch exists only in
the workspace.

**Caching behaviour:** the derived tag is stable as long as the CI source files
are unchanged. Two PRs that both leave the CI files untouched get the same tag
and share the cached Artifactory images. A PR that changes a Dockerfile gets a
new tag and triggers a rebuild automatically.

**You never need to manually update `CI_IMAGE_TAG`.** The `CI_MANAGED`
placeholder signals this clearly.

## Registry push retries

Transient `HTTP 503 Service Unavailable` responses from Artifactory were failing
otherwise-green builds at the push step, after the image had already built
successfully. Every image push now retries, but the mechanism differs because
`--retry` was only added in podman 5.0 and most push steps run on an older one.

**`build-container-matrix.yaml`** runs on `quay.io/podman/stable:v5.7.1` and uses
the native `--retry 5`. podman already retries this class of error —
`go.podman.io/common/pkg/retry` treats HTTP 502-504 and network errors as
retryable, while failing fast on unauthorized, denied and unknown
name/manifest, so a larger budget never delays a genuine failure. `--retry N`
counts retries *after* the initial push, and the backoff is `2^attempt` seconds
plus 10% jitter. The default of 3 retries therefore spans 1+2+4 = 7 seconds,
which the outages above outlasted; `--retry 5` extends it to 1+2+4+8+16 = 31
seconds over 6 attempts. `--retry-delay` is left unset on purpose: passing it
replaces the exponential backoff with a fixed delay.

**`test-matrix.yaml`, `test-dl-matrix.yaml`, `test-dl-ep-matrix.yaml` and
`build-wheel-matrix.yaml`** push from a `Dockerfile.build_helper` container,
which installs podman from Ubuntu 24.04 apt (4.9.x). podman 5 is not available
for 24.04 — not in `noble`, `noble-updates` or `noble-backports` — so these use
a shell retry loop deliberately matched to the flag: 6 attempts with 1, 2, 4, 8
and 16 second delays. The one behavioural difference is that the loop retries on
any failure, so an auth error waits out the backoff instead of failing
immediately.

`build-wheel-matrix.yaml` is the easy one to get wrong: its `Prepare` step
symlinks `docker` to `podman` in two different containers, and the push in
`Build sanity image` runs in the `build_helper_(vllm|sglang)` one, not the
`manylinux` runner. Check the step's `containerSelector` against
`runs_on_dockers`, not just whether `docker` is podman.

## Authenticated github.com clones

The container Dockerfiles build their third-party dependencies (abseil, gRPC,
etcd-cpp-apiv3, aws-sdk-cpp, azure-sdk, gusli, gtest-parallel) from source, which
means ~40 `git clone` calls against github.com per image build. Cloning those
anonymously is unreliable: github.com intermittently answers an anonymous clone
with `HTTP 401` + `www-authenticate: Basic realm="GitHub"`, and unauthenticated
requests are budgeted per source IP — one the Blossom cluster shares across every
tenant, so the failure rate does not track NIXL's own load.

Because git has no credential to offer, a 401 is fatal rather than retried: git
tries to prompt, finds no TTY, and dies with

```
fatal: could not read Username for 'https://github.com': No such device or address
```

That message is misleading in two ways — `github.com` is the credential realm, not
a host it failed to reach, and `No such device or address` is `ENXIO` from opening
`/dev/tty`. Neither has anything to do with DNS or connectivity. With a credential
present git instead answers the 401 by retrying with auth, so the clone succeeds.

**How it is wired.** The `svc-nixl-github-token` credential (already used by
`GithubHelper` for commit statuses) is bound on the image-build steps of
`build-container-pr-matrix.yaml` and `build-container-matrix.yaml` as
`NIXL_GITHUB_USER` / `NIXL_GITHUB_TOKEN`. `contrib/build-container.sh` and
`benchmark/nixlbench/contrib/build.sh` turn those into an env-sourced build secret:

```
--secret id=ghconfig,env=NIXL_GITHUB_GITCONFIG
```

and every `RUN` that clones from github.com mounts it at `/root/.gitconfig`:

```
RUN --mount=type=secret,id=ghconfig,target=/root/.gitconfig git clone ...
```

The token is therefore never written to disk by the build scripts, never lands in
an image layer, and does not appear in `podman history`. It needs no scopes — the
repos are public and the only purpose is to stop being anonymous.

**Local and external builds are unaffected.** With `NIXL_GITHUB_TOKEN` unset the
scripts pass no `--secret`, the mount resolves empty, and the clones stay
anonymous exactly as before. The `ENV GIT_TERMINAL_PROMPT=0` alongside each block
is deliberate: it keeps a future credential problem from reappearing as the same
misleading "could not read Username" message.

**The ci-demo-built images.** Images declared with `file:` in a matrix are built by
ci-demo outside any step (`Matrix.groovy` `buildImage`), so a per-step
`credentials:` entry never reaches them. Two things make it work anyway:

- The token is bound around `matrix.main()` in `.ci/jenkins/pipeline/Jenkinsfile`,
  not just around the `GithubHelper` call, so it is in the environment for the
  image-build phase too.
- `build_args` is spliced verbatim into `docker build`, so those entries carry
  `--secret id=ghconfig,src=${WORKSPACE}/.ghconfig`.
- The config itself is written by a `pipeline_on_image_build` hook calling
  `write_github_gitconfig`. **The hook is load-bearing, not a convenience.** Putting the
  credential in the matrix `env:` block does not work: ci-demo resolves `env:` with
  `resolveTemplate`, a `@NonCPS` Groovy method reading `env.getEnvironment()`, which
  does not see a `withCredentials` binding. `replaceVars` then leaves the unmatched
  `${...}` in place rather than blanking it, so the secret silently becomes a
  literal template string and every clone fails a 401. The hook runs as a real `sh`
  step, where the binding is always present. Build #3095 of `nixl-ci-non-gpu` failed
  exactly this way.

`.ci/dockerfiles/Dockerfile.base` runs as a **non-root** user, so its mount uses
`mode=0444` and mounts straight at `$HOME/.gitconfig`, so nothing is copied and the
file cannot outlive the layer. It logs `github.com clones: authenticated` or `: anonymous` so the mode is
visible at the top of the layer, and CI passes `REQUIRE_GITHUB_AUTH=1` so an empty
secret fails there rather than as a clone failure ten minutes later. Local builds
leave it 0 and clone anonymously as before. That covers `.gitlab/build.sh`'s clones and the `taskflow` /
`prometheus-cpp` meson `wrap-git` subprojects in a single place, for all five
matrices that build from it.

**Why `url.insteadOf` and not a `.netrc`.** git only consults `~/.netrc` from 2.35
onwards. On git 2.34 — which Ubuntu 22.04 ships, and which `build-matrix.yaml` still
builds as the `nixl-ci-non-gpu-base-ubuntu22` variant — a netrc is ignored outright:
the credential is never sent and the clone stays anonymous, with no error to show for
it. `url."https://<user>:<token>@github.com/".insteadOf` works on every version,
because the credential is part of the URL rather than a file git may decline to read.
Verified on 2.34.1 and 2.43.0.

The tradeoff is that the rewritten URL is passed as an argument to `git-remote-https`,
so the token is visible in that process's argv and in `GIT_TRACE` output. Do not enable
`GIT_TRACE`/`GIT_CURL_VERBOSE` in these jobs. git redacts the credential in its own
error messages, keeps it out of `remote.origin.url`, and does not write it into
submodule configs — all verified.

**On CI agents**, where the token is an env var rather than a mounted file,
`setup_github_auth` in `.ci/scripts/common.sh` exports
`GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_n`/`GIT_CONFIG_VALUE_n` (git 2.31+) instead of
writing `$HOME/.gitconfig`. That is deliberate: the config is **additive**, so it
neither skips auth when a `~/.gitconfig` already exists nor deletes one that does.
CI itself writes `git config --global --add safe.directory` on the agent
(`build-wheel-nightly-matrix.yaml`), which a file-based helper would either hide
behind or clobber on cleanup. It also writes the token to no file at all, and
disables `set -x` before expanding it so the value never reaches the log.
`.gitlab/build.sh` and `.gitlab/build-rocm.sh` call it, as do the matrix steps that
clone UCX on the agent.

**Coverage.** Every `RUN` that clones from github.com carries the mount — including
the two `$UCX_REPO` clones in `contrib/Dockerfile` and `contrib/Dockerfile.manylinux`
and the vLLM clone in `Dockerfile.base`, whose URLs come from an ARG or a
continuation line and so are easy to miss when grepping for `github.com` on the
`RUN` line itself. The meson `wrap-file` subprojects fetch release tarballs rather
than cloning and need no credential.

**Still anonymous.** `.ci/dockerfiles/Dockerfile.rocm` (9 clones) is not referenced
by any matrix, so there is nothing to wire a secret through yet. The meson
`wrap-file` subprojects fetch release tarballs rather than cloning; a 401 there
fails the download outright instead of prompting, and they are unaffected by this
change.

## Related docs

- [Build Wheel Matrix CI Job Documentation](build-wheel-matrix-ci.md) — deep dive into `nixl-ci-build-wheel`.
- [Setting up NVIDIA GPU with RDMA support on Ubuntu](setup_nvidia_gpu_with_rdma_support_on_ubuntu.md)
