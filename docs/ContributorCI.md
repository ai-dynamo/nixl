<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Validate a NIXL contribution

From your NIXL checkout, run:

```bash
./contrib/check.py
```

This checks your branch changes and local edits, then builds NIXL with the POSIX
backend and runs two CPU integration tests. You need Linux, Python 3, Git with
full history, Docker and network access. The checker builds its public Ubuntu
image automatically and caches tool environments and compilation results.
The first run takes several minutes; no GPU or NVIDIA credentials are needed.

## Fixing pre-commit issues

For a quick edit/fix cycle, run NIXL's existing pre-commit hooks directly.
Install the tool and Git hook once in your development environment:

```bash
pip install pre-commit
pre-commit install
```

After staging the files you intend to submit, run:

```bash
pre-commit run
```

Formatting hooks fix Python formatting, import order and whitespace in your
checkout. A run that changes files reports failure so you can review the diff,
stage the fixes and rerun. Type errors and other diagnostics need manual fixes.
Installing the Git hook also runs these checks automatically when you commit.

Other useful scopes are:

```bash
# Files changed by commits on your current branch; substitute your PR target
pre-commit run --from-ref upstream/main --to-ref HEAD

# All tracked files, including files outside your change
pre-commit run --all-files
```

These commands select files to check in your current checkout. To validate an
exact committed tree, use `./contrib/check.py --ref REF` as described below.

Once the fixes pass, run `./contrib/check.py` for repository checks and basic
integration tests, or the backend-specific command below for your backend.
NIXL checks C/C++ formatting separately with clang-format 19; it is not part of
the current pre-commit configuration.

## Checking a commit or branch

```bash
./contrib/check.py --ref HEAD                 # committed changes only
./contrib/check.py --ref my-feature-branch    # without switching branches
./contrib/check.py --ref COMMIT --base COMMIT^ # one commit's changes
```

By default, changes are compared with the merge base of `upstream/main`, falling
back to `origin/main`, then `main`. Fetch your PR target before running; use
`--base REF` for another target. `--ref` selects an exact committed tree and
excludes local edits. Without it, staged, unstaged and non-ignored untracked files
are included. Review `git status` so scratch files are not checked accidentally.

Checks run in a disposable copy, including when launched from a linked worktree.
Your files, branch and index are preserved. Submodules are not supported by the
working-tree snapshot. Shallow clones need `git fetch --unshallow` first.

## What passes mean

| Check | Coverage |
|---|---|
| Pre-commit | The repository's `.pre-commit-config.yaml`, on changed files |
| Other PR checks | Clang-format 19 on changed C/C++ lines, SPDX/copyright, Python logging, PR size, Fern configuration and Markdown |
| Build and load | Debug build with warnings as errors, POSIX explicitly enabled, and loading the built plugin through NIXL |
| Basic integration | Existing `posix_plugin_test` and `posix_aio_backend_test`: file transfers and Linux AIO behavior |

Missing tools or skipped integration tests cannot produce a passing result.
The default POSIX checks exercise a baseline CPU path; they do **not** validate
a newly added backend, GPU/RDMA hardware, sanitizers or the full CI matrix.
CI checks the submitted merge/revision, so rerun after relevant target changes.

The command prints its report directory under `.cache/local-checks/`. It contains
`results.json`, command logs, test results, and the checked source tree identity.
Exit **0** means the selected checks passed, **1** means a check failed, and
**2** means validation is incomplete. For docs-only iteration, `--lint-only`
omits the build and integration tests and reports that narrower scope.

When invoked through `contrib/check.py`, pre-commit runs in the disposable copy
and saves fixes in `formatting.patch`. Review and apply with
`git apply PATH/formatting.patch`, then rerun. This also lets you use the
container's pre-commit tools without installing them on your host.
Clang-format differences appear in `clang-format.log`; stage the affected lines
and use `git clang-format --binary clang-format-19` to format them.

## Testing your backend

Choose a registered Meson test that exercises your backend:

```bash
./contrib/check.py --plugin YOUR_PLUGIN --test your_plugin_test
```

Repeat `--test` for additional tests. This replaces the default POSIX selection
and verifies that your backend is actually built, loads through NIXL, and its
selected tests pass. Unknown test names fail and list available tests. Tests
should return failure or an explicit skip when prerequisites are unavailable;
a test that internally skips cases but exits successfully cannot be detected.

For an SDK or hardware setup, use `--host` in your configured environment, or
`--image YOUR_IMAGE` with an image derived from `nixl-local-checks:ubuntu24`.
The tool versions and container recipe are in `contrib/check.py`. The default
container has no device passthrough or service credentials. Extra configuration
can be passed as `--meson-arg=-Ducx_path=/opt/ucx`; `--jobs N` limits parallelism.
The checker enforces shared plugins, tests enabled and warnings as errors.

Before requesting review, confirm:

- Your tests cover real transfers, supported capabilities, failure paths and
  cleanup. Add concurrency, remote metadata and notification coverage as relevant.
- Build/test registration and CI invocation are wired up. Document SDK versions,
  runtime services, hardware requirements and the test command in the backend README.
- Copyright, dependency attributions and redistribution expectations are resolved;
  explain any container, wheel, benchmark or hardware coverage still needed.
- Commits have DCO sign-offs (`git commit -s`), and the PR includes the result
  summary, tested configuration and remaining coverage. Maintainers trigger the
  broader Jenkins pipeline with `/build`.

Use [BackendGuide.md](BackendGuide.md) for the API contract and
[CONTRIBUTING.md](../CONTRIBUTING.md) for contribution standards.
