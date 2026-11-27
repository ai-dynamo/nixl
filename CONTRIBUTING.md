<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Contribution Guidelines

Welcome to NIXL! This document provides guidelines for contributing to our modern C++20 project. Please read through these guidelines carefully before submitting your contribution.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Code Standards](#code-standards)
4. [Contributing Process](#contributing-process)
   - [Review Process Expectations](#review-process-expectations)
5. [Plugin Development](#plugin-development)
6. [Testing Requirements](#testing-requirements)
7. [Documentation Standards](#documentation-standards)
8. [Pull Request Guidelines](#pull-request-guidelines)
9. [Miscelaneous](#miscelaneous)
10. [Developer Certificate of Origin](#developer-certificate-of-origin)

## Getting Started

Before contributing, please:

1. Review existing issues and PRs to avoid duplicate work
2. For significant changes, open an issue for discussion before implementation
3. Familiarize yourself with our code style and project structure
4. Set up your development environment according to our guidelines

## Development Environment

### Required Tools

- C++20 compatible compiler (GCC >= 11, Clang >= 14)
- Meson build system
- Ninja build tool
- clang-format
- Python (for build scripts and testing)
- Git with DCO sign-off configured

### Building the Project

NIXL uses Meson and Ninja for building:

```bash
# Configure the build
meson setup build

# Build the project
ninja -C build

# Run tests
ninja -C build test
```

### Setting Up clang-format

All new C++ code must be formatted using the provided `.clang-format` configuration. Code formatting will be automatically checked for conformance in CI, and improperly formatted code will be rejected:

```bash
# Format only changed lines in staged files (recommended)
git clang-format

# Format only changed lines between commits
git clang-format HEAD~1

# Format only changed lines in specific files
git clang-format --diff path/to/file.cpp

# Alternative: Use clang-format-diff to format only changed lines
git diff -U0 --no-color HEAD^ | clang-format-diff -p1 -i

# Or format only unstaged changes
git diff -U0 --no-color | clang-format-diff -p1 -i
```

### Pre-commit Hooks

The project uses pre-commit hooks for Python code quality. Install them with:

```bash
pip install pre-commit
pre-commit install
```

## Code Standards

Format C++ code with `.clang-format`, and follow [docs/CodeStyle.md](docs/CodeStyle.md).

## Contributing Process

Contributions to the code under `./examples/device/ep` and `./runtime/ep`
(which are derived from DeepEP and licensed under the MIT License) must be
licensed under Apache 2.0.

Contributions that fix documentation errors or that make small changes
to existing code can be contributed directly by following the rules
below and submitting an appropriate PR.

Contributions intended to add significant new functionality must
follow a more collaborative path described in the following
points. Before submitting a large PR that adds a major enhancement or
extension, be sure to submit a GitHub issue that describes the
proposed change so that the NIXL team can provide feedback.

- As part of the GitHub issue discussion, a design for your change
  will be agreed upon. An up-front design discussion is required to
  ensure that your enhancement is done in a manner that is consistent
  with NIXL's overall architecture.

- Testing is a critical part of any NIXL
  enhancement. You should plan on spending significant time on
  creating tests for your change. The NIXL team will help you to
  design your testing so that it is compatible with existing testing
  infrastructure.

- If your enhancement provides a user visible feature then you need to
  provide documentation.

### Review Process Expectations

We greatly appreciate all contributions to NIXL! To maintain the high quality, performance, and security standards that our users depend on, we have a thorough review process. Here's what to expect:

#### Timeline and Iterations

- Initial review typically takes 1-2 weeks depending on PR complexity and reviewer availability
- Most PRs require 2-4 rounds of review before merging
- Complex features may take longer as we ensure architectural consistency

#### Why Our Review Process is Thorough

- NIXL is used in performance-critical environments where reliability is paramount
- We maintain backward compatibility and stable APIs for our users
- Security and correctness are non-negotiable in data movement operations
- We aim for code that is maintainable by the community long-term

#### How We Support Contributors

- Reviewers provide detailed feedback to help improve the contribution
- We're here to collaborate, not just critique - ask questions if feedback is unclear
- For significant changes, we may suggest incremental PRs to make review easier
- The team will help ensure your contribution aligns with NIXL's architecture

#### Tips for Smoother Reviews

- Start with smaller PRs to familiarize yourself with our standards
- Engage early through issues for design discussions
- Be responsive to feedback and ask for clarification when needed
- Consider breaking large changes into logical, reviewable chunks

We recognize that our standards are high, but this ensures NIXL remains a reliable foundation for critical workloads. Your patience and collaboration through the review process is genuinely appreciated, and we're committed to helping your contribution succeed.

## Plugin Development

### Creating New Plugins

When contributing new plugins, follow these guidelines:

#### 1. Plugin Structure

Plugins are located in `src/plugins/`. Your plugin should follow this structure:

```text
src/plugins/your_plugin/
├── meson.build
├── your_plugin.cpp
├── your_plugin.h
├── your_backend.cpp
├── your_backend.h
└── README.md
```

Tests should be added in the GoogleTest-based test directory:

```text
test/gtest/unit/your_plugin/
└── test_your_plugin.cpp
```

#### 2. Build System Integration

Add the plugin name to `all_plugins` in the top-level `meson.build`, then add an entry for the plugin to `src/plugins/meson.build`. If your plugin requires external dependencies, check them there: fail when the user explicitly requested the plugin, and skip the plugin otherwise:

```meson
your_dep = dependency('your-dependency', required: false)
if enabled_plugins.get('YOUR_PLUGIN')
    if not your_dep.found() and is_explicit_enable
        error('YOUR_PLUGIN plugin requested but your-dependency not found')
    elif your_dep.found()
        subdir('your_plugin')
    endif
endif
```

Create a `meson.build` file in your plugin directory that builds the plugin:

```meson
your_plugin_lib = shared_library(
    'YOUR_PLUGIN',
    your_sources,
    dependencies: plugin_deps + [your_dep],
    cpp_args: compile_defs + ['-fPIC'],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    install: true,
    name_prefix: 'libplugin_',
    install_dir: plugin_install_dir)
```

#### 3. Container Build Extension

If your plugin requires system dependencies, update `contrib/Dockerfile`. See existing examples for compiling dependencies from source:

```dockerfile
# Example: Building a dependency from source
WORKDIR /workspace
RUN git clone https://github.com/example/your-dependency.git && \
    cd your-dependency && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# Or for autotools-based projects:
RUN cd /usr/local/src && \
    git clone https://github.com/example/your-lib.git && \
    cd your-lib && \
    ./autogen.sh && \
    ./configure --enable-feature && \
    make -j && \
    make install && \
    ldconfig
```

#### 4. Plugin Documentation

Every new backend plugin must include a `README.md` following the required sections in [docs/BackendGuide.md](docs/BackendGuide.md#plugin-documentation).

### Plugin Testing

1. **Unit tests**: Test individual components in `test/gtest/unit/your_plugin/`
2. **Integration tests**: Test plugin with NIXL framework
3. **Example validation**: Ensure examples compile and run

## Testing Requirements

### Test Framework

- New tests should use GoogleTest framework in `test/gtest/`
- Legacy tests may exist in other locations
- Run tests with: `ninja -C build test`

### Test Coverage

- New features must include comprehensive tests
- Bug fixes must include regression tests
- Test both success and error paths

### Test Organization

```cpp
// Example unit test
TEST(YourPlugin, HandlesValidInput) {
    // Test implementation
}

TEST(YourPlugin, HandlesInvalidInput) {
    // Test error handling
}
```

## Documentation Standards

### Code Documentation

Document code as [docs/CodeStyle.md](docs/CodeStyle.md) describes.

### PR Documentation

Use the provided template in `.github/pull_request_template.md`:

1. **What?**: Clear description of changes
2. **Why?**: Justification and issue references
3. **How?**: Technical approach for complex changes

## Pull Request Guidelines

Please review the [Review Process Expectations](#review-process-expectations) section to understand our thorough review process and timeline.

### Before Submitting

- [ ] Code follows style guidelines (`.clang-format` applied)
- [ ] Follows conventions in `docs/CodeStyle.md`
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated where needed
- [ ] PR template filled out completely
- [ ] Commits are signed with DCO

### Commit Messages

```text
component: Brief description of change

Longer explanation of the change, its motivation, and impact.

Fixes #123
Signed-off-by: Your Name <your.email@example.com>
```

### Agent Skills

Pull requests from a fork that change `skills/` need NVSkills signing:

1. Grant `svc-nvskills-signing` write access to your fork and make sure the invitation is accepted.
2. Ask a NIXL maintainer to comment `/nvskills-ci` after each commit that changes `skills/`.

See the [NVIDIA/skills contributing guide](https://github.com/NVIDIA/skills/blob/main/CONTRIBUTING.md) for details.

## Miscelaneous

- NIXL's default build assumes recent versions of
  dependencies (CUDA, PyTorch, etc.).
  Contributions that add compatibility with older versions of
  those dependencies will be considered, but NVIDIA cannot guarantee
  that all possible build configurations work, are not broken by
  future contributions, and retain highest performance.

- Make sure that you can contribute your work to open source (no
  license and/or patent conflict is introduced by your code).
  You must certify compliance with the
  [license terms](https://github.com/dynemo-ai/nixl/blob/main/LICENSE)
  and sign off on the [Developer Certificate of Origin (DCO)](https://developercertificate.org)
  described below before your pull request (PR) can be merged.

- Thanks in advance for your patience as we review your contributions;
  we do appreciate them!

## Developer Certificate of Origin

NIXL is an open source product released under
the Apache 2.0 license (see either
[the Apache site](https://www.apache.org/licenses/LICENSE-2.0) or
the [LICENSE file](./LICENSE)). The Apache 2.0 license allows you
to freely use, modify, distribute, and sell your own products
that include Apache 2.0 licensed software.

We respect intellectual property rights of others and we want
to make sure all incoming contributions are correctly attributed
and licensed. A Developer Certificate of Origin (DCO) is a
lightweight mechanism to do that.

The DCO is a declaration attached to every contribution made by
every developer. In the commit message of the contribution,
the developer simply adds a `Signed-off-by` statement and thereby
agrees to the DCO, which you can find below or at [DeveloperCertificate.org](http://developercertificate.org/).

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

We require that every contribution to NIXL is signed with
a Developer Certificate of Origin. Additionally, please use your real name.
We do not accept anonymous contributors nor those utilizing pseudonyms.

Each commit must include a DCO which looks like this

```text
Signed-off-by: Jane Smith <jane.smith@email.com>
```

You may type this line on your own when writing your commit messages.
However, if your user.name and user.email are set in your git configs,
you can use `-s` or `--signoff` to add the `Signed-off-by` line to
the end of the commit message.
