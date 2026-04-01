---
title: Contribution Guide
description: How to contribute to the NIXL project -- development setup, code style guidelines, testing requirements, and the contribution process.
---

This guide covers the development setup, code style, testing requirements, and contribution process for NIXL. NIXL is a C++17 project with strict standards for code quality and testing.

## Getting Started

Before contributing, please:

1. Review existing issues and PRs to avoid duplicate work
2. For significant changes, open an issue for discussion before implementation
3. Familiarize yourself with our code style and project structure
4. Set up your development environment according to the guidelines below

<Warning>
All contributions require signing off on the Developer Certificate of Origin (DCO). Each commit must include a `Signed-off-by` line with your real name and email. See the [DCO section](#developer-certificate-of-origin) below.
</Warning>

## Development Setup

### Cloning the Repository

```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
```

### Building with Meson

NIXL uses Meson and Ninja for building:

```bash
# Configure the build
meson setup build

# Build the project
meson compile -C build

# Run tests
meson test -C build
```

<Tip>
See the [Quick Start](./quick-start) for PyPI installation or [Building NIXL from Source](../user-guide/building-nixl) for source build options including Meson flags and Docker containers.
</Tip>

### Setting Up clang-format

All new C++ code must be formatted using the provided `.clang-format` configuration. Code formatting is automatically checked in CI, and improperly formatted code will be rejected:

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

### Building Docs Locally

To validate documentation changes:

```bash
fern check
```

### Required Tools

- C++17 compatible compiler
- Meson build system
- Ninja build tool
- clang-format
- Python (for build scripts and testing)
- Git with DCO sign-off configured

## Code Standards

### C++17 Guidelines

NIXL follows the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) where appropriate:

1. **Use modern C++ features**: Prefer `auto`, range-based loops, structured bindings, `std::optional`, etc.
2. **RAII everywhere**: Resource management through constructors/destructors
3. **Smart pointers for ownership**: Use `std::unique_ptr`, `std::shared_ptr` appropriately
4. **Prefer `const` correctness**: Mark methods and variables `const` when appropriate
5. **Exception handling**: Exceptions are recommended for control-path code, while error codes should be used for data-path

### STL and Abseil Usage

1. **Prefer STL types**: Use rich STL types as the primary choice
   - Standard containers: `std::vector`, `std::unordered_map`, etc.
   - Modern utilities: `std::optional`, `std::variant`, `std::string_view`
   - Algorithms from `<algorithm>` and `<numeric>`

2. **Fallback to Abseil**: When STL lacks required functionality
   - String formatting: `absl::StrFormat`
   - Error handling: `absl::StatusOr` for data-path operations that return values with potential errors
   - High-performance containers: `absl::flat_hash_map` when needed
   - Logging utilities (integrated with NIXL logging)

3. **Never expose Abseil in public APIs**
   - Keep Abseil types internal to implementation files
   - Plug-in and agent public APIs must only use STL types
   - Convert between Abseil and STL types at API boundaries

<Warning>
Exposing Abseil types in public APIs will cause your PR to be rejected. Always use STL types at API boundaries.
</Warning>

### Error Handling

1. **Control-path code**: Use exceptions for exceptional conditions
   - `std::runtime_error` for runtime failures
   - `std::invalid_argument` for invalid parameters
   - Custom exceptions when appropriate

2. **Data-path code**: Use error codes for performance-critical paths
   - Return `nixl_status_t` or similar error codes
   - Avoid exceptions in hot paths

3. **Logging**: Use NIXL logging macros
   - `NIXL_ERROR`: Critical errors that prevent normal operation (system failures, resource exhaustion, unrecoverable errors)
   - `NIXL_WARN`: Warning conditions that don't prevent operation (deprecated API usage, performance degradation, recoverable errors, fallback behavior)
   - `NIXL_INFO`: Informational messages about normal operation (initialization complete, configuration loaded, major state changes)
   - `NIXL_DEBUG`: Detailed debugging information for development (function entry/exit, intermediate values, algorithm decisions)
   - `NIXL_TRACE`: Very detailed trace information for deep debugging (per-packet processing, memory allocations, lock acquisitions)

## Code Style

All code must adhere to these style guidelines and be formatted with `.clang-format`.

### Naming Conventions

- **Lower camelCase** (e.g., `myVariable`):
  - Type names -- classes, structs, unions (e.g., `myClass`, `dataPacket`)
  - Template parameters (e.g., `template <typename dataType>`)
  - Class/struct members -- public and protected data members (e.g., `myField`)
  - Functions -- both member and non-member (e.g., `getValue()`, `processCompletions()`)

- **snake_case** (e.g., `my_variable`):
  - Variables -- function arguments, local variables, global variables, and constants (e.g., `my_var`, `constexpr int default_port = 8080`)
  - Namespaces (e.g., `namespace nixl_utils`)
  - Type aliases with `_t` suffix (e.g., `using test_params_t = std::vector<int>`)
  - Enum class names with `_t` suffix (e.g., `enum class status_t`)
  - File names (e.g., `my_backend.h`, `data_processor.cpp`)

- **UPPER_SNAKE_CASE** (e.g., `MY_CONSTANT`):
  - Enum values (e.g., `SUCCESS`, `ERROR_TIMEOUT`)
  - Preprocessor macros (e.g., `#define MAX_BUFFER_SIZE 1024`)
  - Header guards (e.g., `#ifndef NIXL_BACKEND_H`)

### Class Design

**Member Declaration Order:**

Class members should be declared in this order:
1. **public** section first
2. **protected** section second
3. **private** section last

Within each access level, group declarations logically:
1. Type definitions and nested classes
2. Static member variables
3. Constructors, assignment operators, and destructor
4. Member functions
5. Data members

**Private Member Naming:**

Private class data members must use a trailing underscore suffix (e.g., `memberName_`). This clearly distinguishes private implementation details from the public interface:

```cpp
class plugin {
public:
    explicit plugin(int id);

    [[nodiscard]] int
    getId() const;

private:
    int id_;
    std::string name_;
};
```

### File Organization

**File Headers:**

All source files must begin with the SPDX license header:

```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
```

**Header Guards:**

Use traditional `#ifndef`/`#define` header guards (not `#pragma once`). Header guard names should be upper snake case based on the file path. Add a `NIXL_` project prefix, convert the entire path to upper snake case (replacing `/` and `.` with `_`):

```cpp
// src/utils/ucx/backend.h -> NIXL_SRC_UTILS_UCX_BACKEND_H
#ifndef NIXL_SRC_UTILS_UCX_BACKEND_H
#define NIXL_SRC_UTILS_UCX_BACKEND_H

// ... header contents ...

#endif  // NIXL_SRC_UTILS_UCX_BACKEND_H
```

### Formatting

**Line Length:** Maximum line length is **100 characters**. Break long lines appropriately.

**Indentation:** Use **4 spaces** for indentation (no tabs). Continuation lines should be indented by 4 spaces. Namespace content should be indented.

**Function Declarations:** Return type on a separate line from the function name. If the signature exceeds 100 characters, parameters break to one per line:

```cpp
nixl_status_t
processCompletions(int timeout_ms); // Fits on one line

// When signature exceeds 100 chars, parameters go one per line
void
createConnectionWithAuthenticationAndRetry(
    const std::string &host,
    int port,
    int timeout_ms,
    bool use_ssl);
```

**Function Calls:** If the function call exceeds 100 characters, arguments break to one per line:

```cpp
processData(value); // Fits on one line

// When call exceeds 100 chars, arguments go one per line
createConnectionWithAuthenticationAndRetry(
    "localhost",
    8080,
    5000,
    true);
```

**Braces:** Opening brace on the same line for most constructs (functions, if, else, loops, try, catch). The `catch` keyword goes on a new line:

```cpp
void
myFunction() {
    if (condition) {
        // code
    } else {
        // code
    }

    try {
        // code
    }
    catch (const std::exception &e) {
        // catch on new line, brace on same line as catch
    }
}
```

Short if-statements without else can be on a single line when appropriate. Empty functions/blocks can be on a single line: `void empty() {}`

**Switch Statements:** Case labels are **not indented** relative to the switch statement. Avoid `default` when switching on enum types to enable compiler warnings for unhandled cases:

```cpp
// For enum class - avoid default to get compiler warnings
switch (status) {
case status_t::SUCCESS:
    handleSuccess();
    break;
case status_t::ERROR:
    handleError();
    break;
// No default - compiler will warn if new enum values are added
}
```

**Parentheses:** No space before parentheses in function calls and declarations. Space required before parentheses in control statements (`if`, `for`, `while`, `switch`, `catch`):

```cpp
myFunction(value);      // Correct - no space before (
if (condition) {        // Correct - space before ( for control statement
```

**Pointer and Reference Alignment:** Pointers and references are **right-aligned** -- the `*`, `&`, and `&&` are placed next to the variable name, not the type:

```cpp
int *ptr;                // Correct - asterisk next to variable
int &ref = value;        // Correct - ampersand next to variable
const char *str;         // Correct
```

**Constructor Initializers:** Constructor initializer lists break before the colon:

```cpp
class myClass
    : public baseClass {
public:
    myClass(int id, std::string name)
        : id_(id),
          name_(std::move(name)) {
        // Constructor body
    }
};
```

### Comments

**Documentation Comments:** Use Doxygen-style block comments (`/** ... */`) for documenting public APIs, classes, functions, and types. Include `@brief`, `@param`, `@return`, and other Doxygen tags:

```cpp
/**
 * @brief Process completions on active data rails
 * @param timeout Timeout duration
 * @return NIXL_SUCCESS if completions processed, error code on failure
 */
[[nodiscard]] nixl_status_t
processCompletions(std::chrono::milliseconds timeout);
```

**Inline Comments:** Use `///<` for trailing Doxygen documentation (enum values, struct/class members). Use `//` for regular code comments:

```cpp
enum class status_t {
    SUCCESS,           ///< Operation completed successfully
    ERROR_TIMEOUT,     ///< Operation timed out
};

// Initialize connection state (regular comment)
auto state = connection_state_t::DISCONNECTED;
```

### General Coding Practices

**Prefer Functions over Macros:** Prefer `constexpr` functions, `inline` functions, or templates over preprocessor macros. Functions provide type safety, scoping, and debugging support:

```cpp
// Preferred
constexpr int
square(int x) {
    return x * x;
}

// Avoid
#define SQUARE(x) ((x) * (x))
```

**Anonymous Namespaces:** In implementation files (.cpp), prefer anonymous namespaces over `static` for file-local classes and functions. Do not use anonymous namespaces in header files:

```cpp
// Preferred in .cpp files
namespace {
    void
    helperFunction() {
        // Implementation
    }
}
```

**Type Deduction with `auto`:** Use `auto` for variable declarations when the type is obvious from the initializer or when dealing with verbose type names:

```cpp
auto iter = myContainer.begin();
auto result = std::make_unique<complexType>(args);
auto lambda = [](int x) { return x * 2; };
```

**Override Specifier:** Always explicitly mark virtual methods that override base class methods with the `override` specifier:

```cpp
class derivedClass : public baseClass {
public:
    void
    process() override;

    int
    calculate(double x) const override;
};
```

## Contributing Process

Contributions that fix documentation errors or make small changes to existing code can be contributed directly by following the rules below and submitting a PR.

For significant new functionality, open a GitHub issue first to discuss the design with the NIXL team.

- Agree on a design through the issue discussion before starting implementation.
- Include comprehensive tests. The NIXL team helps design tests compatible with existing infrastructure.
- User-visible features require documentation.

<Note>
Contributions to the code under `./examples/device/ep` (derived from DeepEP, licensed under MIT) must be licensed under Apache 2.0.
</Note>

### Review Process Expectations

Review standards:

**Timeline and Iterations:**
- Initial review typically takes 1-2 weeks depending on PR complexity
- Most PRs require 2-4 rounds of review before merging
- Complex features may take longer as we ensure architectural consistency

**How We Support Contributors:**
- Reviewers provide detailed feedback to improve contributions
- Reviewers collaborate, not just critique. Ask questions if feedback is unclear.
- For significant changes, we may suggest incremental PRs for easier review
- The team helps ensure contributions align with NIXL's architecture

**Tips for Smoother Reviews:**
- Start with smaller PRs to familiarize yourself with our standards
- Engage early through issues for design discussions
- Be responsive to feedback and ask for clarification when needed
- Consider breaking large changes into logical, reviewable chunks

### Commit Messages

```text
component: Brief description of change

Longer explanation of the change, its motivation, and impact.

Fixes #123
Signed-off-by: Your Name <your.email@example.com>
```

## Plugin Development

New plug-in contributions follow this structure:

### Plugin Structure

Plug-ins are located in `src/plugins/`. Your plug-in should follow this structure:

```text
src/plugins/your_plugin/
+-- meson.build
+-- your_plugin.cpp
+-- your_plugin.h
+-- your_backend.cpp
+-- your_backend.h
+-- README.md
```

Tests should be added in the GoogleTest-based test directory:

```text
test/gtest/unit/plugins/your_plugin/
+-- test_your_plugin.cpp
```

### Build System Integration

Create a `meson.build` file for your plug-in. If your plug-in requires external dependencies:

```meson
your_dep = dependency('your-dependency', required: false)
if not your_dep.found()
    subdir_done()
endif

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

### Container Build Extension

If your plug-in requires system dependencies, update `contrib/Dockerfile`. See existing examples for compiling dependencies from source.

### Plugin Documentation

Your plug-in's README.md must include:
- **Overview**: Basic functionality description
- **Dependencies**: List all external requirements
- **Build Instructions**: How to build with/without the plug-in
- **API Reference**: Key classes and functions
- **Example Usage**: Simple, working example

<Tip>
See [Building a Backend Plugin](../development/building-a-backend-plugin) for the full step-by-step tutorial on implementing the SB API.
</Tip>

## Testing Requirements

### Test Framework

- New tests should use GoogleTest framework in `test/gtest/`
- Legacy tests may exist in other locations
- Run tests with: `meson test -C build`

### Test Coverage

- New features must include comprehensive tests
- Fixes must include regression tests
- Test both success and error paths

### Test Organization

```cpp
TEST(YourPlugin, HandlesValidInput) {
    // Test implementation
}

TEST(YourPlugin, HandlesInvalidInput) {
    // Test error handling
}
```

## Documentation Standards

### Code Documentation

Document public APIs and complex implementations using Doxygen-style comments:

```cpp
/**
 * Brief description of the function
 *
 * Detailed explanation if needed
 *
 * @param param1 Description of parameter
 * @return Description of return value
 */
```

### PR Documentation

Use the provided template in `.github/pull_request_template.md`:

1. **What?**: Clear description of changes
2. **Why?**: Justification and issue references
3. **How?**: Technical approach for complex changes

## Pull Request Guidelines

### Before Submitting

- Code follows style guidelines (`.clang-format` applied)
- Follows conventions in the Code Style section above
- All tests pass
- New tests added for new functionality
- Documentation updated where needed
- PR template filled out completely
- Commits are signed with DCO

## Miscellaneous

- NIXL's default build assumes recent versions of dependencies (CUDA, PyTorch, etc.). Contributions that add compatibility with older versions will be considered, but NVIDIA cannot guarantee all possible build configurations work or retain highest performance.
- Make sure you can contribute your work to open source (no license or patent conflict introduced by your code). You must certify compliance with the license terms and sign off on the DCO before your PR can be merged.

## Developer Certificate of Origin

NIXL is an open source product released under the Apache 2.0 license. The Apache 2.0 license allows you to freely use, modify, distribute, and sell your own products that include Apache 2.0 licensed software.

We respect intellectual property rights and want to make sure all incoming contributions are correctly attributed and licensed. A Developer Certificate of Origin (DCO) is a lightweight mechanism to do that.

The DCO is a declaration attached to every contribution made by every developer. In the commit message of the contribution, the developer adds a `Signed-off-by` statement and thereby agrees to the DCO, which you can find at [DeveloperCertificate.org](http://developercertificate.org/).

We require that every contribution to NIXL is signed with a Developer Certificate of Origin. Additionally, please use your real name. We do not accept anonymous contributors nor those utilizing pseudonyms.

Each commit must include a DCO which looks like this:

```text
Signed-off-by: Jane Smith <jane.smith@email.com>
```

You may type this line on your own when writing your commit messages. However, if your `user.name` and `user.email` are set in your git configs, you can use `-s` or `--signoff` to add the `Signed-off-by` line to the end of the commit message.
