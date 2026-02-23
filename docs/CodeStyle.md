<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NIXL Code style

## Naming Conventions

* **Lower camel case** (e.g., `myVariable`):
  * Type names - classes, structs, unions (e.g., `myClass`, `dataPacket`)
  * Template parameters (e.g., `template <typename dataType>`)
  * Class/struct members - public and protected data members (e.g., `myField`)
  * Functions - both member and non-member (e.g., `getValue()`, `processCompletions()`)
* **Snake case** (e.g., `my_variable`):
  * Variables - function arguments, local variables, global variables, and constants
    (e.g., `my_var`, `constexpr int default_port = 8080`)
  * Namespaces (e.g., `namespace nixl_utils`)
  * Type aliases with `_t` suffix (e.g., `using test_params_t = std::vector<int>`)
  * Enum class names with `_t` suffix (e.g., `enum class status_t`)
  * File names (e.g., `my_backend.h`, `data_processor.cpp`)
* **Upper snake case** (e.g., `MY_CONSTANT`):
  * Enum values (e.g., `SUCCESS`, `ERROR_TIMEOUT`)
  * Preprocessor macros (e.g., `#define MAX_BUFFER_SIZE 1024`)
  * Header guards (e.g., `#ifndef NIXL_BACKEND_H`)

## Class Design

### Member Declaration Order

* Class members should be declared in the following order to improve readability:
  1. **public** section first
  2. **protected** section second
  3. **private** section last

* Within each access level section, group declarations logically:
  1. Type definitions and nested classes
  2. Static member variables
  3. Constructors, assignment operators, and destructor
  4. Member functions
  5. Data members

### Private Member Naming

* Private class data members must use a trailing underscore suffix (e.g., `memberName_`)
* This convention clearly distinguishes private implementation details from public interface
* Example:

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

## File Organization

### File Headers

* All source files must begin with the SPDX license header
* Example:

  ```cpp
  /*
   * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   * SPDX-License-Identifier: Apache-2.0
   */
  ```

### Header Guards

* Use traditional `#ifndef`/`#define` header guards (not `#pragma once`)
* Header guard names should be upper snake case based on the file path
* Path mapping: Add a `NIXL_` project prefix, convert the entire path to upper snake case (replacing `/` and `.` with `_`), e.g., `src/utils/ucx/backend.h` → `NIXL_SRC_UTILS_UCX_BACKEND_H`
* Example:

  ```cpp
  #ifndef NIXL_SRC_UTILS_UCX_BACKEND_H
  #define NIXL_SRC_UTILS_UCX_BACKEND_H

  // ... header contents ...

  #endif  // NIXL_SRC_UTILS_UCX_BACKEND_H
  ```

## Formatting

### Line Length

* Maximum line length is **100 characters**
* Break long lines appropriately to stay within this limit

### Indentation

* Use **4 spaces** for indentation (no tabs)
* Continuation lines should be indented by 4 spaces
* Namespace content should be indented (inner namespace indentation)

### Function Declarations

* Return type should be on a separate line from the function name
* If the function signature exceeds 100 characters, parameters break to one per line
* Example:

  ```cpp
  nixl_status_t
  processCompletions(int timeout_ms); // Fits on one line

  void
  myFunction() {
      // Implementation
  }

  // When signature exceeds 100 chars, parameters go one per line
  void
  createConnectionWithAuthenticationAndRetry(
      const std::string &host,
      int port,
      int timeout_ms,
      bool use_ssl);
  ```

### Function Calls

* If the function call exceeds 100 characters, arguments break to one per line
* Example:

  ```cpp
  processData(value); // Fits on one line

  // When call exceeds 100 chars, arguments go one per line
  createConnectionWithAuthenticationAndRetry(
      "localhost",
      8080,
      5000,
      true);
  ```

### Braces

* Opening brace on the same line for most constructs (functions, if, else, loops, try, catch, etc.)
* The `catch` keyword goes on a new line (not on same line as `}` from try block)
* Example:

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

* Short if-statements without else can be on single line when appropriate
* Empty functions/blocks can be on single line: `void empty() {}`

### Switch Statements

* Case labels are **not indented** relative to the switch statement
* Avoid using `default` when switching on enum types - handle all cases explicitly to enable compiler warnings when new enum values are added
* Example:

  ```cpp
  // For enum class - avoid default to get compiler warnings for unhandled cases
  switch (status) {
  case status_t::SUCCESS: // case label aligned with switch
      handleSuccess();
      break;
  case status_t::ERROR:
      handleError();
      break;
  case status_t::PENDING:
      handlePending();
      break;
  // No default - compiler will warn if new enum values are added
  }

  // For non-enum types, default is acceptable
  switch (value) {
  case 1:
      handleOne();
      break;
  default:
      handleOther();
      break;
  }
  ```

### Parentheses

* **No space** before parentheses in function calls and declarations
* **Space required** before parentheses in control statements (`if`, `if constexpr`, `for`, `while`, `switch`, `catch`)
* Example:

  ```cpp
  void
  myFunction(int arg);    // Correct - no space before (

  myFunction(value);      // Correct - no space before (

  if (condition) {        // Correct - space before ( for control statement
      for (int i = 0; i < 10; i++) {
          processItem(i); // Correct - no space before (
      }
  }

  void function (arg);    // Incorrect - space before (
  if(condition) {         // Incorrect - missing space before (
  ```

### Pointer and Reference Alignment

* Pointers and references are **right-aligned** - the `*`, `&`, and `&&` are placed next to the variable name, not the type
* Example:

  ```cpp
  int *ptr;                // Correct - asterisk next to variable
  int &ref = value;        // Correct - ampersand next to variable
  int &&rref = getValue(); // Correct - double ampersand next to variable
  const char *str;         // Correct

  int* ptr;                // Avoid - asterisk next to type
  int & ref = value;       // Avoid - spaces around ampersand
  int&& rref = getValue(); // Avoid - no space before variable
  ```

### Constructor Initializers

* Constructor initializer lists break before the colon
* Example:

  ```cpp
  class myClass
      : public baseClass {
  public:
      myClass(int id, std::string name)
          : id_(id),
            name_(std::move(name)) {
          // Constructor body
      }

  private:
      int id_;
      std::string name_;
  };
  ```

## Comments

### Documentation Comments

* Use Doxygen-style block comments (`/** ... */`) for documenting public APIs, classes, functions, and types
* Include `@brief`, `@param`, `@return`, and other Doxygen tags as appropriate
* Example:

  ```cpp
  /**
   * @brief Process completions on active data rails
   * @param timeout Timeout duration
   * @return NIXL_SUCCESS if completions processed, error code on failure
   */
  [[nodiscard]] nixl_status_t
  processCompletions(std::chrono::milliseconds timeout);
  ```

### Inline Comments

* Use `///<` for trailing Doxygen documentation (enum values, struct/class members)
* Use `//` for regular code comments and explanatory notes (non-Doxygen)
* Example:

  ```cpp
  enum class status_t {
      SUCCESS,           ///< Operation completed successfully
      ERROR_TIMEOUT,     ///< Operation timed out
      ERROR_INVALID,     ///< Invalid parameter
  };

  // Initialize connection state (regular comment, not documentation)
  auto state = connection_state_t::DISCONNECTED;
  ```

## General Coding Practices

### Prefer Functions over Macros

* Prefer `constexpr` functions, `inline` functions, or templates over preprocessor macros
* Functions provide type safety, scoping, and debugging support that macros lack
* Use macros only when absolutely necessary (e.g., conditional compilation, stringification)
* Example:

  ```cpp
  // Preferred
  constexpr int
  square(int x) {
      return x * x;
  }

  template<typename T>
  T
  max(T a, T b) {
      return (a > b) ? a : b;
  }

  // Avoid
  #define SQUARE(x) ((x) * (x))
  #define MAX(a, b) ((a) > (b) ? (a) : (b))
  ```

### Anonymous Namespaces

* In implementation files (.cpp), prefer anonymous namespaces over `static` for file-local classes and functions
* Anonymous namespaces provide better type safety and clearer intent for internal linkage
* Do not use anonymous namespaces in header files - use `inline` functions or named namespaces instead
* Example:

  ```cpp
  // Preferred
  namespace {
      void
      helperFunction() {
          // Implementation
      }

      class internalHelper {
          // Implementation
      };
  }

  // Avoid
  static void
  helperFunction() {
      // Implementation
  }
  ```

### Type Deduction with `auto`

* Use `auto` for variable declarations when the type is obvious from the initializer or when dealing with verbose type names
* This improves readability and maintainability, especially with complex template types
* Example:

  ```cpp
  // Preferred
  auto iter = myContainer.begin();
  auto result = std::make_unique<complexType>(args);
  auto lambda = [](int x) { return x * 2; };

  // Avoid when type is verbose but intent is clear
  std::map<std::string, std::vector<std::shared_ptr<myWidget>>>::iterator iter = myContainer.begin();
  ```

### Override Specifier

* Always explicitly mark virtual methods that override base class methods with the `override` specifier
* This enables compile-time verification of the override relationship and prevents subtle bugs
* Example:

  ```cpp
  class derivedClass : public baseClass {
  public:
      void
      process() override; // Clearly indicates this overrides baseClass::process()

      int
      calculate(double x) const override; // Prevents typos in signature
  };
  ```
