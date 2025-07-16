<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

* **Lower camel case** (e.g., `myVariable`): compound data types (class, struct, union), class members, member functions
* **Snake case** (e.g., `my_variable`): function arguments, local variables, enums and aliases (with _t suffix)

## Class Design

### Member Declaration Order

* Class members should be declared in the following order to improve readability:
  1. **public** section first
  2. **protected** section second
  3. **private** section last

* Within each access level section, group declarations logically:
  1. Type definitions and nested classes
  2. Static member variables
  3. Constructors and destructor
  4. Member functions
  5. Data members

### Private Member Naming

* Private class data members must use a trailing underscore suffix (e.g., `memberName_`)
* This convention clearly distinguishes private implementation details from public interface
* Example:

  ```cpp
  class Plugin {
  public:
      Plugin(int id);
      int getId() const;

  private:
      int id_;
      std::string name_;
  };
  ```

## General Coding Practices

### Anonymous Namespaces

* Prefer anonymous namespaces over `static` for file-local classes and functions
* Anonymous namespaces provide better type safety and clearer intent for internal linkage
* Example:

  ```cpp
  // Preferred
  namespace {
      void helperFunction() {
          // Implementation
      }

      class InternalHelper {
          // Implementation
      };
  }

  // Avoid
  static void helperFunction() {
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
  auto result = std::make_unique<ComplexType>(args);
  auto lambda = [](int x) { return x * 2; };

  // Avoid when type is verbose but intent is clear
  std::map<std::string, std::vector<std::shared_ptr<Widget>>>::iterator iter = myContainer.begin();
  ```

### Override Specifier

* Always explicitly mark virtual methods that override base class methods with the `override` specifier
* This enables compile-time verification of the override relationship and prevents subtle bugs
* Example:

  ```cpp
  class Derived : public Base {
  public:
      void process() override;  // Clearly indicates this overrides Base::process()
      int calculate(double x) const override;  // Prevents typos in signature
  };
  ```
