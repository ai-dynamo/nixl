/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/cli_option.h"

namespace nixlbench {

CliOption
CliOption::option(std::string name,
                  std::string help,
                  bool *target,
                  bool required,
                  bool *provided) {
    return {std::move(name), std::move(help), OptionKind::Value, target, required, provided};
}

CliOption
CliOption::option(std::string name,
                  std::string help,
                  int *target,
                  bool required,
                  bool *provided) {
    return {std::move(name), std::move(help), OptionKind::Value, target, required, provided};
}

CliOption
CliOption::option(std::string name,
                  std::string help,
                  uint64_t *target,
                  bool required,
                  bool *provided) {
    return {std::move(name), std::move(help), OptionKind::Value, target, required, provided};
}

CliOption
CliOption::option(std::string name,
                  std::string help,
                  std::string *target,
                  bool required,
                  bool *provided) {
    return {std::move(name), std::move(help), OptionKind::Value, target, required, provided};
}

CliOption
CliOption::option(std::string name,
                  std::string help,
                  Provided<bool> *target,
                  bool required) {
    return option(std::move(name), std::move(help), target->valuePtr(), required, target->providedPtr());
}

CliOption
CliOption::option(std::string name,
                  std::string help,
                  Provided<int> *target,
                  bool required) {
    return option(std::move(name), std::move(help), target->valuePtr(), required, target->providedPtr());
}

CliOption
CliOption::option(std::string name,
                  std::string help,
                  Provided<uint64_t> *target,
                  bool required) {
    return option(std::move(name), std::move(help), target->valuePtr(), required, target->providedPtr());
}

CliOption
CliOption::option(std::string name,
                  std::string help,
                  Provided<std::string> *target,
                  bool required) {
    return option(std::move(name), std::move(help), target->valuePtr(), required, target->providedPtr());
}

CliOption
CliOption::flag(std::string name, std::string help, bool *target, bool *provided) {
    return {std::move(name), std::move(help), OptionKind::Flag, target, false, provided};
}

CliOption
CliOption::flag(std::string name, std::string help, Provided<bool> *target) {
    return flag(std::move(name), std::move(help), target->valuePtr(), target->providedPtr());
}

} // namespace nixlbench
