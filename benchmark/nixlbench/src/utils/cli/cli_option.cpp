/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/cli_option.h"

namespace nixlbench {

cliOption
cliOption::option(std::string name,
                  std::string help,
                  bool *target,
                  bool required,
                  bool *provided) {
    return {std::move(name), std::move(help), option_kind_t::VALUE, target, required, provided};
}

cliOption
cliOption::option(std::string name,
                  std::string help,
                  int *target,
                  bool required,
                  bool *provided) {
    return {std::move(name), std::move(help), option_kind_t::VALUE, target, required, provided};
}

cliOption
cliOption::option(std::string name,
                  std::string help,
                  uint64_t *target,
                  bool required,
                  bool *provided) {
    return {std::move(name), std::move(help), option_kind_t::VALUE, target, required, provided};
}

cliOption
cliOption::option(std::string name,
                  std::string help,
                  std::string *target,
                  bool required,
                  bool *provided) {
    return {std::move(name), std::move(help), option_kind_t::VALUE, target, required, provided};
}

cliOption
cliOption::option(std::string name,
                  std::string help,
                  providedValue<bool> *target,
                  bool required) {
    return option(std::move(name), std::move(help), target->valuePtr(), required, target->providedPtr());
}

cliOption
cliOption::option(std::string name,
                  std::string help,
                  providedValue<int> *target,
                  bool required) {
    return option(std::move(name), std::move(help), target->valuePtr(), required, target->providedPtr());
}

cliOption
cliOption::option(std::string name,
                  std::string help,
                  providedValue<uint64_t> *target,
                  bool required) {
    return option(std::move(name), std::move(help), target->valuePtr(), required, target->providedPtr());
}

cliOption
cliOption::option(std::string name,
                  std::string help,
                  providedValue<std::string> *target,
                  bool required) {
    return option(std::move(name), std::move(help), target->valuePtr(), required, target->providedPtr());
}

cliOption
cliOption::flag(std::string name, std::string help, bool *target, bool *provided) {
    return {std::move(name), std::move(help), option_kind_t::FLAG, target, false, provided};
}

cliOption
cliOption::flag(std::string name, std::string help, providedValue<bool> *target) {
    return flag(std::move(name), std::move(help), target->valuePtr(), target->providedPtr());
}

} // namespace nixlbench
