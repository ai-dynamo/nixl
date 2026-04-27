/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/cli_option.h"

namespace nixlbench {

CliOption
CliOption::option(std::string name, std::string help, bool *target, bool required) {
    return {std::move(name), std::move(help), OptionKind::Value, target, required};
}

CliOption
CliOption::option(std::string name, std::string help, int *target, bool required) {
    return {std::move(name), std::move(help), OptionKind::Value, target, required};
}

CliOption
CliOption::option(std::string name, std::string help, uint64_t *target, bool required) {
    return {std::move(name), std::move(help), OptionKind::Value, target, required};
}

CliOption
CliOption::option(std::string name, std::string help, std::string *target, bool required) {
    return {std::move(name), std::move(help), OptionKind::Value, target, required};
}

CliOption
CliOption::flag(std::string name, std::string help, bool *target) {
    return {std::move(name), std::move(help), OptionKind::Flag, target, false};
}

} // namespace nixlbench
