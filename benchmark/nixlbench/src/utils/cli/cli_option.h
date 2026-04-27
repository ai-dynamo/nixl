/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_CLI_OPTION_H
#define NIXLBENCH_CLI_OPTION_H

#include <cstdint>
#include <string>
#include <variant>

namespace nixlbench {

enum class OptionKind {
    Value,
    Flag,
};

using OptionTarget = std::variant<bool *, int *, uint64_t *, std::string *>;

struct CliOption {
    std::string name;
    std::string help;
    OptionKind kind;
    OptionTarget target;
    bool required = false;

    static CliOption
    option(std::string name, std::string help, bool *target, bool required = false);
    static CliOption
    option(std::string name, std::string help, int *target, bool required = false);
    static CliOption
    option(std::string name, std::string help, uint64_t *target, bool required = false);
    static CliOption
    option(std::string name, std::string help, std::string *target, bool required = false);
    static CliOption
    flag(std::string name, std::string help, bool *target);
};

} // namespace nixlbench

#endif // NIXLBENCH_CLI_OPTION_H
