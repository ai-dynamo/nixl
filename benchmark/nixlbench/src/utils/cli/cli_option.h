/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_CLI_OPTION_H
#define NIXLBENCH_CLI_OPTION_H

#include <cstdint>
#include <string>
#include <variant>

#include "utils/cli/benchmark_requests.h"

namespace nixlbench {

enum class option_kind_t {
    VALUE,
    FLAG,
};

using option_target_t = std::variant<bool *, int *, uint64_t *, std::string *>;

struct cliOption {
    std::string name;
    std::string help;
    option_kind_t kind;
    option_target_t target;
    bool required = false;
    bool *provided = nullptr;

    static cliOption
    option(std::string name,
           std::string help,
           bool *target,
           bool required = false,
           bool *provided = nullptr);
    static cliOption
    option(std::string name,
           std::string help,
           int *target,
           bool required = false,
           bool *provided = nullptr);
    static cliOption
    option(std::string name,
           std::string help,
           uint64_t *target,
           bool required = false,
           bool *provided = nullptr);
    static cliOption
    option(std::string name,
           std::string help,
           std::string *target,
           bool required = false,
           bool *provided = nullptr);
    static cliOption
    option(std::string name,
           std::string help,
           providedValue<bool> *target,
           bool required = false);
    static cliOption
    option(std::string name,
           std::string help,
           providedValue<int> *target,
           bool required = false);
    static cliOption
    option(std::string name,
           std::string help,
           providedValue<uint64_t> *target,
           bool required = false);
    static cliOption
    option(std::string name,
           std::string help,
           providedValue<std::string> *target,
           bool required = false);
    static cliOption
    flag(std::string name, std::string help, bool *target, bool *provided = nullptr);
    static cliOption
    flag(std::string name, std::string help, providedValue<bool> *target);
};

} // namespace nixlbench

#endif // NIXLBENCH_CLI_OPTION_H
