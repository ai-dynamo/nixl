/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_COMMAND_H
#define NIXLBENCH_BENCHMARK_COMMAND_H

#include "utils/cli/benchmark_requests.h"
#include "utils/cli/cli_option.h"

#include <string_view>
#include <vector>

namespace nixlbench {

class IBenchmarkCommand {
public:
    virtual ~IBenchmarkCommand() = default;

    virtual std::string_view
    name() const = 0;

    virtual std::string_view
    description() const = 0;

    virtual const std::vector<CliOption> &
    getOptions() const = 0;
};

class IBenchmarkScenario : public IBenchmarkCommand {
public:
    virtual ScenarioType
    scenarioType() const = 0;
};

class ISouthboundPluginBenchmarkCommand : public IBenchmarkCommand {
public:
    virtual PluginType
    pluginType() const = 0;

    virtual bool
    supportsScenario(ScenarioType scenario) const = 0;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_COMMAND_H
