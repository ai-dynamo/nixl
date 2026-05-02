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

class southboundPluginBenchmarkCommand;

class benchmarkCommand {
public:
    virtual ~benchmarkCommand() = default;

    virtual std::string_view
    name() const = 0;

    virtual std::string_view
    description() const = 0;

    virtual const std::vector<cliOption> &
    getOptions() const = 0;
};

class benchmarkScenario : public benchmarkCommand {
public:
    virtual scenario_type_t
    scenarioType() const = 0;

    virtual bool
    supportsPlugin(plugin_type_t plugin) const = 0;

    virtual int
    run(southboundPluginBenchmarkCommand &plugin) = 0;
};

class southboundPluginBenchmarkCommand : public benchmarkCommand {
public:
    virtual plugin_type_t
    pluginType() const = 0;

    virtual bool
    supportsScenario(scenario_type_t scenario) const = 0;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_COMMAND_H
