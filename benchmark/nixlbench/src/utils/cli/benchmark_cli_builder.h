/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_CLI_BUILDER_H
#define NIXLBENCH_BENCHMARK_CLI_BUILDER_H

#include "utils/cli/benchmark_command.h"
#include "utils/cli/g3_scenario_command.h"
#include "utils/cli/g4_scenario_command.h"
#include "utils/cli/raw_command.h"

#include <memory>
#include <vector>

namespace nixlbench {

class BenchmarkCliBuilder {
public:
    BenchmarkCliBuilder();

    int
    parse(int argc, char **argv);

    int
    run();

    bool
    helpRequested() const;

    const IBenchmarkScenario *
    selectedScenario() const;

    const ISouthboundPluginBenchmarkCommand *
    selectedPlugin() const;

private:
    G3ScenarioCommand g3_;
    G4ScenarioCommand g4_;
    RawCommand raw_;

    std::vector<std::unique_ptr<ISouthboundPluginBenchmarkCommand>> g3_plugins_;
    std::vector<std::unique_ptr<ISouthboundPluginBenchmarkCommand>> g4_plugins_;
    std::vector<std::unique_ptr<ISouthboundPluginBenchmarkCommand>> raw_plugins_;

    IBenchmarkScenario *selected_scenario_ = nullptr;
    ISouthboundPluginBenchmarkCommand *selected_plugin_ = nullptr;
    bool help_ = false;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_CLI_BUILDER_H
