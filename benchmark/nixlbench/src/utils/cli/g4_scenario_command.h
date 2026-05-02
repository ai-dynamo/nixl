/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_G4_SCENARIO_COMMAND_H
#define NIXLBENCH_G4_SCENARIO_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class g4ScenarioCommand : public benchmarkScenario {
public:
    g4ScenarioCommand();

    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<cliOption> &getOptions() const override;
    scenario_type_t scenarioType() const override;
    bool supportsPlugin(plugin_type_t plugin) const override;
    int run(southboundPluginBenchmarkCommand &plugin) override;
    const g4ScenarioRequest &request() const;

private:
    g4ScenarioRequest request_;
    std::vector<cliOption> options_;
};

} // namespace nixlbench

#endif // NIXLBENCH_G4_SCENARIO_COMMAND_H
