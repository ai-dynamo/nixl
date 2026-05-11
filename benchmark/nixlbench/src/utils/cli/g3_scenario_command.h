/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_G3_SCENARIO_COMMAND_H
#define NIXLBENCH_G3_SCENARIO_COMMAND_H

#include "nixl_types.h"
#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class g3ScenarioCommand : public benchmarkScenario {
public:
    g3ScenarioCommand();

    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<cliOption> &getOptions() const override;
    scenario_type_t scenarioType() const override;
    bool supportsPlugin(nixl_mem_list_t supportedMemoryTypes, nixlBackendPluginCapabilities pluginCapabilities) const override;
    int run(southboundPluginBenchmarkCommand &plugin) override;
    const g3ScenarioRequest &request() const;

private:
    g3ScenarioRequest request_;
    std::vector<cliOption> options_;

    bool isRequestValid(const g3ScenarioRequest &request) const;
};

} // namespace nixlbench

#endif // NIXLBENCH_G3_SCENARIO_COMMAND_H
