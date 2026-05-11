/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_RAW_COMMAND_H
#define NIXLBENCH_RAW_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class rawCommand : public benchmarkScenario {
public:
    rawCommand();

    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<cliOption> &getOptions() const override;
    scenario_type_t scenarioType() const override;
    bool supportsPlugin(nixl_mem_list_t supportedMemoryTypes, nixlBackendPluginCapabilities pluginCapabilities) const override;
    int run(southboundPluginBenchmarkCommand &plugin) override;
    const rawRequest &request() const;
    bool finalizeRequest(const southboundPluginBenchmarkCommand &plugin, std::string &error);

private:
    rawRequest request_;
    std::vector<cliOption> options_;
};

} // namespace nixlbench

#endif // NIXLBENCH_RAW_COMMAND_H
