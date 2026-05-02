/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_GPUNETIO_PLUGIN_COMMAND_H
#define NIXLBENCH_GPUNETIO_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class gpunetioPluginCommand : public southboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<cliOption> &getOptions() const override;
    plugin_type_t pluginType() const override;
    bool supportsScenario(scenario_type_t scenario) const override;
    const gpunetioPluginRequest &request() const;

private:
    gpunetioPluginRequest request_;
    std::vector<cliOption> options_{
        cliOption::option("device_list", "GPUNetIO device list", &request_.device_list),
        cliOption::option("oob_list", "GPUNetIO OOB list", &request_.oob_list),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_GPUNETIO_PLUGIN_COMMAND_H
