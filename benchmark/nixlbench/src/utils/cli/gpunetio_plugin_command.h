/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_GPUNETIO_PLUGIN_COMMAND_H
#define NIXLBENCH_GPUNETIO_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class GpuNetIoPluginCommand : public ISouthboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    PluginType pluginType() const override;
    bool supportsScenario(ScenarioType scenario) const override;
    const GpuNetIoPluginRequest &request() const;

private:
    GpuNetIoPluginRequest request_;
    std::vector<CliOption> options_{
        CliOption::option("gpunetio_device_list", "GPUNetIO device list", &request_.gpunetio_device_list, false, &request_.gpunetio_device_list_provided),
        CliOption::option("gpunetio_oob_list", "GPUNetIO OOB list", &request_.gpunetio_oob_list, false, &request_.gpunetio_oob_list_provided),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_GPUNETIO_PLUGIN_COMMAND_H
