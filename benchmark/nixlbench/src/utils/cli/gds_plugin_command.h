/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_GDS_PLUGIN_COMMAND_H
#define NIXLBENCH_GDS_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class GdsPluginCommand : public ISouthboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    PluginType pluginType() const override;
    bool supportsScenario(ScenarioType scenario) const override;
    const GdsPluginRequest &request() const;

private:
    GdsPluginRequest request_;
    std::vector<CliOption> options_{
        CliOption::option("filepath", "Storage file path", &request_.filepath),
        CliOption::option("filenames", "Storage filenames", &request_.filenames),
        CliOption::option("num_files", "Storage file count", &request_.num_files),
        CliOption::flag("storage_enable_direct", "Enable direct storage I/O", &request_.storage_enable_direct),
        CliOption::option("gds_batch_pool_size", "GDS batch pool size", &request_.gds_batch_pool_size),
        CliOption::option("gds_batch_limit", "GDS batch limit", &request_.gds_batch_limit),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_GDS_PLUGIN_COMMAND_H
