/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_GDS_MT_PLUGIN_COMMAND_H
#define NIXLBENCH_GDS_MT_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class gdsMtPluginCommand : public southboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<cliOption> &getOptions() const override;
    plugin_type_t pluginType() const override;
    bool supportsScenario(scenario_type_t scenario) const override;
    const gdsMtPluginRequest &request() const;

private:
    gdsMtPluginRequest request_;
    std::vector<cliOption> options_{
        cliOption::option("filepath", "Storage file path", &request_.filepath),
        cliOption::option("filenames", "Storage filenames", &request_.filenames),
        cliOption::option("num_files", "Storage file count", &request_.num_files),
        cliOption::flag("storage_enable_direct", "Enable direct storage I/O", &request_.storage_enable_direct),
        cliOption::option("num_threads", "GDS-MT thread count", &request_.num_threads),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_GDS_MT_PLUGIN_COMMAND_H
