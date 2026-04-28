/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_GDS_MT_PLUGIN_COMMAND_H
#define NIXLBENCH_GDS_MT_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class GdsMtPluginCommand : public ISouthboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    PluginType pluginType() const override;
    bool supportsScenario(ScenarioType scenario) const override;
    const GdsMtPluginRequest &request() const;

private:
    GdsMtPluginRequest request_;
    std::vector<CliOption> options_{
        CliOption::option("filepath", "Storage file path", &request_.filepath, false, &request_.filepath_provided),
        CliOption::option("filenames", "Storage filenames", &request_.filenames, false, &request_.filenames_provided),
        CliOption::option("num_files", "Storage file count", &request_.num_files, false, &request_.num_files_provided),
        CliOption::flag("storage_enable_direct", "Enable direct storage I/O", &request_.storage_enable_direct, &request_.storage_enable_direct_provided),
        CliOption::option("gds_mt_num_threads", "GDS-MT thread count", &request_.gds_mt_num_threads, false, &request_.gds_mt_num_threads_provided),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_GDS_MT_PLUGIN_COMMAND_H
