/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_HF3FS_PLUGIN_COMMAND_H
#define NIXLBENCH_HF3FS_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class Hf3fsPluginCommand : public ISouthboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    PluginType pluginType() const override;
    bool supportsScenario(ScenarioType scenario) const override;
    const Hf3fsPluginRequest &request() const;

private:
    Hf3fsPluginRequest request_;
    std::vector<CliOption> options_{
        CliOption::option("filepath", "Storage file path", &request_.filepath),
        CliOption::option("filenames", "Storage filenames", &request_.filenames),
        CliOption::option("num_files", "Storage file count", &request_.num_files),
        CliOption::flag("storage_enable_direct", "Enable direct storage I/O", &request_.storage_enable_direct),
        CliOption::option("iopool_size", "HF3FS IO pool size", &request_.iopool_size),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_HF3FS_PLUGIN_COMMAND_H
