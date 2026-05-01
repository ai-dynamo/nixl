/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_GUSLI_PLUGIN_COMMAND_H
#define NIXLBENCH_GUSLI_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class GusliPluginCommand : public ISouthboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    PluginType pluginType() const override;
    bool supportsScenario(ScenarioType scenario) const override;
    const GusliPluginRequest &request() const;

private:
    GusliPluginRequest request_;
    std::vector<CliOption> options_{
        CliOption::option("filepath", "Storage file path", &request_.filepath),
        CliOption::option("filenames", "Storage filenames", &request_.filenames),
        CliOption::option("num_files", "Storage file count", &request_.num_files),
        CliOption::flag("storage_enable_direct", "Enable direct storage I/O", &request_.storage_enable_direct),
        CliOption::option("client_name", "GUSLI client name", &request_.client_name),
        CliOption::option("max_simultaneous_requests", "GUSLI request concurrency", &request_.max_simultaneous_requests),
        CliOption::option("config_file", "GUSLI config file", &request_.config_file),
        CliOption::option("device_byte_offsets", "GUSLI device byte offsets", &request_.device_byte_offsets),
        CliOption::option("device_security", "GUSLI device security flags", &request_.device_security),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_GUSLI_PLUGIN_COMMAND_H
