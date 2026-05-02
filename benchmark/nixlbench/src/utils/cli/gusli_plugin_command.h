/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_GUSLI_PLUGIN_COMMAND_H
#define NIXLBENCH_GUSLI_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class gusliPluginCommand : public southboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<cliOption> &getOptions() const override;
    plugin_type_t pluginType() const override;
    bool supportsScenario(scenario_type_t scenario) const override;
    const gusliPluginRequest &request() const;

private:
    gusliPluginRequest request_;
    std::vector<cliOption> options_{
        cliOption::option("filepath", "Storage file path", &request_.filepath),
        cliOption::option("filenames", "Storage filenames", &request_.filenames),
        cliOption::option("num_files", "Storage file count", &request_.num_files),
        cliOption::flag("storage_enable_direct", "Enable direct storage I/O", &request_.storage_enable_direct),
        cliOption::option("client_name", "GUSLI client name", &request_.client_name),
        cliOption::option("max_simultaneous_requests", "GUSLI request concurrency", &request_.max_simultaneous_requests),
        cliOption::option("config_file", "GUSLI config file", &request_.config_file),
        cliOption::option("device_byte_offsets", "GUSLI device byte offsets", &request_.device_byte_offsets),
        cliOption::option("device_security", "GUSLI device security flags", &request_.device_security),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_GUSLI_PLUGIN_COMMAND_H
