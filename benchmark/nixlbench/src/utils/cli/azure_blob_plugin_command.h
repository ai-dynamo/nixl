/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_AZURE_BLOB_PLUGIN_COMMAND_H
#define NIXLBENCH_AZURE_BLOB_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class azureBlobPluginCommand : public southboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<cliOption> &getOptions() const override;
    plugin_type_t pluginType() const override;
    bool supportsScenario(scenario_type_t scenario) const override;
    const azureBlobPluginRequest &request() const;

private:
    azureBlobPluginRequest request_;
    std::vector<cliOption> options_{
        cliOption::option("account_url", "Azure Blob account URL", &request_.blob_account_url),
        cliOption::option("container_name", "Azure Blob container", &request_.blob_container_name),
        cliOption::option("connection_string", "Azure Blob connection string", &request_.blob_connection_string),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_AZURE_BLOB_PLUGIN_COMMAND_H
