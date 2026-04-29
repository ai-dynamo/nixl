/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_AZURE_BLOB_PLUGIN_COMMAND_H
#define NIXLBENCH_AZURE_BLOB_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class AzureBlobPluginCommand : public ISouthboundPluginBenchmarkCommand {
public:
    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    PluginType pluginType() const override;
    bool supportsScenario(ScenarioType scenario) const override;
    const AzureBlobPluginRequest &request() const;

private:
    AzureBlobPluginRequest request_;
    std::vector<CliOption> options_{
        CliOption::option("azure_blob_account_url", "Azure Blob account URL", &request_.azure_blob_account_url),
        CliOption::option("azure_blob_container_name", "Azure Blob container", &request_.azure_blob_container_name),
        CliOption::option("azure_blob_connection_string", "Azure Blob connection string", &request_.azure_blob_connection_string),
    };
};

} // namespace nixlbench

#endif // NIXLBENCH_AZURE_BLOB_PLUGIN_COMMAND_H
