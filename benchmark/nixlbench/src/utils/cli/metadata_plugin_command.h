/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_METADATA_PLUGIN_COMMAND_H
#define NIXLBENCH_METADATA_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

#include "nixl.h"

namespace nixlbench {

class metadataPluginCommand : public southboundPluginBenchmarkCommand {
public:
    metadataPluginCommand(std::string backend_name,
                          nixlBackendPluginCapabilities capabilities,
                          nixl_backend_option_list_t option_specs);

    std::string_view
    name() const override;

    std::string_view
    description() const override;

    const std::vector<cliOption> &
    getOptions() const override;

    const nixlBackendPluginCapabilities &
    capabilities() const override;

    const metadata_plugin_option_map_t &
    metadataOptions() const override;

private:
    std::string name_;
    std::string description_;
    nixlBackendPluginCapabilities capabilities_;
    nixl_backend_option_list_t optionSpecs_;
    metadata_plugin_option_map_t optionValues_;
    std::vector<cliOption> options_;
};

} // namespace nixlbench

#endif // NIXLBENCH_METADATA_PLUGIN_COMMAND_H
