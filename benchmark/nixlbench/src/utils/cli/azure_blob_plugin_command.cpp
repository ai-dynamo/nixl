/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/azure_blob_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view AzureBlobPluginCommand::name() const { return "azure_blob"; }

std::string_view AzureBlobPluginCommand::description() const { return "Use the Azure Blob backend"; }

const std::vector<CliOption> &AzureBlobPluginCommand::getOptions() const { return options_; }

PluginType AzureBlobPluginCommand::pluginType() const { return PluginType::AzureBlob; }

bool AzureBlobPluginCommand::supportsScenario(ScenarioType) const { return false; }

const AzureBlobPluginRequest &AzureBlobPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(AzureBlobPluginCommand)
