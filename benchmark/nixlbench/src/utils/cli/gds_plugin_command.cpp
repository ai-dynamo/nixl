/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/gds_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view GdsPluginCommand::name() const { return "gds"; }

std::string_view GdsPluginCommand::description() const { return "Use the GDS storage backend"; }

const std::vector<CliOption> &GdsPluginCommand::getOptions() const { return options_; }

PluginType GdsPluginCommand::pluginType() const { return PluginType::Gds; }

bool GdsPluginCommand::supportsScenario(ScenarioType) const { return false; }

const GdsPluginRequest &GdsPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(GdsPluginCommand)
