/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/gds_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view gdsPluginCommand::name() const { return "gds"; }

std::string_view gdsPluginCommand::description() const { return "Use the GDS storage backend"; }

const std::vector<cliOption> &gdsPluginCommand::getOptions() const { return options_; }

plugin_type_t gdsPluginCommand::pluginType() const { return plugin_type_t::GDS; }

bool gdsPluginCommand::supportsScenario(scenario_type_t scenario) const { return scenario == scenario_type_t::RAW; }

const gdsPluginRequest &gdsPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(gdsPluginCommand)
