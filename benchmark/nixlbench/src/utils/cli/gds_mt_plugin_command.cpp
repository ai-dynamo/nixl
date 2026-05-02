/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/gds_mt_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view gdsMtPluginCommand::name() const { return "gds_mt"; }

std::string_view gdsMtPluginCommand::description() const { return "Use the GDS-MT storage backend"; }

const std::vector<cliOption> &gdsMtPluginCommand::getOptions() const { return options_; }

plugin_type_t gdsMtPluginCommand::pluginType() const { return plugin_type_t::GDS_MT; }

bool gdsMtPluginCommand::supportsScenario(scenario_type_t scenario) const { return scenario == scenario_type_t::RAW; }

const gdsMtPluginRequest &gdsMtPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(gdsMtPluginCommand)
