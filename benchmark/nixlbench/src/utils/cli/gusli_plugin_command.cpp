/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/gusli_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view gusliPluginCommand::name() const { return "gusli"; }

std::string_view gusliPluginCommand::description() const { return "Use the GUSLI storage backend"; }

const std::vector<cliOption> &gusliPluginCommand::getOptions() const { return options_; }

plugin_type_t gusliPluginCommand::pluginType() const { return plugin_type_t::GUSLI; }

bool gusliPluginCommand::supportsScenario(scenario_type_t scenario) const { return scenario == scenario_type_t::RAW; }

const gusliPluginRequest &gusliPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(gusliPluginCommand)
