/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/gusli_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view GusliPluginCommand::name() const { return "gusli"; }

std::string_view GusliPluginCommand::description() const { return "Use the GUSLI storage backend"; }

const std::vector<CliOption> &GusliPluginCommand::getOptions() const { return options_; }

PluginType GusliPluginCommand::pluginType() const { return PluginType::Gusli; }

bool GusliPluginCommand::supportsScenario(ScenarioType scenario) const { return scenario == ScenarioType::Raw; }

const GusliPluginRequest &GusliPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(GusliPluginCommand)
