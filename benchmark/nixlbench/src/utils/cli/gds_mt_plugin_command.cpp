/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/gds_mt_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view GdsMtPluginCommand::name() const { return "gds_mt"; }

std::string_view GdsMtPluginCommand::description() const { return "Use the GDS-MT storage backend"; }

const std::vector<CliOption> &GdsMtPluginCommand::getOptions() const { return options_; }

PluginType GdsMtPluginCommand::pluginType() const { return PluginType::GdsMt; }

bool GdsMtPluginCommand::supportsScenario(ScenarioType scenario) const { return scenario == ScenarioType::Raw; }

const GdsMtPluginRequest &GdsMtPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(GdsMtPluginCommand)
