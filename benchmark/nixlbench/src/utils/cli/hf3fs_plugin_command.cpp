/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/hf3fs_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view hf3fsPluginCommand::name() const { return "hf3fs"; }

std::string_view hf3fsPluginCommand::description() const { return "Use the HF3FS storage backend"; }

const std::vector<cliOption> &hf3fsPluginCommand::getOptions() const { return options_; }

plugin_type_t hf3fsPluginCommand::pluginType() const { return plugin_type_t::HF3FS; }

bool hf3fsPluginCommand::supportsScenario(scenario_type_t scenario) const { return scenario == scenario_type_t::RAW; }

const hf3fsPluginRequest &hf3fsPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(hf3fsPluginCommand)
