/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/hf3fs_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view Hf3fsPluginCommand::name() const { return "hf3fs"; }

std::string_view Hf3fsPluginCommand::description() const { return "Use the HF3FS storage backend"; }

const std::vector<CliOption> &Hf3fsPluginCommand::getOptions() const { return options_; }

PluginType Hf3fsPluginCommand::pluginType() const { return PluginType::Hf3fs; }

bool Hf3fsPluginCommand::supportsScenario(ScenarioType) const { return false; }

const Hf3fsPluginRequest &Hf3fsPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(Hf3fsPluginCommand)
