/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/gpunetio_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

std::string_view gpunetioPluginCommand::name() const { return "gpunetio"; }

std::string_view gpunetioPluginCommand::description() const { return "Use the GPUNetIO backend"; }

const std::vector<cliOption> &gpunetioPluginCommand::getOptions() const { return options_; }

plugin_type_t gpunetioPluginCommand::pluginType() const { return plugin_type_t::GPUNETIO; }

bool gpunetioPluginCommand::supportsScenario(scenario_type_t scenario) const { return scenario == scenario_type_t::RAW; }

const gpunetioPluginRequest &gpunetioPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(gpunetioPluginCommand)
