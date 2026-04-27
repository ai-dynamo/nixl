/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/obj_plugin_command.h"

namespace nixlbench {

ObjPluginCommand::ObjPluginCommand()
    : options_{CliOption::option("endpoint-url", "Object endpoint URL", &request_.endpoint_url),
               CliOption::option("bucket_name", "Object bucket name", &request_.bucket_name)} {}

std::string_view ObjPluginCommand::name() const { return "obj"; }

std::string_view ObjPluginCommand::description() const { return "Use the object storage backend"; }

const std::vector<CliOption> &ObjPluginCommand::getOptions() const { return options_; }

PluginType ObjPluginCommand::pluginType() const { return PluginType::Obj; }

bool ObjPluginCommand::supportsScenario(ScenarioType scenario) const {
    return scenario == ScenarioType::G4;
}

const ObjPluginRequest &ObjPluginCommand::request() const { return request_; }

} // namespace nixlbench
