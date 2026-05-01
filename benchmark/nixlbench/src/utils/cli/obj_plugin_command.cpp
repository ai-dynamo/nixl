/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/obj_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

ObjPluginCommand::ObjPluginCommand()
    : options_{CliOption::option("bucket_name", "Object bucket name", &request_.bucket_name),
                    CliOption::option("access_key", "S3 access key", &request_.access_key),
                    CliOption::option("secret_key", "S3 secret key", &request_.secret_key),
                    CliOption::option("session_token", "S3 session token", &request_.session_token),
                    CliOption::option("scheme", "S3 scheme", &request_.scheme),
                    CliOption::option("region", "S3 region", &request_.region),
                    CliOption::flag("use_virtual_addressing", "Use S3 virtual addressing", &request_.use_virtual_addressing),
                    CliOption::option("endpoint_override", "S3 endpoint override", &request_.endpoint_override),
                    CliOption::option("req_checksum", "S3 checksum mode", &request_.req_checksum),
                    CliOption::option("ca_bundle", "S3 CA bundle", &request_.ca_bundle),
                    CliOption::option("crt_min_limit", "S3 CRT minimum object size", &request_.crt_min_limit),
                    CliOption::flag("accelerated_enable", "Enable S3 accelerated client", &request_.accelerated_enable),
                    CliOption::option("accelerated_type", "S3 accelerated client type", &request_.accelerated_type)} 
{
}

std::string_view ObjPluginCommand::name() const { return "obj"; }

std::string_view ObjPluginCommand::description() const { return "Use the object storage backend"; }

const std::vector<CliOption> &ObjPluginCommand::getOptions() const { return options_; }

PluginType ObjPluginCommand::pluginType() const { return PluginType::Obj; }

bool ObjPluginCommand::supportsScenario(ScenarioType scenario) const {
    return scenario == ScenarioType::Raw || scenario == ScenarioType::G4;
}

const ObjPluginRequest &ObjPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(ObjPluginCommand)
