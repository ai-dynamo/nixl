/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/obj_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

objPluginCommand::objPluginCommand()
    : options_{cliOption::option("bucket_name", "Object bucket name", &request_.bucket_name),
                    cliOption::option("access_key", "S3 access key", &request_.access_key),
                    cliOption::option("secret_key", "S3 secret key", &request_.secret_key),
                    cliOption::option("session_token", "S3 session token", &request_.session_token),
                    cliOption::option("scheme", "S3 scheme", &request_.scheme),
                    cliOption::option("region", "S3 region", &request_.region),
                    cliOption::flag("use_virtual_addressing", "Use S3 virtual addressing", &request_.use_virtual_addressing),
                    cliOption::option("endpoint_override", "S3 endpoint override", &request_.endpoint_override),
                    cliOption::option("req_checksum", "S3 checksum mode", &request_.req_checksum),
                    cliOption::option("ca_bundle", "S3 CA bundle", &request_.ca_bundle),
                    cliOption::option("crt_min_limit", "S3 CRT minimum object size", &request_.crt_min_limit),
                    cliOption::flag("accelerated_enable", "Enable S3 accelerated client", &request_.accelerated_enable),
                    cliOption::option("accelerated_type", "S3 accelerated client type", &request_.accelerated_type)} 
{
}

std::string_view objPluginCommand::name() const { return "obj"; }

std::string_view objPluginCommand::description() const { return "Use the object storage backend"; }

const std::vector<cliOption> &objPluginCommand::getOptions() const { return options_; }

plugin_type_t objPluginCommand::pluginType() const { return plugin_type_t::OBJ; }

bool objPluginCommand::supportsScenario(scenario_type_t scenario) const {
    return scenario == scenario_type_t::RAW || scenario == scenario_type_t::G4;
}

const objPluginRequest &objPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(objPluginCommand)
