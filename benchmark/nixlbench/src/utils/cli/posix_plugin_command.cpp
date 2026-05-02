/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/posix_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

posixPluginCommand::posixPluginCommand()
    : options_{cliOption::option("should-split-dir-per-thread", "Use a directory per thread", &request_.should_split_dir_per_thread),
               cliOption::option("filepath", "Storage file path", &request_.filepath),
               cliOption::option("filenames", "Storage filenames", &request_.filenames),
               cliOption::option("num_files", "Storage file count", &request_.num_files),
               cliOption::flag("storage_enable_direct", "Enable direct storage I/O", &request_.storage_enable_direct),
               cliOption::option("ios_pool_size", "POSIX IO pool size", &request_.ios_pool_size),
               cliOption::option("kernel_queue_size", "POSIX kernel queue size", &request_.kernel_queue_size),
               cliOption::option("api_type", "POSIX API type", &request_.api_type),
               cliOption::flag("enable-direct", "Enable direct I/O", &request_.enable_direct)}
{}


std::string_view posixPluginCommand::name() const { return "posix"; }

std::string_view posixPluginCommand::description() const { return "Use the POSIX storage backend"; }

const std::vector<cliOption> &posixPluginCommand::getOptions() const { return options_; }

plugin_type_t posixPluginCommand::pluginType() const { return plugin_type_t::POSIX; }

bool posixPluginCommand::supportsScenario(scenario_type_t scenario) const {
    return scenario == scenario_type_t::RAW || scenario == scenario_type_t::G3 || scenario == scenario_type_t::G4;
}

const posixPluginRequest &posixPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(posixPluginCommand)
