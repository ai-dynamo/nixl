/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/posix_plugin_command.h"
#include "utils/cli/plugin_registry.h"

namespace nixlbench {

PosixPluginCommand::PosixPluginCommand()
    : options_{CliOption::option("storage_path", "Storage path", &request_.storage_path),
               CliOption::option("should-split-dir-per-thread", "Use a directory per thread", &request_.should_split_dir_per_thread),
               CliOption::option("mode", "POSIX mode: aio or iouring", &request_.mode),
               CliOption::option("filepath", "Storage file path", &request_.filepath),
               CliOption::option("filenames", "Storage filenames", &request_.filenames),
               CliOption::option("num_files", "Storage file count", &request_.num_files),
               CliOption::flag("storage_enable_direct", "Enable direct storage I/O", &request_.storage_enable_direct),
               CliOption::option("ios_pool_size", "POSIX IO pool size", &request_.ios_pool_size),
               CliOption::option("kernel_queue_size", "POSIX kernel queue size", &request_.kernel_queue_size),
               CliOption::option("api_type", "POSIX API type", &request_.api_type),
               CliOption::flag("enable-direct", "Enable direct I/O", &request_.enable_direct)}
{}


std::string_view PosixPluginCommand::name() const { return "posix"; }

std::string_view PosixPluginCommand::description() const { return "Use the POSIX storage backend"; }

const std::vector<CliOption> &PosixPluginCommand::getOptions() const { return options_; }

PluginType PosixPluginCommand::pluginType() const { return PluginType::Posix; }

bool PosixPluginCommand::supportsScenario(ScenarioType scenario) const {
    return scenario == ScenarioType::Raw || scenario == ScenarioType::G3 || scenario == ScenarioType::G4;
}

const PosixPluginRequest &PosixPluginCommand::request() const { return request_; }

} // namespace nixlbench

REGISTER_SOUTHBOUND_PLUGIN(PosixPluginCommand)
