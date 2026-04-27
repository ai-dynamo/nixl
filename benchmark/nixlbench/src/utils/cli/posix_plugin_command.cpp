/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/posix_plugin_command.h"

namespace nixlbench {

PosixPluginCommand::PosixPluginCommand(bool rawCompatibilityOptions)
    : options_{CliOption::option("storage_path", "Storage path", &request_.storage_path),
               CliOption::option("should-split-dir-per-thread", "Use a directory per thread", &request_.should_split_dir_per_thread),
               CliOption::option("mode", "POSIX mode: aio or iouring", &request_.mode)} {
    if (rawCompatibilityOptions) {
        options_.push_back(CliOption::option("api_type", "Raw POSIX API type", &request_.api_type));
        options_.push_back(CliOption::flag("enable-direct", "Enable direct I/O", &request_.enable_direct));
    }
}

std::string_view PosixPluginCommand::name() const { return "posix"; }

std::string_view PosixPluginCommand::description() const { return "Use the POSIX storage backend"; }

const std::vector<CliOption> &PosixPluginCommand::getOptions() const { return options_; }

PluginType PosixPluginCommand::pluginType() const { return PluginType::Posix; }

bool PosixPluginCommand::supportsScenario(ScenarioType scenario) const {
    return scenario == ScenarioType::G3 || scenario == ScenarioType::G4;
}

const PosixPluginRequest &PosixPluginCommand::request() const { return request_; }

} // namespace nixlbench
