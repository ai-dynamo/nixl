/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/g4_scenario_command.h"

namespace nixlbench {

G4ScenarioCommand::G4ScenarioCommand()
    : options_{CliOption::option("file_size", "File size", &request_.file_size, true),
               CliOption::option("num_kvs", "Number of key-value pairs", &request_.num_kvs, true),
               CliOption::option("parallel-threads", "Parallel threads", &request_.parallel_threads, true),
               CliOption::option("batch-size", "Batch size", &request_.batch_size, true)} {}

std::string_view G4ScenarioCommand::name() const { return "g4"; }

std::string_view G4ScenarioCommand::description() const { return "Run G4 key-value scenario"; }

const std::vector<CliOption> &G4ScenarioCommand::getOptions() const { return options_; }

ScenarioType G4ScenarioCommand::scenarioType() const { return ScenarioType::G4; }

bool G4ScenarioCommand::supportsPlugin(PluginType plugin) const {
    return plugin == PluginType::Posix || plugin == PluginType::Obj;
}

int G4ScenarioCommand::run(ISouthboundPluginBenchmarkCommand &) { return 0; }

const G4ScenarioRequest &G4ScenarioCommand::request() const { return request_; }

} // namespace nixlbench
