/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/g4_scenario_command.h"

namespace nixlbench {

g4ScenarioCommand::g4ScenarioCommand()
    : options_{cliOption::option("file_size", "File size", &request_.file_size, true),
               cliOption::option("num_kvs", "Number of key-value pairs", &request_.num_kvs, true),
               cliOption::option("parallel-threads", "Parallel threads", &request_.parallel_threads, true),
               cliOption::option("batch-size", "Batch size", &request_.batch_size, true)} {}

std::string_view g4ScenarioCommand::name() const { return "g4"; }

std::string_view g4ScenarioCommand::description() const { return "Run G4 key-value scenario"; }

const std::vector<cliOption> &g4ScenarioCommand::getOptions() const { return options_; }

scenario_type_t g4ScenarioCommand::scenarioType() const { return scenario_type_t::G4; }

bool g4ScenarioCommand::supportsPlugin(plugin_type_t plugin) const {
    return plugin == plugin_type_t::POSIX || plugin == plugin_type_t::OBJ;
}

int g4ScenarioCommand::run(southboundPluginBenchmarkCommand &) { return 0; }

const g4ScenarioRequest &g4ScenarioCommand::request() const { return request_; }

} // namespace nixlbench
