/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/g3_scenario_command.h"

namespace nixlbench {

g3ScenarioCommand::g3ScenarioCommand()
    : options_{cliOption::option("file_size", "File size", &request_.file_size, true),
               cliOption::option("parallel-threads", "Parallel threads", &request_.parallel_threads, true),
               cliOption::option("batch-size", "Batch size", &request_.batch_size, true)} {}

std::string_view g3ScenarioCommand::name() const { return "g3"; }

std::string_view g3ScenarioCommand::description() const { return "Run G3 storage scenario"; }

const std::vector<cliOption> &g3ScenarioCommand::getOptions() const { return options_; }

scenario_type_t g3ScenarioCommand::scenarioType() const { return scenario_type_t::G3; }

bool g3ScenarioCommand::supportsPlugin(plugin_type_t plugin) const { return plugin == plugin_type_t::POSIX; }

int g3ScenarioCommand::run(southboundPluginBenchmarkCommand &) { return 0; }

const g3ScenarioRequest &g3ScenarioCommand::request() const { return request_; }

} // namespace nixlbench
