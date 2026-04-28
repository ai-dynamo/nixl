/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/g3_scenario_command.h"

namespace nixlbench {

G3ScenarioCommand::G3ScenarioCommand()
    : options_{CliOption::option("file_size", "File size", &request_.file_size, true),
               CliOption::option("parallel-threads", "Parallel threads", &request_.parallel_threads, true),
               CliOption::option("batch-size", "Batch size", &request_.batch_size, true)} {}

std::string_view G3ScenarioCommand::name() const { return "g3"; }

std::string_view G3ScenarioCommand::description() const { return "Run G3 storage scenario"; }

const std::vector<CliOption> &G3ScenarioCommand::getOptions() const { return options_; }

ScenarioType G3ScenarioCommand::scenarioType() const { return ScenarioType::G3; }

bool G3ScenarioCommand::supportsPlugin(PluginType plugin) const { return plugin == PluginType::Posix; }

int G3ScenarioCommand::run(ISouthboundPluginBenchmarkCommand &) { return 0; }

const G3ScenarioRequest &G3ScenarioCommand::request() const { return request_; }

} // namespace nixlbench
