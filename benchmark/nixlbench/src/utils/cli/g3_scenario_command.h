/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_G3_SCENARIO_COMMAND_H
#define NIXLBENCH_G3_SCENARIO_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class G3ScenarioCommand : public IBenchmarkScenario {
public:
    G3ScenarioCommand();

    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    ScenarioType scenarioType() const override;
    const G3ScenarioRequest &request() const;

private:
    G3ScenarioRequest request_;
    std::vector<CliOption> options_;
};

} // namespace nixlbench

#endif // NIXLBENCH_G3_SCENARIO_COMMAND_H
