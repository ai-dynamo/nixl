/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_OBJ_PLUGIN_COMMAND_H
#define NIXLBENCH_OBJ_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class ObjPluginCommand : public ISouthboundPluginBenchmarkCommand {
public:
    ObjPluginCommand();

    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<CliOption> &getOptions() const override;
    PluginType pluginType() const override;
    bool supportsScenario(ScenarioType scenario) const override;
    const ObjPluginRequest &request() const;

private:
    ObjPluginRequest request_;
    std::vector<CliOption> options_;
};

} // namespace nixlbench

#endif // NIXLBENCH_OBJ_PLUGIN_COMMAND_H
