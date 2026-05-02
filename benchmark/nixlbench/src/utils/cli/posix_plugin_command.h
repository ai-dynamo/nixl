/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_POSIX_PLUGIN_COMMAND_H
#define NIXLBENCH_POSIX_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class posixPluginCommand : public southboundPluginBenchmarkCommand {
public:
    posixPluginCommand();

    std::string_view name() const override;
    std::string_view description() const override;
    const std::vector<cliOption> &getOptions() const override;
    plugin_type_t pluginType() const override;
    bool supportsScenario(scenario_type_t scenario) const override;
    const posixPluginRequest &request() const;

private:
    posixPluginRequest request_;
    std::vector<cliOption> options_;
};

} // namespace nixlbench

#endif // NIXLBENCH_POSIX_PLUGIN_COMMAND_H
