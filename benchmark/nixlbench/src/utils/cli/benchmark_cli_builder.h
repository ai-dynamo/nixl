/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_CLI_BUILDER_H
#define NIXLBENCH_BENCHMARK_CLI_BUILDER_H

#include "utils/cli/azure_blob_plugin_command.h"
#include "utils/cli/g3_scenario_command.h"
#include "utils/cli/g4_scenario_command.h"
#include "utils/cli/gds_mt_plugin_command.h"
#include "utils/cli/gds_plugin_command.h"
#include "utils/cli/gpunetio_plugin_command.h"
#include "utils/cli/gusli_plugin_command.h"
#include "utils/cli/hf3fs_plugin_command.h"
#include "utils/cli/obj_plugin_command.h"
#include "utils/cli/posix_plugin_command.h"
#include "utils/cli/raw_command.h"

#include <memory>

namespace nixlbench {

class BenchmarkCliBuilder {
public:
    BenchmarkCliBuilder();

    int
    parse(int argc, char **argv);

    int
    run();

    bool
    helpRequested() const;

    const IBenchmarkScenario *
    selectedScenario() const;

    const ISouthboundPluginBenchmarkCommand *
    selectedPlugin() const;

private:
    G3ScenarioCommand g3_;
    G4ScenarioCommand g4_;
    RawCommand raw_;
    PosixPluginCommand scenario_g3_posix_;
    PosixPluginCommand scenario_g4_posix_;
    ObjPluginCommand scenario_g4_obj_;
    PosixPluginCommand raw_posix_;
    ObjPluginCommand raw_obj_;
    GdsPluginCommand raw_gds_;
    GdsMtPluginCommand raw_gds_mt_;
    GpuNetIoPluginCommand raw_gpunetio_;
    AzureBlobPluginCommand raw_azure_blob_;
    Hf3fsPluginCommand raw_hf3fs_;
    GusliPluginCommand raw_gusli_;
    IBenchmarkScenario *selected_scenario_ = nullptr;
    ISouthboundPluginBenchmarkCommand *selected_plugin_ = nullptr;
    bool help_ = false;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_CLI_BUILDER_H
