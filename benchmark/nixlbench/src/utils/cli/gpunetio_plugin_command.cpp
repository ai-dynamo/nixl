/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/gpunetio_plugin_command.h"

namespace nixlbench {

std::string_view GpuNetIoPluginCommand::name() const { return "gpunetio"; }

std::string_view GpuNetIoPluginCommand::description() const { return "Use the GPUNetIO backend"; }

const std::vector<CliOption> &GpuNetIoPluginCommand::getOptions() const { return options_; }

PluginType GpuNetIoPluginCommand::pluginType() const { return PluginType::GpuNetIo; }

bool GpuNetIoPluginCommand::supportsScenario(ScenarioType) const { return false; }

const GpuNetIoPluginRequest &GpuNetIoPluginCommand::request() const { return request_; }

} // namespace nixlbench
