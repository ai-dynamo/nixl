/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_WORKER_NIXL_NIXL_BACKEND_PARAMS_H
#define NIXLBENCH_WORKER_NIXL_NIXL_BACKEND_PARAMS_H

#include "benchmark_config.h"
#include "nixl_types.h"
#include "utils/utils.h"

#include <string>
#include <vector>

namespace nixlbench {

std::vector<GusliDeviceConfig>
buildGusliDeviceConfigs(const benchmarkConfig &config, bool is_initiator);

std::string
generateGusliConfigFile(const std::vector<GusliDeviceConfig> &devices);

nixl_b_params_t
applyPluginOptions(const metadata_plugin_option_map_t &options, nixl_b_params_t &backend_params);

nixl_b_params_t
buildNixlBackendParams(const benchmarkConfig &config,
                       nixl_b_params_t backend_params,
                       const std::vector<std::string> &devices,
                       bool is_initiator,
                       int rank,
                       const std::vector<GusliDeviceConfig> &gusli_devices = {});

} // namespace nixlbench

#endif // NIXLBENCH_WORKER_NIXL_NIXL_BACKEND_PARAMS_H
