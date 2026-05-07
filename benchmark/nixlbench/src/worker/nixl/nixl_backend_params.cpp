/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "worker/nixl/nixl_backend_params.h"
#include "nixl_types.h"

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <utility>

namespace nixlbench {
namespace {

std::string
optionStringValue(const metadataPluginOptionValue &option) {
    if (option.boolValue) {
        return "true";
    }
    return option.value.empty() ? "false" : option.value;
}

bool
isFileWorkloadOption(const std::string &name) {
    return name == "filepath" || name == "filenames" || name == "num_files" ||
           name == "enable_direct";
}

void
applyPosixCompatibility(nixl_b_params_t &backend_params) {
    const auto iter = backend_params.find("use_aio");
    if (iter == backend_params.end()) {
        return;
    }

    const std::string value = iter->second;
    if (value == XFERBENCH_POSIX_API_AIO || value == "aio" || value == "true" || value == "1") {
        backend_params["use_aio"] = "true";
        backend_params["use_uring"] = "false";
        backend_params["use_posix_aio"] = "false";
    } else if (value == XFERBENCH_POSIX_API_URING || value == "iouring" || value == "uring") {
        backend_params["use_aio"] = "false";
        backend_params["use_uring"] = "true";
        backend_params["use_posix_aio"] = "false";
    } else if (value == XFERBENCH_POSIX_API_POSIXAIO || value == "posixaio") {
        backend_params["use_aio"] = "false";
        backend_params["use_uring"] = "false";
        backend_params["use_posix_aio"] = "true";
    }
}

void
applyUcxBenchmarkOverrides(const benchmarkConfig &config,
                           const std::vector<std::string> &devices,
                           int rank,
                           nixl_b_params_t &backend_params) {
    backend_params["num_threads"] = std::to_string(config.worker.progress_threads);
    backend_params["num_workers"] = std::to_string(config.transfer.num_threads + 1);

    if (!devices.empty() && devices[0] != "all") {
        if (rank < config.worker.num_initiator_dev) {
            backend_params["device_list"] = devices[rank];
        } else {
            backend_params["device_list"] = devices[rank - config.worker.num_initiator_dev];
        }
    }
}

void
applyGpuNetIoBenchmarkOverrides(const benchmarkConfig &config,
                                const std::vector<std::string> &devices,
                                nixl_b_params_t &backend_params) {
    if (!devices.empty()) {
        backend_params["network_devices"] = devices[0];
    }
    const auto gpu_devices = config.backend.options.find("gpu_devices");
    if (gpu_devices == config.backend.options.end() || !gpu_devices->second.isProvided) {
        backend_params["gpu_devices"] = "0";
    }
}

void
applyObjBenchmarkOverrides(const benchmarkConfig &config, nixl_b_params_t &backend_params) {
    const auto crt_min_limit = backend_params.find("crtMinLimit");
    if (crt_min_limit != backend_params.end() && std::stoull(crt_min_limit->second) > 0) {
        const auto accelerated = backend_params.find("accelerated");
        if (accelerated != backend_params.end() && accelerated->second == "true") {
            std::cerr << "Warning: Both obj_crt_min_limit and obj_accelerated_enable are set. "
                      << "CRT client will be used (takes precedence over accelerated)."
                      << std::endl;
        }
        return;
    }

    const auto accelerated = backend_params.find("accelerated");
    if (accelerated == backend_params.end() || accelerated->second != "true") {
        backend_params.erase("type");
        return;
    }

    if (!config.backend.options.count("type")) {
        backend_params.erase("type");
    }
}

void
applyGusliBenchmarkOverrides(const std::vector<GusliDeviceConfig> &gusli_devices,
                             nixl_b_params_t &backend_params) {
    const auto config_file = backend_params.find("config_file");
    if (config_file == backend_params.end() || config_file->second.empty()) {
        backend_params["config_file"] = generateGusliConfigFile(gusli_devices);
    }
}

void
applyBenchmarkOverrides(const benchmarkConfig &config,
                        const std::vector<std::string> &devices,
                        bool is_initiator,
                        int rank,
                        const std::vector<GusliDeviceConfig> &gusli_devices,
                        nixl_b_params_t &backend_params) {
    if (config.backend.name == XFERBENCH_BACKEND_UCX) {
        applyUcxBenchmarkOverrides(config, devices, rank, backend_params);
    }  else if (config.backend.name == XFERBENCH_BACKEND_GPUNETIO) {
        applyGpuNetIoBenchmarkOverrides(config, devices, backend_params);
    } else if (config.backend.name == XFERBENCH_BACKEND_OBJ) {
        applyObjBenchmarkOverrides(config, backend_params);
    } else if (config.backend.name == XFERBENCH_BACKEND_GUSLI) {
        applyGusliBenchmarkOverrides(gusli_devices, backend_params);
    } else if (config.backend.name == XFERBENCH_BACKEND_UCCL) {
        backend_params["in_python"] = "0";
    }

    (void)is_initiator;
}

} // namespace

nixl_b_params_t
applyPluginOptions(const metadata_plugin_option_map_t &options, nixl_b_params_t &backend_params) {
    for (const auto &[name, option] : options) {
        if (option.isProvided && !isFileWorkloadOption(name)) {
            backend_params[name] = optionStringValue(option);
        }
    }
    applyPosixCompatibility(backend_params);

    return backend_params;
}

std::vector<GusliDeviceConfig>
buildGusliDeviceConfigs(const benchmarkConfig &config, bool is_initiator) {
    if (config.backend.name != XFERBENCH_BACKEND_GUSLI) {
        return {};
    }

    const auto security = config.backend.options.find("device_security");
    const auto offsets = config.backend.options.find("device_byte_offsets");
    const int expected_num_devices =
        is_initiator ? config.worker.num_initiator_dev : config.worker.num_target_dev;
    return parseGusliDeviceList(
        config.worker.device_list,
        security == config.backend.options.end() ? "" : optionStringValue(security->second),
        offsets == config.backend.options.end() ? "" : optionStringValue(offsets->second),
        expected_num_devices);
}

std::string
generateGusliConfigFile(const std::vector<GusliDeviceConfig> &devices) {
    std::stringstream config;
    config << "# Config file\nversion=1\n";

    for (const auto &dev : devices) {
        config << dev.device_id << " " << dev.device_type << " "
               << "W D "
               << dev.device_path << " " << dev.security_flags << "\n";
    }

    std::cout << "GUSLI Device Config: " << config.str() << std::endl;

    return config.str();
}

nixl_b_params_t
buildNixlBackendParams(const benchmarkConfig &config,
                       nixl_b_params_t backend_params,
                       const std::vector<std::string> &devices,
                       bool is_initiator,
                       int rank,
                       const std::vector<GusliDeviceConfig> &gusli_devices) {
    applyPluginOptions(config.backend.options, backend_params);
    applyBenchmarkOverrides(config,
                            devices,
                            is_initiator,
                            rank,
                            gusli_devices,
                            backend_params);
    return backend_params;
}

} // namespace nixlbench
