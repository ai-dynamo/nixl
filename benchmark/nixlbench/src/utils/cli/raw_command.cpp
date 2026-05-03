/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/raw_command.h"

#include "utils/cli/raw_execution.h"

#include <toml++/toml.hpp>

#include <exception>
#include <cctype>
#include <memory>
#include <sstream>

namespace nixlbench {
namespace {

constexpr const char *kBackendPosix = "POSIX";
constexpr const char *kBackendObj = "OBJ";
constexpr const char *kBackendGds = "GDS";
constexpr const char *kBackendGdsMt = "GDS_MT";
constexpr const char *kBackendGpuNetIo = "GPUNETIO";
constexpr const char *kBackendAzureBlob = "AZURE_BLOB";
constexpr const char *kBackendHf3fs = "HF3FS";
constexpr const char *kBackendGusli = "GUSLI";

template <typename T>
T getTomlValue(const toml::table *tbl, const char *name, const T &fallback) {
    if (tbl == nullptr) {
        return fallback;
    }
    try {
        return tbl->at_path(name).value<T>().value_or(fallback);
    }
    catch (const std::exception &) {
        return fallback;
    }
}

template <typename T>
void resolveRawArg(rawRequest &raw, const toml::table *tbl, const char *name, providedValue<T> rawRequest::*field) {
    auto &arg = raw.*field;
    if (!arg.wasProvided()) {
        arg.value = getTomlValue(tbl, name, arg.value);
    }
}

#define RESOLVE_RAW_ARG(name) resolveRawArg(raw, tbl, #name, &rawRequest::name)

void resolveConfigValues(rawRequest &raw, const toml::table *tbl) {
    RESOLVE_RAW_ARG(benchmark_group);
    RESOLVE_RAW_ARG(runtime_type);
    RESOLVE_RAW_ARG(worker_type);
    RESOLVE_RAW_ARG(backend);
    RESOLVE_RAW_ARG(initiator_seg_type);
    RESOLVE_RAW_ARG(target_seg_type);
    RESOLVE_RAW_ARG(scheme);
    RESOLVE_RAW_ARG(mode);
    RESOLVE_RAW_ARG(op_type);
    RESOLVE_RAW_ARG(check_consistency);
    RESOLVE_RAW_ARG(total_buffer_size);
    RESOLVE_RAW_ARG(start_block_size);
    RESOLVE_RAW_ARG(max_block_size);
    RESOLVE_RAW_ARG(start_batch_size);
    RESOLVE_RAW_ARG(max_batch_size);
    RESOLVE_RAW_ARG(num_iter);
    RESOLVE_RAW_ARG(recreate_xfer);
    RESOLVE_RAW_ARG(large_blk_iter_ftr);
    RESOLVE_RAW_ARG(warmup_iter);
    RESOLVE_RAW_ARG(num_threads);
    RESOLVE_RAW_ARG(num_initiator_dev);
    RESOLVE_RAW_ARG(num_target_dev);
    RESOLVE_RAW_ARG(enable_pt);
    RESOLVE_RAW_ARG(progress_threads);
    RESOLVE_RAW_ARG(enable_vmm);
    RESOLVE_RAW_ARG(filepath);
    RESOLVE_RAW_ARG(filenames);
    RESOLVE_RAW_ARG(num_files);
    RESOLVE_RAW_ARG(storage_enable_direct);
    RESOLVE_RAW_ARG(gds_batch_pool_size);
    RESOLVE_RAW_ARG(gds_batch_limit);
    RESOLVE_RAW_ARG(gds_mt_num_threads);
    RESOLVE_RAW_ARG(device_list);
    RESOLVE_RAW_ARG(etcd_endpoints);
    RESOLVE_RAW_ARG(posix_api_type);
    RESOLVE_RAW_ARG(posix_ios_pool_size);
    RESOLVE_RAW_ARG(posix_kernel_queue_size);
    RESOLVE_RAW_ARG(gpunetio_device_list);
    RESOLVE_RAW_ARG(gpunetio_oob_list);
    RESOLVE_RAW_ARG(obj_access_key);
    RESOLVE_RAW_ARG(obj_secret_key);
    RESOLVE_RAW_ARG(obj_session_token);
    RESOLVE_RAW_ARG(obj_bucket_name);
    RESOLVE_RAW_ARG(obj_scheme);
    RESOLVE_RAW_ARG(obj_region);
    RESOLVE_RAW_ARG(obj_use_virtual_addressing);
    RESOLVE_RAW_ARG(obj_endpoint_override);
    RESOLVE_RAW_ARG(obj_req_checksum);
    RESOLVE_RAW_ARG(obj_ca_bundle);
    RESOLVE_RAW_ARG(obj_crt_min_limit);
    RESOLVE_RAW_ARG(obj_accelerated_enable);
    RESOLVE_RAW_ARG(obj_accelerated_type);
    RESOLVE_RAW_ARG(azure_blob_account_url);
    RESOLVE_RAW_ARG(azure_blob_container_name);
    RESOLVE_RAW_ARG(azure_blob_connection_string);
    RESOLVE_RAW_ARG(hf3fs_iopool_size);
    RESOLVE_RAW_ARG(gusli_client_name);
    RESOLVE_RAW_ARG(gusli_max_simultaneous_requests);
    RESOLVE_RAW_ARG(gusli_config_file);
    RESOLVE_RAW_ARG(gusli_device_byte_offsets);
    RESOLVE_RAW_ARG(gusli_device_security);
}

#undef RESOLVE_RAW_ARG

const metadataPluginOptionValue *
findOption(const metadata_plugin_option_map_t &options, const std::string &name) {
    const auto iter = options.find(name);
    if (iter == options.end() || !iter->second.isProvided) {
        return nullptr;
    }
    return &iter->second;
}

std::string
optionStringValue(const metadataPluginOptionValue &option) {
    return option.value.empty() ? (option.boolValue ? "true" : "false") : option.value;
}

void setStringOption(providedValue<std::string> &raw,
                     const metadata_plugin_option_map_t &options,
                     const char *name) {
    if (const auto *option = findOption(options, name)) {
        raw.setProvided(optionStringValue(*option));
    }
}

void setBoolOption(providedValue<bool> &raw,
                   const metadata_plugin_option_map_t &options,
                   const char *name) {
    if (const auto *option = findOption(options, name)) {
        raw.setProvided(option->boolValue || option->value == "true");
    }
}

void setIntOption(providedValue<int> &raw,
                  const metadata_plugin_option_map_t &options,
                  const char *name) {
    if (const auto *option = findOption(options, name)) {
        raw.setProvided(std::stoi(optionStringValue(*option)));
    }
}

void setUint64Option(providedValue<uint64_t> &raw,
                     const metadata_plugin_option_map_t &options,
                     const char *name) {
    if (const auto *option = findOption(options, name)) {
        raw.setProvided(std::stoull(optionStringValue(*option)));
    }
}

void applyPluginOptions(rawRequest &raw, const southboundPluginBenchmarkCommand &plugin) {
    const auto &options = plugin.metadataOptions();
    if (plugin.name() == kBackendPosix) {
        raw.backend.setProvided(kBackendPosix);
        if (const auto *option = findOption(options, "use_aio")) {
            const auto value = optionStringValue(*option);
            if (value == "aio" || value == "AIO" || option->boolValue || value == "true") {
                raw.posix_api_type.setProvided("AIO");
            } else if (value == "iouring" || value == "uring" || value == "URING") {
                raw.posix_api_type.setProvided("URING");
            } else if (value == "posixaio" || value == "POSIXAIO") {
                raw.posix_api_type.setProvided("POSIXAIO");
            }
        }
        if (const auto *option = findOption(options, "use_uring")) {
            if (option->boolValue || optionStringValue(*option) == "true") {
                raw.posix_api_type.setProvided("URING");
            }
        }
        if (const auto *option = findOption(options, "use_posix_aio")) {
            if (option->boolValue || optionStringValue(*option) == "true") {
                raw.posix_api_type.setProvided("POSIXAIO");
            }
        }
        setIntOption(raw.posix_ios_pool_size, options, "ios_pool_size");
        setIntOption(raw.posix_kernel_queue_size, options, "kernel_queue_size");
    } else if (plugin.name() == kBackendObj) {
        raw.backend.setProvided(kBackendObj);
        setStringOption(raw.obj_bucket_name, options, "bucket");
        setStringOption(raw.obj_access_key, options, "access_key");
        setStringOption(raw.obj_secret_key, options, "secret_key");
        setStringOption(raw.obj_session_token, options, "session_token");
        setStringOption(raw.obj_scheme, options, "scheme");
        setStringOption(raw.obj_region, options, "region");
        setBoolOption(raw.obj_use_virtual_addressing, options, "use_virtual_addressing");
        setStringOption(raw.obj_endpoint_override, options, "endpoint_override");
        setStringOption(raw.obj_req_checksum, options, "req_checksum");
        setStringOption(raw.obj_ca_bundle, options, "ca_bundle");
        setUint64Option(raw.obj_crt_min_limit, options, "crtMinLimit");
        setBoolOption(raw.obj_accelerated_enable, options, "accelerated");
        setStringOption(raw.obj_accelerated_type, options, "type");
    } else if (plugin.name() == kBackendGds) {
        raw.backend.setProvided(kBackendGds);
        setIntOption(raw.gds_batch_pool_size, options, "batch_pool_size");
        setIntOption(raw.gds_batch_limit, options, "batch_limit");
    } else if (plugin.name() == kBackendGdsMt) {
        raw.backend.setProvided(kBackendGdsMt);
        setIntOption(raw.gds_mt_num_threads, options, "thread_count");
    } else if (plugin.name() == kBackendGpuNetIo) {
        raw.backend.setProvided(kBackendGpuNetIo);
        setStringOption(raw.device_list, options, "network_devices");
        setStringOption(raw.gpunetio_oob_list, options, "oob_interface");
        setStringOption(raw.gpunetio_device_list, options, "gpu_devices");
    } else if (plugin.name() == kBackendAzureBlob) {
        raw.backend.setProvided(kBackendAzureBlob);
        setStringOption(raw.azure_blob_account_url, options, "account_url");
        setStringOption(raw.azure_blob_container_name, options, "container_name");
        setStringOption(raw.azure_blob_connection_string, options, "connection_string");
    } else if (plugin.name() == kBackendHf3fs) {
        raw.backend.setProvided(kBackendHf3fs);
        setStringOption(raw.filepath, options, "mount_point");
        setIntOption(raw.hf3fs_iopool_size, options, "iopool_size");
    } else if (plugin.name() == kBackendGusli) {
        raw.backend.setProvided(kBackendGusli);
        setStringOption(raw.gusli_client_name, options, "client_name");
        setIntOption(raw.gusli_max_simultaneous_requests, options, "max_num_simultaneous_requests");
        setStringOption(raw.gusli_config_file, options, "config_file");
    }
}

} // namespace

rawCommand::rawCommand()
    : options_{cliOption::option("config_file", "Config file", &request_.config_file),
               cliOption::option("benchmark_group", "Benchmark group", &request_.benchmark_group),
               cliOption::option("runtime_type", "Runtime type", &request_.runtime_type),
               cliOption::option("worker_type", "Worker type", &request_.worker_type),
               cliOption::option("backend", "NIXL backend", &request_.backend),
               cliOption::option("initiator_seg_type", "Initiator segment type", &request_.initiator_seg_type),
               cliOption::option("target_seg_type", "Target segment type", &request_.target_seg_type),
               cliOption::option("scheme", "Transfer scheme", &request_.scheme),
               cliOption::option("mode", "Benchmark GPU mode", &request_.mode),
               cliOption::option("op_type", "Operation type", &request_.op_type),
               cliOption::flag("check_consistency", "Enable consistency check", &request_.check_consistency),
               cliOption::option("total_buffer_size", "Total buffer size", &request_.total_buffer_size),
               cliOption::option("start_block_size", "Starting block size", &request_.start_block_size),
               cliOption::option("max_block_size", "Maximum block size", &request_.max_block_size),
               cliOption::option("start_batch_size", "Starting batch size", &request_.start_batch_size),
               cliOption::option("max_batch_size", "Maximum batch size", &request_.max_batch_size),
               cliOption::option("num_iter,num-iterations", "Benchmark iterations", &request_.num_iter),
               cliOption::flag("recreate_xfer", "Recreate transfers each iteration", &request_.recreate_xfer),
               cliOption::option("large_blk_iter_ftr", "Large block iteration factor", &request_.large_blk_iter_ftr),
               cliOption::option("warmup_iter", "Warmup iterations", &request_.warmup_iter),
               cliOption::option("num_threads", "Benchmark threads", &request_.num_threads),
               cliOption::option("num_initiator_dev", "Initiator device count", &request_.num_initiator_dev),
               cliOption::option("num_target_dev", "Target device count", &request_.num_target_dev),
               cliOption::flag("enable_pt", "Enable progress thread", &request_.enable_pt),
               cliOption::option("progress_threads", "Progress thread count", &request_.progress_threads),
               cliOption::flag("enable_vmm", "Enable VMM allocation", &request_.enable_vmm),
               cliOption::option("device_list", "Device list", &request_.device_list),
               cliOption::option("etcd_endpoints", "ETCD endpoints", &request_.etcd_endpoints)} {}

std::string_view rawCommand::name() const { return "raw"; }

std::string_view rawCommand::description() const { return "Run compatibility benchmark command"; }

const std::vector<cliOption> &rawCommand::getOptions() const { return options_; }

const rawRequest &rawCommand::request() const { return request_; }

scenario_type_t rawCommand::scenarioType() const { return scenario_type_t::RAW; }

bool rawCommand::supportsPlugin(nixlBackendPluginCapabilities pluginCapabilities) const { return true; }

int rawCommand::run(southboundPluginBenchmarkCommand &) { return runRawRequest(request_); }

bool rawCommand::finalizeRequest(const southboundPluginBenchmarkCommand &plugin, std::string &error) {
    applyPluginOptions(request_, plugin);

    std::unique_ptr<toml::table> tbl;
    if (!request_.config_file.value.empty()) {
        try {
            tbl = std::make_unique<toml::table>(toml::parse_file(request_.config_file.value));
        }
        catch (const toml::parse_error &err) {
            std::ostringstream oss;
            oss << "Failed to load config file: " << request_.config_file.value << ": " << err.what();
            error = oss.str();
            return false;
        }
    }

    resolveConfigValues(request_, tbl.get());
    return true;
}

} // namespace nixlbench
