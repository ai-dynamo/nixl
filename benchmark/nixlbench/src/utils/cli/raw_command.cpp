/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/raw_command.h"

#include "utils/cli/azure_blob_plugin_command.h"
#include "utils/cli/gds_mt_plugin_command.h"
#include "utils/cli/gds_plugin_command.h"
#include "utils/cli/gpunetio_plugin_command.h"
#include "utils/cli/gusli_plugin_command.h"
#include "utils/cli/hf3fs_plugin_command.h"
#include "utils/cli/obj_plugin_command.h"
#include "utils/cli/posix_plugin_command.h"

#include <toml++/toml.hpp>

#include <exception>
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
void resolveRawArg(RawRequest &raw, const toml::table *tbl, const char *name, T RawRequest::*value, bool RawRequest::*provided) {
    if (!(raw.*provided)) {
        raw.*value = getTomlValue(tbl, name, raw.*value);
    }
}

#define RESOLVE_RAW_ARG(name) resolveRawArg(raw, tbl, #name, &RawRequest::name, &RawRequest::name##_provided)

void resolveConfigValues(RawRequest &raw, const toml::table *tbl) {
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

template <typename StoragePluginRequest>
void applyStoragePluginOptions(RawRequest &raw, const StoragePluginRequest &plugin) {
    if (plugin.filepath_provided) {
        raw.filepath = plugin.filepath;
        raw.filepath_provided = true;
    }
    if (plugin.filenames_provided) {
        raw.filenames = plugin.filenames;
        raw.filenames_provided = true;
    }
    if (plugin.num_files_provided) {
        raw.num_files = plugin.num_files;
        raw.num_files_provided = true;
    }
    if (plugin.storage_enable_direct_provided) {
        raw.storage_enable_direct = plugin.storage_enable_direct;
        raw.storage_enable_direct_provided = true;
    }
}

void applyPluginOptions(RawRequest &raw, const ISouthboundPluginBenchmarkCommand &plugin) {
    if (plugin.pluginType() == PluginType::Posix) {
        const auto &posix = static_cast<const PosixPluginCommand &>(plugin).request();
        raw.backend = kBackendPosix;
        raw.backend_provided = true;
        applyStoragePluginOptions(raw, posix);
        if (posix.posix_api_type_provided) {
            raw.posix_api_type = posix.posix_api_type;
            raw.posix_api_type_provided = true;
        }
        if (posix.posix_ios_pool_size_provided) {
            raw.posix_ios_pool_size = posix.posix_ios_pool_size;
            raw.posix_ios_pool_size_provided = true;
        }
        if (posix.posix_kernel_queue_size_provided) {
            raw.posix_kernel_queue_size = posix.posix_kernel_queue_size;
            raw.posix_kernel_queue_size_provided = true;
        }
        if (posix.api_type_provided) {
            raw.posix_api_type = posix.api_type;
            raw.posix_api_type_provided = true;
        }
        if (posix.enable_direct_provided) {
            raw.storage_enable_direct = posix.enable_direct;
            raw.storage_enable_direct_provided = true;
        }
    } else if (plugin.pluginType() == PluginType::Obj) {
        const auto &obj = static_cast<const ObjPluginCommand &>(plugin).request();
        raw.backend = kBackendObj;
        raw.backend_provided = true;
        if (obj.obj_access_key_provided) {
            raw.obj_access_key = obj.obj_access_key;
            raw.obj_access_key_provided = true;
        }
        if (obj.obj_secret_key_provided) {
            raw.obj_secret_key = obj.obj_secret_key;
            raw.obj_secret_key_provided = true;
        }
        if (obj.obj_session_token_provided) {
            raw.obj_session_token = obj.obj_session_token;
            raw.obj_session_token_provided = true;
        }
        if (obj.obj_bucket_name_provided) {
            raw.obj_bucket_name = obj.obj_bucket_name;
            raw.obj_bucket_name_provided = true;
        }
        if (!obj.bucket_name.empty()) {
            raw.obj_bucket_name = obj.bucket_name;
            raw.obj_bucket_name_provided = true;
        }
        if (obj.obj_scheme_provided) {
            raw.obj_scheme = obj.obj_scheme;
            raw.obj_scheme_provided = true;
        }
        if (obj.obj_region_provided) {
            raw.obj_region = obj.obj_region;
            raw.obj_region_provided = true;
        }
        if (obj.obj_use_virtual_addressing_provided) {
            raw.obj_use_virtual_addressing = obj.obj_use_virtual_addressing;
            raw.obj_use_virtual_addressing_provided = true;
        }
        if (obj.obj_endpoint_override_provided) {
            raw.obj_endpoint_override = obj.obj_endpoint_override;
            raw.obj_endpoint_override_provided = true;
        }
        if (!obj.endpoint_url.empty()) {
            raw.obj_endpoint_override = obj.endpoint_url;
            raw.obj_endpoint_override_provided = true;
        }
        if (obj.obj_req_checksum_provided) {
            raw.obj_req_checksum = obj.obj_req_checksum;
            raw.obj_req_checksum_provided = true;
        }
        if (obj.obj_ca_bundle_provided) {
            raw.obj_ca_bundle = obj.obj_ca_bundle;
            raw.obj_ca_bundle_provided = true;
        }
        if (obj.obj_crt_min_limit_provided) {
            raw.obj_crt_min_limit = obj.obj_crt_min_limit;
            raw.obj_crt_min_limit_provided = true;
        }
        if (obj.obj_accelerated_enable_provided) {
            raw.obj_accelerated_enable = obj.obj_accelerated_enable;
            raw.obj_accelerated_enable_provided = true;
        }
        if (obj.obj_accelerated_type_provided) {
            raw.obj_accelerated_type = obj.obj_accelerated_type;
            raw.obj_accelerated_type_provided = true;
        }
    } else if (plugin.pluginType() == PluginType::Gds) {
        const auto &gds = static_cast<const GdsPluginCommand &>(plugin).request();
        raw.backend = kBackendGds;
        raw.backend_provided = true;
        applyStoragePluginOptions(raw, gds);
        if (gds.gds_batch_pool_size_provided) {
            raw.gds_batch_pool_size = gds.gds_batch_pool_size;
            raw.gds_batch_pool_size_provided = true;
        }
        if (gds.gds_batch_limit_provided) {
            raw.gds_batch_limit = gds.gds_batch_limit;
            raw.gds_batch_limit_provided = true;
        }
    } else if (plugin.pluginType() == PluginType::GdsMt) {
        const auto &gds_mt = static_cast<const GdsMtPluginCommand &>(plugin).request();
        raw.backend = kBackendGdsMt;
        raw.backend_provided = true;
        applyStoragePluginOptions(raw, gds_mt);
        if (gds_mt.gds_mt_num_threads_provided) {
            raw.gds_mt_num_threads = gds_mt.gds_mt_num_threads;
            raw.gds_mt_num_threads_provided = true;
        }
    } else if (plugin.pluginType() == PluginType::GpuNetIo) {
        const auto &gpunetio = static_cast<const GpuNetIoPluginCommand &>(plugin).request();
        raw.backend = kBackendGpuNetIo;
        raw.backend_provided = true;
        if (gpunetio.gpunetio_device_list_provided) {
            raw.gpunetio_device_list = gpunetio.gpunetio_device_list;
            raw.gpunetio_device_list_provided = true;
        }
        if (gpunetio.gpunetio_oob_list_provided) {
            raw.gpunetio_oob_list = gpunetio.gpunetio_oob_list;
            raw.gpunetio_oob_list_provided = true;
        }
    } else if (plugin.pluginType() == PluginType::AzureBlob) {
        const auto &azure_blob = static_cast<const AzureBlobPluginCommand &>(plugin).request();
        raw.backend = kBackendAzureBlob;
        raw.backend_provided = true;
        if (azure_blob.azure_blob_account_url_provided) {
            raw.azure_blob_account_url = azure_blob.azure_blob_account_url;
            raw.azure_blob_account_url_provided = true;
        }
        if (azure_blob.azure_blob_container_name_provided) {
            raw.azure_blob_container_name = azure_blob.azure_blob_container_name;
            raw.azure_blob_container_name_provided = true;
        }
        if (azure_blob.azure_blob_connection_string_provided) {
            raw.azure_blob_connection_string = azure_blob.azure_blob_connection_string;
            raw.azure_blob_connection_string_provided = true;
        }
    } else if (plugin.pluginType() == PluginType::Hf3fs) {
        const auto &hf3fs = static_cast<const Hf3fsPluginCommand &>(plugin).request();
        raw.backend = kBackendHf3fs;
        raw.backend_provided = true;
        applyStoragePluginOptions(raw, hf3fs);
        if (hf3fs.hf3fs_iopool_size_provided) {
            raw.hf3fs_iopool_size = hf3fs.hf3fs_iopool_size;
            raw.hf3fs_iopool_size_provided = true;
        }
    } else if (plugin.pluginType() == PluginType::Gusli) {
        const auto &gusli = static_cast<const GusliPluginCommand &>(plugin).request();
        raw.backend = kBackendGusli;
        raw.backend_provided = true;
        applyStoragePluginOptions(raw, gusli);
        if (gusli.gusli_client_name_provided) {
            raw.gusli_client_name = gusli.gusli_client_name;
            raw.gusli_client_name_provided = true;
        }
        if (gusli.gusli_max_simultaneous_requests_provided) {
            raw.gusli_max_simultaneous_requests = gusli.gusli_max_simultaneous_requests;
            raw.gusli_max_simultaneous_requests_provided = true;
        }
        if (gusli.gusli_config_file_provided) {
            raw.gusli_config_file = gusli.gusli_config_file;
            raw.gusli_config_file_provided = true;
        }
        if (gusli.gusli_device_byte_offsets_provided) {
            raw.gusli_device_byte_offsets = gusli.gusli_device_byte_offsets;
            raw.gusli_device_byte_offsets_provided = true;
        }
        if (gusli.gusli_device_security_provided) {
            raw.gusli_device_security = gusli.gusli_device_security;
            raw.gusli_device_security_provided = true;
        }
    }
}

} // namespace

RawCommand::RawCommand()
    : options_{CliOption::option("config_file", "Config file", &request_.config_file, false, &request_.config_file_provided),
               CliOption::option("benchmark_group", "Benchmark group", &request_.benchmark_group, false, &request_.benchmark_group_provided),
               CliOption::option("runtime_type", "Runtime type", &request_.runtime_type, false, &request_.runtime_type_provided),
               CliOption::option("worker_type", "Worker type", &request_.worker_type, false, &request_.worker_type_provided),
               CliOption::option("backend", "NIXL backend", &request_.backend, false, &request_.backend_provided),
               CliOption::option("initiator_seg_type", "Initiator segment type", &request_.initiator_seg_type, false, &request_.initiator_seg_type_provided),
               CliOption::option("target_seg_type", "Target segment type", &request_.target_seg_type, false, &request_.target_seg_type_provided),
               CliOption::option("scheme", "Transfer scheme", &request_.scheme, false, &request_.scheme_provided),
               CliOption::option("mode", "Benchmark GPU mode", &request_.mode, false, &request_.mode_provided),
               CliOption::option("op_type", "Operation type", &request_.op_type, false, &request_.op_type_provided),
               CliOption::flag("check_consistency", "Enable consistency check", &request_.check_consistency, &request_.check_consistency_provided),
               CliOption::option("total_buffer_size", "Total buffer size", &request_.total_buffer_size, false, &request_.total_buffer_size_provided),
               CliOption::option("start_block_size", "Starting block size", &request_.start_block_size, false, &request_.start_block_size_provided),
               CliOption::option("max_block_size", "Maximum block size", &request_.max_block_size, false, &request_.max_block_size_provided),
               CliOption::option("start_batch_size", "Starting batch size", &request_.start_batch_size, false, &request_.start_batch_size_provided),
               CliOption::option("max_batch_size", "Maximum batch size", &request_.max_batch_size, false, &request_.max_batch_size_provided),
               CliOption::option("num_iter,num-iterations", "Benchmark iterations", &request_.num_iter, false, &request_.num_iter_provided),
               CliOption::flag("recreate_xfer", "Recreate transfers each iteration", &request_.recreate_xfer, &request_.recreate_xfer_provided),
               CliOption::option("large_blk_iter_ftr", "Large block iteration factor", &request_.large_blk_iter_ftr, false, &request_.large_blk_iter_ftr_provided),
               CliOption::option("warmup_iter", "Warmup iterations", &request_.warmup_iter, false, &request_.warmup_iter_provided),
               CliOption::option("num_threads", "Benchmark threads", &request_.num_threads, false, &request_.num_threads_provided),
               CliOption::option("num_initiator_dev", "Initiator device count", &request_.num_initiator_dev, false, &request_.num_initiator_dev_provided),
               CliOption::option("num_target_dev", "Target device count", &request_.num_target_dev, false, &request_.num_target_dev_provided),
               CliOption::flag("enable_pt", "Enable progress thread", &request_.enable_pt, &request_.enable_pt_provided),
               CliOption::option("progress_threads", "Progress thread count", &request_.progress_threads, false, &request_.progress_threads_provided),
                CliOption::flag("enable_vmm", "Enable VMM allocation", &request_.enable_vmm, &request_.enable_vmm_provided),
                CliOption::option("device_list", "Device list", &request_.device_list, false, &request_.device_list_provided),
                CliOption::option("etcd_endpoints", "ETCD endpoints", &request_.etcd_endpoints, false, &request_.etcd_endpoints_provided)} {}

std::string_view RawCommand::name() const { return "raw"; }

std::string_view RawCommand::description() const { return "Run compatibility benchmark command"; }

const std::vector<CliOption> &RawCommand::getOptions() const { return options_; }

const RawRequest &RawCommand::request() const { return request_; }

ScenarioType RawCommand::scenarioType() const { return ScenarioType::Raw; }

bool RawCommand::supportsPlugin(PluginType plugin) const { return plugin != PluginType::None; }

int RawCommand::run(ISouthboundPluginBenchmarkCommand &) { return 0; }

bool RawCommand::finalizeRequest(const ISouthboundPluginBenchmarkCommand &plugin, std::string &error) {
    applyPluginOptions(request_, plugin);

    std::unique_ptr<toml::table> tbl;
    if (!request_.config_file.empty()) {
        try {
            tbl = std::make_unique<toml::table>(toml::parse_file(request_.config_file));
        }
        catch (const toml::parse_error &err) {
            std::ostringstream oss;
            oss << "Failed to load config file: " << request_.config_file << ": " << err.what();
            error = oss.str();
            return false;
        }
    }

    resolveConfigValues(request_, tbl.get());
    return true;
}

} // namespace nixlbench
