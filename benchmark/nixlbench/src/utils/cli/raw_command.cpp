/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/raw_command.h"

#include "utils/cli/obj_plugin_command.h"
#include "utils/cli/posix_plugin_command.h"
#include "utils/cli/raw_execution.h"

#ifdef WITH_GDS_PLUGIN
#include "utils/cli/gds_plugin_command.h"
#endif
#ifdef WITH_GDS_MT_PLUGIN
#include "utils/cli/gds_mt_plugin_command.h"
#endif
#ifdef WITH_GPUNETIO_PLUGIN
#include "utils/cli/gpunetio_plugin_command.h"
#endif
#ifdef WITH_AZURE_BLOB_PLUGIN
#include "utils/cli/azure_blob_plugin_command.h"
#endif
#ifdef WITH_HF3FS_PLUGIN
#include "utils/cli/hf3fs_plugin_command.h"
#endif
#ifdef WITH_GUSLI_PLUGIN
#include "utils/cli/gusli_plugin_command.h"
#endif

#include <toml++/toml.hpp>

#include <exception>
#include <cctype>
#include <memory>
#include <sstream>

namespace nixlbench {
namespace {

constexpr const char *kBackendPosix = "POSIX";
constexpr const char *kBackendObj = "OBJ";
#ifdef WITH_GDS_PLUGIN
constexpr const char *kBackendGds = "GDS";
#endif
#ifdef WITH_GDS_MT_PLUGIN
constexpr const char *kBackendGdsMt = "GDS_MT";
#endif
#ifdef WITH_GPUNETIO_PLUGIN
constexpr const char *kBackendGpuNetIo = "GPUNETIO";
#endif
#ifdef WITH_AZURE_BLOB_PLUGIN
constexpr const char *kBackendAzureBlob = "AZURE_BLOB";
#endif
#ifdef WITH_HF3FS_PLUGIN
constexpr const char *kBackendHf3fs = "HF3FS";
#endif
#ifdef WITH_GUSLI_PLUGIN
constexpr const char *kBackendGusli = "GUSLI";
#endif

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

template <typename storagePluginRequest>
void applyStoragePluginOptions(rawRequest &raw, const storagePluginRequest &plugin) {
    if (plugin.filepath.wasProvided()) {
        raw.filepath.setProvided(plugin.filepath.value);
    }
    if (plugin.filenames.wasProvided()) {
        raw.filenames.setProvided(plugin.filenames.value);
    }
    if (plugin.num_files.wasProvided()) {
        raw.num_files.setProvided(plugin.num_files.value);
    }
    if (plugin.storage_enable_direct.wasProvided()) {
        raw.storage_enable_direct.setProvided(plugin.storage_enable_direct.value);
    }
}

void applyPluginOptions(rawRequest &raw, const southboundPluginBenchmarkCommand &plugin) {
    if (plugin.pluginType() == plugin_type_t::POSIX) {
        const auto &posix = static_cast<const posixPluginCommand &>(plugin).request();
        raw.backend.setProvided(kBackendPosix);
        applyStoragePluginOptions(raw, posix);
        if (posix.api_type.wasProvided()) {
            raw.posix_api_type.setProvided(posix.api_type.value);
        }
        if (posix.ios_pool_size.wasProvided()) {
            raw.posix_ios_pool_size.setProvided(posix.ios_pool_size.value);
        }
        if (posix.kernel_queue_size.wasProvided()) {
            raw.posix_kernel_queue_size.setProvided(posix.kernel_queue_size.value);
        }
        if (posix.api_type.wasProvided()) {
            raw.posix_api_type.setProvided(posix.api_type.value);
        }
        if (posix.enable_direct.wasProvided()) {
            raw.storage_enable_direct.setProvided(posix.enable_direct.value);
        }
    } else if (plugin.pluginType() == plugin_type_t::OBJ) {
        const auto &obj = static_cast<const objPluginCommand &>(plugin).request();
        raw.backend.setProvided(kBackendObj);
        if (obj.access_key.wasProvided()) {
            raw.obj_access_key.setProvided(obj.access_key.value);
        }
        if (obj.secret_key.wasProvided()) {
            raw.obj_secret_key.setProvided(obj.secret_key.value);
        }
        if (obj.session_token.wasProvided()) {
            raw.obj_session_token.setProvided(obj.session_token.value);
        }
        if (obj.bucket_name.wasProvided()) {
            raw.obj_bucket_name.setProvided(obj.bucket_name.value);
        }
        if (obj.scheme.wasProvided()) {
            raw.obj_scheme.setProvided(obj.scheme.value);
        }
        if (obj.region.wasProvided()) {
            raw.obj_region.setProvided(obj.region.value);
        }
        if (obj.use_virtual_addressing.wasProvided()) {
            raw.obj_use_virtual_addressing.setProvided(obj.use_virtual_addressing.value);
        }
        if (obj.endpoint_override.wasProvided()) {
            raw.obj_endpoint_override.setProvided(obj.endpoint_override.value);
        }
        if (obj.req_checksum.wasProvided()) {
            raw.obj_req_checksum.setProvided(obj.req_checksum.value);
        }
        if (obj.ca_bundle.wasProvided()) {
            raw.obj_ca_bundle.setProvided(obj.ca_bundle.value);
        }
        if (obj.crt_min_limit.wasProvided()) {
            raw.obj_crt_min_limit.setProvided(obj.crt_min_limit.value);
        }
        if (obj.accelerated_enable.wasProvided()) {
            raw.obj_accelerated_enable.setProvided(obj.accelerated_enable.value);
        }
        if (obj.accelerated_type.wasProvided()) {
            raw.obj_accelerated_type.setProvided(obj.accelerated_type.value);
        }
    }
#ifdef WITH_GDS_PLUGIN
    else if (plugin.pluginType() == plugin_type_t::GDS) {
        const auto &gds = static_cast<const gdsPluginCommand &>(plugin).request();
        raw.backend.setProvided(kBackendGds);
        applyStoragePluginOptions(raw, gds);
        if (gds.batch_pool_size.wasProvided()) {
            raw.gds_batch_pool_size.setProvided(gds.batch_pool_size.value);
        }
        if (gds.batch_limit.wasProvided()) {
            raw.gds_batch_limit.setProvided(gds.batch_limit.value);
        }
    }
#endif
#ifdef WITH_GDS_MT_PLUGIN
    else if (plugin.pluginType() == plugin_type_t::GDS_MT) {
        const auto &gds_mt = static_cast<const gdsMtPluginCommand &>(plugin).request();
        raw.backend.setProvided(kBackendGdsMt);
        applyStoragePluginOptions(raw, gds_mt);
        if (gds_mt.num_threads.wasProvided()) {
            raw.gds_mt_num_threads.setProvided(gds_mt.num_threads.value);
        }
    }
#endif
#ifdef WITH_GPUNETIO_PLUGIN
    else if (plugin.pluginType() == plugin_type_t::GPUNETIO) {
        const auto &gpunetio = static_cast<const gpunetioPluginCommand &>(plugin).request();
        raw.backend.setProvided(kBackendGpuNetIo);
        if (gpunetio.device_list.wasProvided()) {
            raw.gpunetio_device_list.setProvided(gpunetio.device_list.value);
        }
        if (gpunetio.oob_list.wasProvided()) {
            raw.gpunetio_oob_list.setProvided(gpunetio.oob_list.value);
        }
    }
#endif
#ifdef WITH_AZURE_BLOB_PLUGIN
    else if (plugin.pluginType() == plugin_type_t::AZURE_BLOB) {
        const auto &azure_blob = static_cast<const azureBlobPluginCommand &>(plugin).request();
        raw.backend.setProvided(kBackendAzureBlob);
        if (azure_blob.blob_account_url.wasProvided()) {
            raw.azure_blob_account_url.setProvided(azure_blob.blob_account_url.value);
        }
        if (azure_blob.blob_container_name.wasProvided()) {
            raw.azure_blob_container_name.setProvided(azure_blob.blob_container_name.value);
        }
        if (azure_blob.blob_connection_string.wasProvided()) {
            raw.azure_blob_connection_string.setProvided(azure_blob.blob_connection_string.value);
        }
    }
#endif
#ifdef WITH_HF3FS_PLUGIN
    else if (plugin.pluginType() == plugin_type_t::HF3FS) {
        const auto &hf3fs = static_cast<const hf3fsPluginCommand &>(plugin).request();
        raw.backend.setProvided(kBackendHf3fs);
        applyStoragePluginOptions(raw, hf3fs);
        if (hf3fs.iopool_size.wasProvided()) {
            raw.hf3fs_iopool_size.setProvided(hf3fs.iopool_size.value);
        }
    }
#endif
#ifdef WITH_GUSLI_PLUGIN
    else if (plugin.pluginType() == plugin_type_t::GUSLI) {
        const auto &gusli = static_cast<const gusliPluginCommand &>(plugin).request();
        raw.backend.setProvided(kBackendGusli);
        applyStoragePluginOptions(raw, gusli);
        if (gusli.client_name.wasProvided()) {
            raw.gusli_client_name.setProvided(gusli.client_name.value);
        }
        if (gusli.max_simultaneous_requests.wasProvided()) {
            raw.gusli_max_simultaneous_requests.setProvided(gusli.max_simultaneous_requests.value);
        }
        if (gusli.config_file.wasProvided()) {
            raw.gusli_config_file.setProvided(gusli.config_file.value);
        }
        if (gusli.device_byte_offsets.wasProvided()) {
            raw.gusli_device_byte_offsets.setProvided(gusli.device_byte_offsets.value);
        }
        if (gusli.device_security.wasProvided()) {
            raw.gusli_device_security.setProvided(gusli.device_security.value);
        }
    }
#endif
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

bool rawCommand::supportsPlugin(plugin_type_t plugin) const { return plugin != plugin_type_t::NONE; }

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
