/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_config.h"

#include "utils/utils.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace nixlbench {
namespace {

metadataPluginOptionValue
makeStringOption(std::string value) {
    return metadataPluginOptionValue{std::move(value), false, true};
}

metadataPluginOptionValue
makeBoolOption(bool value) {
    return metadataPluginOptionValue{value ? "true" : "false", value, true};
}

void
setOption(metadata_plugin_option_map_t &options, const std::string &name, std::string value) {
    options[name] = makeStringOption(std::move(value));
}

void
setOption(metadata_plugin_option_map_t &options, const std::string &name, bool value) {
    options[name] = makeBoolOption(value);
}

void
setOption(metadata_plugin_option_map_t &options, const std::string &name, int value) {
    options[name] = makeStringOption(std::to_string(value));
}

void
setOption(metadata_plugin_option_map_t &options, const std::string &name, std::uint64_t value) {
    options[name] = makeStringOption(std::to_string(value));
}

bool
isLegacyStorageBackend(std::string_view backend) {
    return backend == XFERBENCH_BACKEND_GDS || backend == XFERBENCH_BACKEND_GDS_MT ||
           backend == XFERBENCH_BACKEND_HF3FS || backend == XFERBENCH_BACKEND_POSIX ||
           backend == XFERBENCH_BACKEND_OBJ || backend == XFERBENCH_BACKEND_GUSLI ||
           backend == XFERBENCH_BACKEND_AZURE_BLOB;
}

bool
isLegacyFileWorkloadBackend(std::string_view backend) {
    return backend == XFERBENCH_BACKEND_GDS || backend == XFERBENCH_BACKEND_GDS_MT ||
           backend == XFERBENCH_BACKEND_HF3FS || backend == XFERBENCH_BACKEND_POSIX ||
           backend == XFERBENCH_BACKEND_GUSLI;
}

bool
isLegacyNetworkDestinationBackend(std::string_view backend) {
    return backend == XFERBENCH_BACKEND_UCX || backend == XFERBENCH_BACKEND_LIBFABRIC ||
           backend == XFERBENCH_BACKEND_GPUNETIO || backend == XFERBENCH_BACKEND_MOONCAKE ||
           backend == XFERBENCH_BACKEND_UCCL;
}

nixlBackendPluginCapabilities
legacyBackendCapabilities(std::string_view backend) {
    nixlBackendPluginCapabilities capabilities;
    capabilities.canUseAsStorage = isLegacyStorageBackend(backend);
    capabilities.canUseAsNetworkDestination = isLegacyNetworkDestinationBackend(backend);
    capabilities.canReadWriteFiles = isLegacyFileWorkloadBackend(backend);
    return capabilities;
}

template<typename T>
void
setOption(metadata_plugin_option_map_t &options, const std::string &name, const providedValue<T> &value) {
    setOption(options, name, value.value);
}

void
setPosixApiOptions(metadata_plugin_option_map_t &options, const std::string &api_type) {
    setOption(options, "use_aio", api_type == XFERBENCH_POSIX_API_AIO);
    setOption(options, "use_uring", api_type == XFERBENCH_POSIX_API_URING);
    setOption(options, "use_posix_aio", api_type == XFERBENCH_POSIX_API_POSIXAIO);
}

void
addLegacyBackendOptions(benchmarkConfig &config, const xferBenchConfig &legacy_config) {
    auto &options = config.backend.options;

    if (legacy_config.backend == XFERBENCH_BACKEND_GDS) {
        setOption(options, "batch_pool_size", legacy_config.gds_batch_pool_size);
        setOption(options, "batch_limit", legacy_config.gds_batch_limit);
    } else if (legacy_config.backend == XFERBENCH_BACKEND_GDS_MT) {
        setOption(options, "thread_count", legacy_config.gds_mt_num_threads);
    } else if (legacy_config.backend == XFERBENCH_BACKEND_POSIX) {
        setPosixApiOptions(options, legacy_config.posix_api_type);
        setOption(options, "ios_pool_size", legacy_config.posix_ios_pool_size);
        setOption(options, "kernel_queue_size", legacy_config.posix_kernel_queue_size);
    } else if (legacy_config.backend == XFERBENCH_BACKEND_GPUNETIO) {
        setOption(options, "network_devices", legacy_config.device_list);
        setOption(options, "gpu_devices", legacy_config.gpunetio_device_list);
        setOption(options, "oob_interface", legacy_config.gpunetio_oob_list);
    } else if (legacy_config.backend == XFERBENCH_BACKEND_HF3FS) {
        setOption(options, "iopool_size", legacy_config.hf3fs_iopool_size);
    } else if (legacy_config.backend == XFERBENCH_BACKEND_OBJ) {
        setOption(options, "access_key", legacy_config.obj_access_key);
        setOption(options, "secret_key", legacy_config.obj_secret_key);
        setOption(options, "session_token", legacy_config.obj_session_token);
        setOption(options, "bucket", legacy_config.obj_bucket_name);
        setOption(options, "scheme", legacy_config.obj_scheme);
        setOption(options, "region", legacy_config.obj_region);
        setOption(options, "use_virtual_addressing", legacy_config.obj_use_virtual_addressing);
        setOption(options, "endpoint_override", legacy_config.obj_endpoint_override);
        setOption(options, "req_checksum", legacy_config.obj_req_checksum);
        setOption(options, "ca_bundle", legacy_config.obj_ca_bundle);
        setOption(options, "crtMinLimit", static_cast<std::uint64_t>(legacy_config.obj_crt_min_limit));
        setOption(options, "accelerated", legacy_config.obj_accelerated_enable);
        setOption(options, "type", legacy_config.obj_accelerated_type);
    } else if (legacy_config.backend == XFERBENCH_BACKEND_GUSLI) {
        setOption(options, "client_name", legacy_config.gusli_client_name);
        setOption(options,
                  "max_num_simultaneous_requests",
                  legacy_config.gusli_max_simultaneous_requests);
        setOption(options, "config_file", legacy_config.gusli_config_file);
        setOption(options, "device_byte_offsets", legacy_config.gusli_device_byte_offsets);
        setOption(options, "device_security", legacy_config.gusli_device_security);
    } else if (legacy_config.backend == XFERBENCH_BACKEND_AZURE_BLOB) {
        setOption(options, "account_url", legacy_config.azure_blob_account_url);
        setOption(options, "container_name", legacy_config.azure_blob_container_name);
        setOption(options, "connection_string", legacy_config.azure_blob_connection_string);
    }
}

void
addRawBackendOptions(benchmarkConfig &config, const rawRequest &request) {
    auto &options = config.backend.options;

    if (request.backend.value == XFERBENCH_BACKEND_GDS) {
        setOption(options, "batch_pool_size", request.gds_batch_pool_size);
        setOption(options, "batch_limit", request.gds_batch_limit);
    } else if (request.backend.value == XFERBENCH_BACKEND_GDS_MT) {
        setOption(options, "thread_count", request.gds_mt_num_threads);
    } else if (request.backend.value == XFERBENCH_BACKEND_POSIX) {
        setPosixApiOptions(options, request.posix_api_type.value);
        setOption(options, "ios_pool_size", request.posix_ios_pool_size);
        setOption(options, "kernel_queue_size", request.posix_kernel_queue_size);
    } else if (request.backend.value == XFERBENCH_BACKEND_GPUNETIO) {
        setOption(options, "network_devices", request.device_list);
        setOption(options, "gpu_devices", request.gpunetio_device_list);
        setOption(options, "oob_interface", request.gpunetio_oob_list);
    } else if (request.backend.value == XFERBENCH_BACKEND_HF3FS) {
        setOption(options, "iopool_size", request.hf3fs_iopool_size);
    } else if (request.backend.value == XFERBENCH_BACKEND_OBJ) {
        setOption(options, "access_key", request.obj_access_key);
        setOption(options, "secret_key", request.obj_secret_key);
        setOption(options, "session_token", request.obj_session_token);
        setOption(options, "bucket", request.obj_bucket_name);
        setOption(options, "scheme", request.obj_scheme);
        setOption(options, "region", request.obj_region);
        setOption(options, "use_virtual_addressing", request.obj_use_virtual_addressing);
        setOption(options, "endpoint_override", request.obj_endpoint_override);
        setOption(options, "req_checksum", request.obj_req_checksum);
        setOption(options, "ca_bundle", request.obj_ca_bundle);
        setOption(options, "crtMinLimit", request.obj_crt_min_limit);
        setOption(options, "accelerated", request.obj_accelerated_enable);
        setOption(options, "type", request.obj_accelerated_type);
    } else if (request.backend.value == XFERBENCH_BACKEND_GUSLI) {
        setOption(options, "client_name", request.gusli_client_name);
        setOption(options,
                  "max_num_simultaneous_requests",
                  request.gusli_max_simultaneous_requests);
        setOption(options, "config_file", request.gusli_config_file);
        setOption(options, "device_byte_offsets", request.gusli_device_byte_offsets);
        setOption(options, "device_security", request.gusli_device_security);
    } else if (request.backend.value == XFERBENCH_BACKEND_AZURE_BLOB) {
        setOption(options, "account_url", request.azure_blob_account_url);
        setOption(options, "container_name", request.azure_blob_container_name);
        setOption(options, "connection_string", request.azure_blob_connection_string);
    }
}

} // namespace

benchmarkConfig
makeBenchmarkConfigFromLegacy(const xferBenchConfig &legacy_config) {
    benchmarkConfig config;

    config.common.benchmark_group = legacy_config.benchmark_group;
    config.common.check_consistency = legacy_config.check_consistency;
    config.common.recreate_xfer = legacy_config.recreate_xfer;
    config.common.num_iter = legacy_config.num_iter;
    config.common.large_blk_iter_ftr = legacy_config.large_blk_iter_ftr;
    config.common.warmup_iter = legacy_config.warmup_iter;

    config.runtime.type = legacy_config.runtime_type;
    config.runtime.etcd_endpoints = legacy_config.etcd_endpoints;

    config.transfer.initiator_seg_type = legacy_config.initiator_seg_type;
    config.transfer.target_seg_type = legacy_config.target_seg_type;
    config.transfer.scheme = legacy_config.scheme;
    config.transfer.mode = legacy_config.mode;
    config.transfer.op_type = legacy_config.op_type;
    config.transfer.total_buffer_size = legacy_config.total_buffer_size;
    config.transfer.start_block_size = legacy_config.start_block_size;
    config.transfer.max_block_size = legacy_config.max_block_size;
    config.transfer.start_batch_size = legacy_config.start_batch_size;
    config.transfer.max_batch_size = legacy_config.max_batch_size;
    config.transfer.num_threads = legacy_config.num_threads;

    config.worker.type = legacy_config.worker_type;
    config.worker.num_initiator_dev = legacy_config.num_initiator_dev;
    config.worker.num_target_dev = legacy_config.num_target_dev;
    config.worker.enable_pt = legacy_config.enable_pt;
    config.worker.progress_threads = legacy_config.progress_threads;
    config.worker.device_list = legacy_config.device_list;
    config.worker.enable_vmm = legacy_config.enable_vmm;

    config.backend.name = legacy_config.backend;
    config.backend.capabilities = legacyBackendCapabilities(legacy_config.backend);
    addLegacyBackendOptions(config, legacy_config);

    config.storage.filepath = legacy_config.filepath;
    config.storage.filenames = legacy_config.filenames;
    config.storage.num_files = legacy_config.num_files;
    config.storage.enable_direct = legacy_config.storage_enable_direct;

    return config;
}

benchmarkConfig
makeBenchmarkConfigFromRawRequest(const rawRequest &request) {
    benchmarkConfig config;

    config.common.benchmark_group = request.benchmark_group.value;
    config.common.check_consistency = request.check_consistency.value;
    config.common.recreate_xfer = request.recreate_xfer.value;
    config.common.num_iter = request.num_iter.value;
    config.common.large_blk_iter_ftr = request.large_blk_iter_ftr.value;
    config.common.warmup_iter = request.warmup_iter.value;

    config.runtime.type = request.runtime_type.value;
    config.runtime.etcd_endpoints = request.etcd_endpoints.value;

    config.transfer.initiator_seg_type = request.initiator_seg_type.value;
    config.transfer.target_seg_type = request.target_seg_type.value;
    config.transfer.scheme = request.scheme.value;
    config.transfer.mode = request.mode.value;
    config.transfer.op_type = request.op_type.value;
    config.transfer.total_buffer_size = request.total_buffer_size.value;
    config.transfer.start_block_size = request.start_block_size.value;
    config.transfer.max_block_size = request.max_block_size.value;
    config.transfer.start_batch_size = request.start_batch_size.value;
    config.transfer.max_batch_size = request.max_batch_size.value;
    config.transfer.num_threads = request.num_threads.value;

    config.worker.type = request.worker_type.value;
    config.worker.num_initiator_dev = request.num_initiator_dev.value;
    config.worker.num_target_dev = request.num_target_dev.value;
    config.worker.enable_pt = request.enable_pt.value;
    config.worker.progress_threads = request.progress_threads.value;
    config.worker.device_list = request.device_list.value;
    config.worker.enable_vmm = request.enable_vmm.value;

    config.backend.name = request.backend.value;
    config.backend.capabilities = legacyBackendCapabilities(request.backend.value);
    addRawBackendOptions(config, request);

    config.storage.filepath = request.filepath.value;
    config.storage.filenames = request.filenames.value;
    config.storage.num_files = request.num_files.value;
    config.storage.enable_direct = request.storage_enable_direct.value;

    return config;
}

} // namespace nixlbench
