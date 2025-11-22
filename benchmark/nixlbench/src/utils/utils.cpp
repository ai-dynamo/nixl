/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <gflags/gflags.h>
#include <numeric>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <utility>
#include <iomanip>
#include <omp.h>
#if HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include <fcntl.h>
#include <filesystem>

#include "runtime/etcd/etcd_rt.h"
#include "utils/utils.h"

enum xferBenchParamType { XBPT_STRING, XBPT_BOOL, XBPT_UINT64, XBPT_INT32, __XBPT_LAST };

struct xferBenchParamInfo {
    const char *name;
    const char *help;
    xferBenchParamType type;

    union {
        const char *str;
        bool b;
        uint64_t u64;
        int32_t i32;
    } def_value;
};

// Define gflags params
#define NB_ARG_STRING(param_name, def_val, help_text)                               \
    {                                                                               \
        .name = #param_name, .help = help_text, .type = XBPT_STRING, .def_value = { \
            .str = def_val                                                          \
        }                                                                           \
    }
#define NB_ARG_BOOL(param_name, def_val, help_text)                                             \
    {                                                                                           \
        .name = #param_name, .help = help_text, .type = XBPT_BOOL, .def_value = {.b = def_val } \
    }
#define NB_ARG_UINT64(param_name, def_val, help_text)                               \
    {                                                                               \
        .name = #param_name, .help = help_text, .type = XBPT_UINT64, .def_value = { \
            .u64 = def_val                                                          \
        }                                                                           \
    }
#define NB_ARG_INT32(param_name, def_val, help_text)                                               \
    {                                                                                              \
        .name = #param_name, .help = help_text, .type = XBPT_INT32, .def_value = {.i32 = def_val } \
    }

/**********
 * xferBench Config
 **********/
const xferBenchParamInfo xbench_params[] = {
    NB_ARG_STRING(config_file, "", "Config file (Default: None)"),
    NB_ARG_STRING(
        benchmark_group,
        "default",
        "Name of benchmark group. Use different names to run multiple benchmarks in parallel "
        "(Default: default)"),
    NB_ARG_STRING(runtime_type, XFERBENCH_RT_ETCD, "Runtime type to use for communication [ETCD]"),
    NB_ARG_STRING(worker_type, XFERBENCH_WORKER_NIXL, "Type of worker [nixl, nvshmem]"),
    NB_ARG_STRING(
        backend,
        XFERBENCH_BACKEND_UCX,
        "Name of NIXL backend [UCX, GDS, GDS_MT, POSIX, GPUNETIO, Mooncake, HF3FS, OBJ, GUSLI] \
              (only used with nixl worker)"),
    NB_ARG_STRING(initiator_seg_type,
                  XFERBENCH_SEG_TYPE_DRAM,
                  "Type of memory segment for initiator \
              [DRAM, VRAM]. Note: Storage backends always use DRAM locally."),
    NB_ARG_STRING(target_seg_type, XFERBENCH_SEG_TYPE_DRAM, "Type of memory segment for target \
              [DRAM, VRAM]. Note: Storage backends determine remote type automatically."),
    NB_ARG_STRING(scheme, XFERBENCH_SCHEME_PAIRWISE, "Scheme: pairwise, maytoone, onetomany, tp"),
    NB_ARG_STRING(mode,
                  XFERBENCH_MODE_SG,
                  "MODE: SG (Single GPU per proc), MG (Multi GPU per proc) [default: SG]"),
    NB_ARG_STRING(op_type, XFERBENCH_OP_WRITE, "Op type: READ, WRITE"),
    NB_ARG_BOOL(check_consistency, false, "Enable Consistency Check"),
    NB_ARG_UINT64(total_buffer_size, 8LL * 1024 * (1 << 20), "Total buffer \
              size across device for each process (Default: 80 GiB)"),
    NB_ARG_UINT64(start_block_size, 4 * (1 << 10), "Max size of block \
              (Default: 4 KiB)"),
    NB_ARG_UINT64(max_block_size, 64 * (1 << 20), "Max size of block \
              (Default: 64 MiB)"),
    NB_ARG_UINT64(start_batch_size, 1, "Starting size of batch (Default: 1)"),
    NB_ARG_UINT64(max_batch_size, 1, "Max size of batch (starts from 1)"),
    NB_ARG_INT32(num_iter, 1000, "Max iterations"),
    NB_ARG_INT32(large_blk_iter_ftr,
                 16,
                 "factor to reduce test iteration when testing large block size(>1MB)"),
    NB_ARG_INT32(warmup_iter, 100, "Number of warmup iterations before timing"),
    NB_ARG_INT32(
        num_threads,
        1,
        "Number of threads used by benchmark."
        " Num_iter must be greater or equal than num_threads and equally divisible by num_threads."
        " (Default: 1)"),
    NB_ARG_INT32(num_initiator_dev, 1, "Number of device in initiator process"),
    NB_ARG_INT32(num_target_dev, 1, "Number of device in target process"),
    NB_ARG_BOOL(enable_pt, false, "Enable Progress Thread (only used with nixl worker)"),
    NB_ARG_UINT64(progress_threads, 0, "Number of progress threads (default: 0)"),
    NB_ARG_BOOL(enable_vmm, false, "Enable VMM memory allocation when DRAM is requested"),

    // Storage backend(GDS, GDS_MT, POSIX, HF3FS, OBJ) options
    NB_ARG_STRING(filepath, "", "File path for storage operations"),
    NB_ARG_INT32(num_files, 1, "Number of files used by benchmark"),
    NB_ARG_BOOL(storage_enable_direct, false, "Enable direct I/O for storage operations"),

    // GDS options - only used when backend is GDS
    NB_ARG_INT32(gds_batch_pool_size,
                 32,
                 "Batch pool size for GDS operations (default: 32, only used with GDS backend)"),
    NB_ARG_INT32(gds_batch_limit,
                 128,
                 "Batch limit for GDS operations (default: 128, only used with GDS backend)"),
    NB_ARG_INT32(gds_mt_num_threads, 1, "Number of threads used by GDS MT plugin (Default: 1)"),

    // TODO: We should take rank wise device list as input to extend support
    // <rank>:<device_list>, ...
    // For example- 0:mlx5_0,mlx5_1,mlx5_2,1:mlx5_3,mlx5_4, ...
    NB_ARG_STRING(device_list, "all", "Comma-separated device name to use for \
		      communication (only used with nixl worker)"),
    NB_ARG_STRING(etcd_endpoints,
                  "",
                  "ETCD server endpoints for communication (optional for storage backends)"),

    // POSIX options - only used when backend is POSIX
    NB_ARG_STRING(
        posix_api_type,
        XFERBENCH_POSIX_API_AIO,
        "API type for POSIX operations [AIO, URING, POSIXAIO] (only used with POSIX backend)"),

    // DOCA GPUNetIO options - only used when backend is DOCA GPUNetIO
    NB_ARG_STRING(gpunetio_device_list, "0", "Comma-separated GPU CUDA device id to use for \
		      communication (only used with nixl worker)"),
    // DOCA GPUNetIO options - only used when backend is DOCA GPUNetIO
    NB_ARG_STRING(gpunetio_oob_list, "", "Comma-separated OOB network interface name \
		      for control path (only used with nixl worker)"),

    // OBJ options - only used when backend is OBJ
    NB_ARG_STRING(obj_access_key, "", "Access key for S3 backend"),
    NB_ARG_STRING(obj_secret_key, "", "Secret key for S3 backend"),
    NB_ARG_STRING(obj_session_token, "", "Session token for S3 backend"),
    NB_ARG_STRING(obj_bucket_name, XFERBENCH_OBJ_BUCKET_NAME_DEFAULT, "Bucket name for S3 backend"),
    NB_ARG_STRING(obj_scheme,
                  XFERBENCH_OBJ_SCHEME_HTTP,
                  "HTTP scheme for S3 backend [http, https]"),
    NB_ARG_STRING(obj_region, XFERBENCH_OBJ_REGION_EU_CENTRAL_1, "Region for S3 backend"),
    NB_ARG_BOOL(obj_use_virtual_addressing, false, "Use virtual addressing for S3 backend"),
    NB_ARG_STRING(obj_endpoint_override, "", "Endpoint override for S3 backend"),
    NB_ARG_STRING(obj_req_checksum,
                  XFERBENCH_OBJ_REQ_CHECKSUM_SUPPORTED,
                  "Required checksum for S3 backend [supported, required]"),
    NB_ARG_STRING(obj_ca_bundle, "", "Path to CA bundle for S3 backend"),

    // HF3FS options - only used when backend is HF3FS
    NB_ARG_INT32(hf3fs_iopool_size, 64, "Size of io memory pool"),

    // GUSLI options - only used when backend is GUSLI
    NB_ARG_STRING(gusli_client_name, "NIXLBench", "Client name for GUSLI backend"),
    NB_ARG_INT32(gusli_max_simultaneous_requests,
                 32,
                 "Maximum number of simultaneous requests for GUSLI backend"),
    NB_ARG_STRING(
        gusli_config_file,
        "",
        "Configuration file content for GUSLI backend (if empty, auto-generated from device_list)"),
    NB_ARG_UINT64(gusli_bdev_byte_offset,
                  1048576,
                  "Byte offset in block device for GUSLI operations (default: 1MB)"),
    NB_ARG_STRING(
        gusli_device_security,
        "",
        "Comma-separated list of security flags per device (e.g. 'sec=0x3,sec=0x71'). "
        "If empty or fewer than devices, uses 'sec=0x3' as default. "
        "For GUSLI backend, use device_list in format 'id:type:path' where type is F (file) "
        "or K (kernel device)."),
};

#undef NB_ARG_INT32
#undef NB_ARG_UINT64
#undef NB_ARG_BOOL
#undef NB_ARG_STRING

std::string xferBenchConfig::config_file = "";
std::string xferBenchConfig::runtime_type = "";
std::string xferBenchConfig::worker_type = "";
std::string xferBenchConfig::backend = "";
std::string xferBenchConfig::initiator_seg_type = "";
std::string xferBenchConfig::target_seg_type = "";
std::string xferBenchConfig::scheme = "";
std::string xferBenchConfig::mode = "";
std::string xferBenchConfig::op_type = "";
bool xferBenchConfig::check_consistency = false;
size_t xferBenchConfig::total_buffer_size = 0;
int xferBenchConfig::num_initiator_dev = 0;
int xferBenchConfig::num_target_dev = 0;
size_t xferBenchConfig::start_block_size = 0;
size_t xferBenchConfig::max_block_size = 0;
size_t xferBenchConfig::start_batch_size = 0;
size_t xferBenchConfig::max_batch_size = 0;
int xferBenchConfig::num_iter = 0;
int xferBenchConfig::large_blk_iter_ftr = 16;
int xferBenchConfig::warmup_iter = 0;
int xferBenchConfig::num_threads = 0;
bool xferBenchConfig::enable_pt = false;
size_t xferBenchConfig::progress_threads = 0;
bool xferBenchConfig::enable_vmm = false;
std::string xferBenchConfig::device_list = "";
std::string xferBenchConfig::etcd_endpoints = "";
std::string xferBenchConfig::benchmark_group = "default";
int xferBenchConfig::gds_batch_pool_size = 0;
int xferBenchConfig::gds_batch_limit = 0;
int xferBenchConfig::gds_mt_num_threads = 0;
std::string xferBenchConfig::gpunetio_device_list = "";
std::string xferBenchConfig::gpunetio_oob_list = "";
std::vector<std::string> devices = {};
int xferBenchConfig::num_files = 0;
std::string xferBenchConfig::posix_api_type = "";
std::string xferBenchConfig::filepath = "";
bool xferBenchConfig::storage_enable_direct = false;
long xferBenchConfig::page_size = sysconf(_SC_PAGESIZE);
std::string xferBenchConfig::obj_access_key = "";
std::string xferBenchConfig::obj_secret_key = "";
std::string xferBenchConfig::obj_session_token = "";
std::string xferBenchConfig::obj_bucket_name = "";
std::string xferBenchConfig::obj_scheme = "";
std::string xferBenchConfig::obj_region = "";
bool xferBenchConfig::obj_use_virtual_addressing = false;
std::string xferBenchConfig::obj_endpoint_override = "";
std::string xferBenchConfig::obj_req_checksum = "";
std::string xferBenchConfig::obj_ca_bundle = "";
int xferBenchConfig::hf3fs_iopool_size = 0;
std::string xferBenchConfig::gusli_client_name = "";
int xferBenchConfig::gusli_max_simultaneous_requests = 0;
std::string xferBenchConfig::gusli_config_file = "";
uint64_t xferBenchConfig::gusli_bdev_byte_offset = 0;
std::string xferBenchConfig::gusli_device_security = "";

int
xferBenchConfig::parseConfig(int argc, char *argv[]) {
    cxxopts::Options options("nixlbench", "NIXL Benchmark Tool");

    options.add_options()("help", "Print usage");

    for (size_t i = 0; i < sizeof(xbench_params) / sizeof(xbench_params[0]); i++) {
        const xferBenchParamInfo *param = &xbench_params[i];

        switch (param->type) {
        case XBPT_STRING:
            options.add_options()(
                param->name,
                param->help,
                cxxopts::value<std::string>()->default_value(param->def_value.str));
            break;
        case XBPT_BOOL:
            options.add_options()(
                param->name,
                param->help,
                cxxopts::value<bool>()->default_value(param->def_value.b ? "true" : "false"));
            break;
        case XBPT_UINT64:
            options.add_options()(
                param->name,
                param->help,
                cxxopts::value<uint64_t>()->default_value(std::to_string(param->def_value.u64)));
            break;
        case XBPT_INT32:
            options.add_options()(
                param->name,
                param->help,
                cxxopts::value<int32_t>()->default_value(std::to_string(param->def_value.i32)));
            break;
        default:
            std::cerr << param->name << ": unsupported param type: " << param->type << std::endl;
            return -1;
        }
    }

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return -1;
    }

    return loadParams(result);
}

// getParamValue() provides a parameter value, giving priority to explicitly passed
// parameters over those specified in the config_file or set as defaults.
template<class T>
T
xferBenchConfig::getParamValue(std::unique_ptr<inih::INIReader> &ini,
                               cxxopts::ParseResult &result,
                               const char *name) {
    if (ini && !result.count(name)) {
        // config_file exists and the parameter is not specified explicitly ->
        // try to read the value from config_file first
        try {
            return ini->Get<T>("global", name);
        }
        catch (std::runtime_error &) {
            // the parameter is not in the config_file -> fallback to ParseResult
        }
    }

    return result[name].as<T>();
}

int
xferBenchConfig::loadParams(cxxopts::ParseResult &result) {
    std::unique_ptr<inih::INIReader> ini;

    if (result.count("config_file")) {
        /* if config_file parameter specified - try to read the config from file */
        config_file = result["config_file"].as<std::string>();
        try {
            ini = std::make_unique<inih::INIReader>(config_file);
        }
        catch (std::runtime_error &e) {
            std::cerr << "Failed to load ini: " << e.what() << std::endl;
            return -1;
        }
    }

#define NB_ARG(name) getParamValue<decltype(name)>(ini, result, #name)

    benchmark_group = NB_ARG(benchmark_group);
    runtime_type = NB_ARG(runtime_type);
    worker_type = NB_ARG(worker_type);

    // Only load NIXL-specific configurations if using NIXL worker
    if (worker_type == XFERBENCH_WORKER_NIXL) {
        backend = NB_ARG(backend);
        enable_pt = NB_ARG(enable_pt);
        progress_threads = NB_ARG(progress_threads);
        device_list = NB_ARG(device_list);
        enable_vmm = NB_ARG(enable_vmm);

#if defined(HAVE_CUDA) && !defined(HAVE_CUDA_FABRIC)
        if (enable_vmm) {
            std::cerr << "VMM is not supported in CUDA version " << CUDA_VERSION << std::endl;
            return -1;
        }
#endif
        // Load GDS-specific configurations if backend is GDS
        if (backend == XFERBENCH_BACKEND_GDS) {
            gds_batch_pool_size = NB_ARG(gds_batch_pool_size);
            gds_batch_limit = NB_ARG(gds_batch_limit);
        }

        if (backend == XFERBENCH_BACKEND_GDS_MT) {
            gds_mt_num_threads = NB_ARG(gds_mt_num_threads);
        }

        // Load POSIX-specific configurations if backend is POSIX
        if (backend == XFERBENCH_BACKEND_POSIX) {
            posix_api_type = NB_ARG(posix_api_type);

            // Validate POSIX API type
            if (posix_api_type != XFERBENCH_POSIX_API_AIO &&
                posix_api_type != XFERBENCH_POSIX_API_URING &&
                posix_api_type != XFERBENCH_POSIX_API_POSIXAIO) {
                std::cerr << "Invalid POSIX API type: " << posix_api_type
                          << ". Must be one of [AIO, URING, POSIXAIO]" << std::endl;
                return -1;
            }
        }

        // Load DOCA-specific configurations if backend is DOCA
        if (backend == XFERBENCH_BACKEND_GPUNETIO) {
            gpunetio_device_list = NB_ARG(gpunetio_device_list);
            gpunetio_oob_list = NB_ARG(gpunetio_oob_list);
        }

        // Load HD3FS-specific configurations if backend is HD3FS
        if (backend == XFERBENCH_BACKEND_HF3FS) {
            hf3fs_iopool_size = NB_ARG(hf3fs_iopool_size);
        }

        // Load GUSLI-specific configurations if backend is GUSLI
        if (backend == XFERBENCH_BACKEND_GUSLI) {
            gusli_client_name = NB_ARG(gusli_client_name);
            gusli_max_simultaneous_requests = NB_ARG(gusli_max_simultaneous_requests);
            gusli_config_file = NB_ARG(gusli_config_file);
            gusli_bdev_byte_offset = NB_ARG(gusli_bdev_byte_offset);
            gusli_device_security = NB_ARG(gusli_device_security);
        }

        // Load OBJ-specific configurations if backend is OBJ
        if (backend == XFERBENCH_BACKEND_OBJ) {
            obj_access_key = NB_ARG(obj_access_key);
            obj_secret_key = NB_ARG(obj_secret_key);
            obj_session_token = NB_ARG(obj_session_token);
            obj_bucket_name = NB_ARG(obj_bucket_name);
            obj_scheme = NB_ARG(obj_scheme);
            obj_region = NB_ARG(obj_region);
            obj_use_virtual_addressing = NB_ARG(obj_use_virtual_addressing);
            obj_endpoint_override = NB_ARG(obj_endpoint_override);
            obj_req_checksum = NB_ARG(obj_req_checksum);
            obj_ca_bundle = NB_ARG(obj_ca_bundle);

            // Validate OBJ S3 scheme
            if (obj_scheme != XFERBENCH_OBJ_SCHEME_HTTP &&
                obj_scheme != XFERBENCH_OBJ_SCHEME_HTTPS) {
                std::cerr << "Invalid OBJ S3 scheme: " << obj_scheme
                          << ". Must be one of [http, https]" << std::endl;
                return -1;
            }
            // Validate OBJ S3 required checksum
            if (obj_req_checksum != XFERBENCH_OBJ_REQ_CHECKSUM_SUPPORTED &&
                obj_req_checksum != XFERBENCH_OBJ_REQ_CHECKSUM_REQUIRED) {
                std::cerr << "Invalid OBJ S3 required checksum: " << obj_req_checksum
                          << ". Must be one of [supported, required]" << std::endl;
                return -1;
            }
        }
    }

    initiator_seg_type = NB_ARG(initiator_seg_type);
    target_seg_type = NB_ARG(target_seg_type);
    scheme = NB_ARG(scheme);
    mode = NB_ARG(mode);
    op_type = NB_ARG(op_type);
    check_consistency = NB_ARG(check_consistency);
    total_buffer_size = NB_ARG(total_buffer_size);
    num_initiator_dev = NB_ARG(num_initiator_dev);
    num_target_dev = NB_ARG(num_target_dev);
    start_block_size = NB_ARG(start_block_size);
    max_block_size = NB_ARG(max_block_size);
    start_batch_size = NB_ARG(start_batch_size);
    max_batch_size = NB_ARG(max_batch_size);
    num_iter = NB_ARG(num_iter);
    large_blk_iter_ftr = NB_ARG(large_blk_iter_ftr);
    warmup_iter = NB_ARG(warmup_iter);
    num_threads = NB_ARG(num_threads);
    etcd_endpoints = NB_ARG(etcd_endpoints);
    filepath = NB_ARG(filepath);
    num_files = NB_ARG(num_files);
    posix_api_type = NB_ARG(posix_api_type);
    storage_enable_direct = NB_ARG(storage_enable_direct);

    // Validate ETCD configuration
    if (!isStorageBackend() && etcd_endpoints.empty()) {
        // For non-storage backends, set default ETCD endpoint
        etcd_endpoints = "http://localhost:2379";
        std::cout << "Using default ETCD endpoint for non-storage backend: " << etcd_endpoints
                  << std::endl;
    }

    if (worker_type == XFERBENCH_WORKER_NVSHMEM) {
        if (!((XFERBENCH_SEG_TYPE_VRAM == initiator_seg_type) &&
              (XFERBENCH_SEG_TYPE_VRAM == target_seg_type) && (1 == num_threads) &&
              (1 == num_initiator_dev) && (1 == num_target_dev) &&
              (XFERBENCH_SCHEME_PAIRWISE == scheme))) {
            std::cerr << "Unsupported configuration for NVSHMEM worker" << std::endl;
            std::cerr << "Supported configuration: " << std::endl;
            std::cerr << std::string(20, '*') << std::endl;
            std::cerr << "initiator_seg_type = VRAM" << std::endl;
            std::cerr << "target_seg_type = VRAM" << std::endl;
            std::cerr << "num_threads = 1" << std::endl;
            std::cerr << "num_initiator_dev = 1" << std::endl;
            std::cerr << "num_target_dev = 1" << std::endl;
            std::cerr << "scheme = pairwise" << std::endl;
            std::cerr << std::string(20, '*') << std::endl;
            return -1;
        }
    }

    if ((max_block_size * max_batch_size) > (total_buffer_size / num_initiator_dev)) {
        std::cerr
            << "Incorrect buffer size configuration for Initiator"
            << "(max_block_size * max_batch_size) is > (total_buffer_size / num_initiator_dev)"
            << std::endl;
        return -1;
    }
    if ((max_block_size * max_batch_size) > (total_buffer_size / num_target_dev)) {
        std::cerr
            << "Incorrect buffer size configuration for Target"
            << "(max_block_size * max_batch_size) is > (total_buffer_size / num_initiator_dev)"
            << std::endl;
        return -1;
    }
    if ((max_block_size * max_batch_size) > (total_buffer_size / num_threads)) {
        std::cerr << "Incorrect buffer size configuration "
                  << "(max_block_size * max_batch_size) "
                  << "(" << (max_block_size * max_batch_size) << ")"
                  << " is > (total_buffer_size / num_threads) ("
                  << (total_buffer_size / num_threads) << ")" << std::endl;
        return -1;
    }

    if (large_blk_iter_ftr <= 0) {
        std::cerr << "iter_factor must be greater than 0" << std::endl;
        return -1;
    }

    int partition = (num_threads * large_blk_iter_ftr);
    if (num_iter % partition) {
        num_iter += partition - (num_iter % partition);
        std::cout << "WARNING: Adjusting num_iter to " << num_iter
                  << " to allow equal distribution to " << num_threads << " threads" << std::endl;
    }
    if (warmup_iter % partition) {
        warmup_iter += partition - (warmup_iter % partition);
        std::cout << "WARNING: Adjusting warmup_iter to " << warmup_iter
                  << " to allow equal distribution to " << num_threads << " threads" << std::endl;
    }
    partition = (num_initiator_dev * num_threads);
    if (total_buffer_size % partition) {
        std::cerr << "Total_buffer_size must be divisible by the product of num_threads and "
                     "num_initiator_dev"
                  << ", next such value is "
                  << total_buffer_size + partition - (total_buffer_size % partition) << std::endl;
        return -1;
    }
    partition = (num_target_dev * num_threads);
    if (total_buffer_size % partition) {
        std::cerr << "Total_buffer_size must be divisible by the product of num_threads and "
                     "num_target_dev"
                  << ", next such value is "
                  << total_buffer_size + partition - (total_buffer_size % partition) << std::endl;
        return -1;
    }

    return 0;
#undef NB_ARG
}

void
xferBenchConfig::printOption(const std::string &desc, const std::string &value) {
    std::cout << std::left << std::setw(60) << desc << ": " << value << std::endl;
}

void
xferBenchConfig::printSeparator(const char sep) {
    std::cout << std::string(160, sep) << std::endl;
}

void
xferBenchConfig::printConfig() {
    printSeparator('*');
    std::cout << "NIXLBench Configuration" << std::endl;
    printSeparator('*');
    printOption("Runtime (--runtime_type=[etcd])", runtime_type);
    if (runtime_type == XFERBENCH_RT_ETCD) {
        if (etcd_endpoints.empty()) {
            printOption("ETCD Endpoint ", "disabled (storage backend)");
        } else {
            printOption("ETCD Endpoint ", etcd_endpoints);
        }
    }
    printOption("Worker type (--worker_type=[nixl,nvshmem])", worker_type);
    if (worker_type == XFERBENCH_WORKER_NIXL) {
        printOption("Backend (--backend=[UCX,GDS,GDS_MT,POSIX,Mooncake,HF3FS,OBJ])", backend);
        printOption("Enable pt (--enable_pt=[0,1])", std::to_string(enable_pt));
        printOption("Progress threads (--progress_threads=N)", std::to_string(progress_threads));
        printOption("Device list (--device_list=dev1,dev2,...)", device_list);
        printOption("Enable VMM (--enable_vmm=[0,1])", std::to_string(enable_vmm));

        // Print GDS options if backend is GDS
        if (backend == XFERBENCH_BACKEND_GDS) {
            printOption("GDS batch pool size (--gds_batch_pool_size=N)",
                        std::to_string(gds_batch_pool_size));
            printOption("GDS batch limit (--gds_batch_limit=N)", std::to_string(gds_batch_limit));
        }

        if (backend == XFERBENCH_BACKEND_GDS_MT) {
            printOption("GDS MT Number of threads (--gds_mt_num_threads=N)",
                        std::to_string(gds_mt_num_threads));
        }

        // Print POSIX options if backend is POSIX
        if (backend == XFERBENCH_BACKEND_POSIX) {
            printOption("POSIX API type (--posix_api_type=[AIO,URING,POSIXAIO])", posix_api_type);
        }

        // Print OBJ options if backend is OBJ
        if (backend == XFERBENCH_BACKEND_OBJ) {
            printOption("OBJ S3 access key (--obj_access_key=key)", obj_access_key);
            printOption("OBJ S3 secret key (--obj_secret_key=key)", obj_secret_key);
            printOption("OBJ S3 session token (--obj_session_token=token)", obj_session_token);
            printOption("OBJ S3 bucket name (--obj_bucket_name=nixlbench-bucket)", obj_bucket_name);
            printOption("OBJ S3 scheme (--obj_scheme=[http, https])", obj_scheme);
            printOption("OBJ S3 region (--obj_region=region)", obj_region);
            printOption("OBJ S3 use virtual addressing (--obj_use_virtual_addressing=[0,1])",
                        std::to_string(obj_use_virtual_addressing));
            printOption("OBJ S3 endpoint override (--obj_endpoint_override=endpoint)",
                        obj_endpoint_override);
            printOption("OBJ S3 required checksum (--obj_req_checksum=[supported, required])",
                        obj_req_checksum);
            printOption("OBJ S3 CA bundle (--obj_ca_bundle=cert-path)", obj_ca_bundle);
        }

        if (xferBenchConfig::isStorageBackend()) {
            printOption("filepath (--filepath=path)", filepath);
            printOption("Number of files (--num_files=N)", std::to_string(num_files));
            printOption("Storage enable direct (--storage_enable_direct=[0,1])",
                        std::to_string(storage_enable_direct));
        }

        // Print DOCA GPUNetIO options if backend is DOCA GPUNetIO
        if (backend == XFERBENCH_BACKEND_GPUNETIO) {
            printOption("GPU CUDA Device id list (--device_list=dev1,dev2,...)",
                        gpunetio_device_list);
            printOption("OOB network interface name for control path (--oob_list=ifface)",
                        gpunetio_oob_list);
        }
    }
    printOption("Initiator seg type (--initiator_seg_type=[DRAM,VRAM])", initiator_seg_type);
    printOption("Target seg type (--target_seg_type=[DRAM,VRAM])", target_seg_type);
    printOption("Scheme (--scheme=[pairwise,manytoone,onetomany,tp])", scheme);
    printOption("Mode (--mode=[SG,MG])", mode);
    printOption("Op type (--op_type=[READ,WRITE])", op_type);
    printOption("Check consistency (--check_consistency=[0,1])", std::to_string(check_consistency));
    printOption("Total buffer size (--total_buffer_size=N)", std::to_string(total_buffer_size));
    printOption("Num initiator dev (--num_initiator_dev=N)", std::to_string(num_initiator_dev));
    printOption("Num target dev (--num_target_dev=N)", std::to_string(num_target_dev));
    printOption("Start block size (--start_block_size=N)", std::to_string(start_block_size));
    printOption("Max block size (--max_block_size=N)", std::to_string(max_block_size));
    printOption("Start batch size (--start_batch_size=N)", std::to_string(start_batch_size));
    printOption("Max batch size (--max_batch_size=N)", std::to_string(max_batch_size));
    printOption("Num iter (--num_iter=N)", std::to_string(num_iter));
    printOption("Warmup iter (--warmup_iter=N)", std::to_string(warmup_iter));
    printOption("Large block iter factor (--large_blk_iter_ftr=N)",
                std::to_string(large_blk_iter_ftr));
    printOption("Num threads (--num_threads=N)", std::to_string(num_threads));
    printSeparator('-');
    std::cout << std::endl;
}

std::vector<std::string>
xferBenchConfig::parseDeviceList() {
    std::vector<std::string> devices;
    std::string dev;
    std::stringstream ss(xferBenchConfig::device_list);

    // TODO: Add support for other schemes
    if (xferBenchConfig::scheme == XFERBENCH_SCHEME_PAIRWISE &&
        xferBenchConfig::device_list != "all") {
        while (std::getline(ss, dev, ',')) {
            devices.push_back(dev);
        }

        if ((int)devices.size() != xferBenchConfig::num_initiator_dev ||
            (int)devices.size() != xferBenchConfig::num_target_dev) {
            std::cerr << "Incorrect device list " << xferBenchConfig::device_list
                      << " provided for pairwise scheme " << devices.size() << "# devices"
                      << std::endl;
            return {};
        }
    } else {
        devices.push_back("all");
    }

    return devices;
}

bool
xferBenchConfig::isStorageBackend() {
    return (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend ||
            XFERBENCH_BACKEND_GDS_MT == xferBenchConfig::backend ||
            XFERBENCH_BACKEND_HF3FS == xferBenchConfig::backend ||
            XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend ||
            XFERBENCH_BACKEND_OBJ == xferBenchConfig::backend ||
            XFERBENCH_BACKEND_GUSLI == xferBenchConfig::backend);
}

/**********
 * xferBench Utils
 **********/
xferBenchRT *xferBenchUtils::rt = nullptr;
std::string xferBenchUtils::dev_to_use = "";

void
xferBenchUtils::setRT(xferBenchRT *rt) {
    xferBenchUtils::rt = rt;
}

void
xferBenchUtils::setDevToUse(std::string dev) {
    dev_to_use = dev;
}

std::string
xferBenchUtils::getDevToUse() {
    return dev_to_use;
}

static bool
allBytesAre(void *buffer, size_t size, uint8_t value) {
    uint8_t *byte_buffer = static_cast<uint8_t *>(buffer);

    // Iterate over each byte in the buffer
    for (size_t i = 0; i < size; ++i) {
        if (byte_buffer[i] != value) {
            return false; // Return false if any byte doesn't match the value
        }
    }
    return true; // All bytes match the value
}

// Implement GUSLI device parser (declared in utils.h) so it can be reused by both utils and worker
std::vector<GusliDeviceConfig>
parseGusliDeviceList(const std::string &device_list,
                     const std::string &security_list,
                     int num_devices) {
    std::vector<GusliDeviceConfig> devices;

    // Parse security flags
    std::vector<std::string> security_flags;
    if (!security_list.empty()) {
        std::stringstream sec_ss(security_list);
        std::string sec_flag;
        while (std::getline(sec_ss, sec_flag, ',')) {
            security_flags.push_back(sec_flag);
        }
    }

    // For GUSLI, device_list cannot be "all" - must specify devices explicitly
    if (device_list.empty() || device_list == "all") {
        std::cerr << "Error: GUSLI backend requires explicit device_list in format 'id:type:path'"
                  << std::endl;
        std::cerr << "Example: --device_list='11:F:./store0.bin,14:K:/dev/zero,20:N:t192.168.1.100'"
                  << std::endl;
        std::cerr << "  id: Device identifier (numeric)" << std::endl;
        std::cerr << "  type: F (file), K (kernel block device), or N (networked server)"
                  << std::endl;
        std::cerr << "  path: Device path or server address (for N type, prefix with 't' for TCP "
                     "or 'u' for UDP)"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    std::stringstream ss(device_list);
    std::string device_spec;
    size_t device_count = 0;

    while (std::getline(ss, device_spec, ',')) {
        std::stringstream dev_ss(device_spec);
        std::string id_str, type_str, path;
        if (std::getline(dev_ss, id_str, ':') && std::getline(dev_ss, type_str, ':') &&
            std::getline(dev_ss, path)) {
            int device_id = std::stoi(id_str);
            char device_type = type_str[0];
            if (device_type != 'F' && device_type != 'K' && device_type != 'N') {
                std::cerr << "Invalid GUSLI device type: " << device_type
                          << ". Must be 'F' (file), 'K' (kernel device), or 'N' (networked server)"
                          << std::endl;
                exit(EXIT_FAILURE);
            }
            std::string sec_flag =
                (device_count < security_flags.size()) ? security_flags[device_count] : "sec=0x3";
            devices.push_back({device_id, device_type, path, sec_flag});
            device_count++;
        } else {
            std::cerr << "Invalid GUSLI device specification: " << device_spec
                      << ". Expected format: 'id:type:path'" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if (!security_flags.empty() && security_flags.size() != devices.size()) {
        std::cerr << "Warning: Number of security flags (" << security_flags.size()
                  << ") doesn't match number of devices (" << devices.size()
                  << "). Using 'sec=0x3' for missing entries." << std::endl;
    }

    if (num_devices > 0 && devices.size() != static_cast<size_t>(num_devices)) {
        std::cerr << "Error: Number of devices in device_list (" << devices.size()
                  << ") must match num_devices (" << num_devices << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    return devices;
}

void
xferBenchUtils::checkConsistency(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    int i = 0, j = 0;
    static bool gusli_devmap_init = false;
    static std::vector<GusliDeviceConfig> gusli_devs;
    if (!gusli_devmap_init && xferBenchConfig::backend == XFERBENCH_BACKEND_GUSLI) {
        gusli_devs = parseGusliDeviceList(xferBenchConfig::device_list,
                                          xferBenchConfig::gusli_device_security,
                                          xferBenchConfig::num_initiator_dev);
        gusli_devmap_init = true;
    }
    bool pass_check_consistency = true;
    for (const auto &iov_list : iov_lists) {
        for (const auto &iov : iov_list) {
            void *addr = NULL;
            size_t len;
            uint8_t check_val = 0x00;
            bool rc = false;
            bool is_allocated = false;

            len = iov.len;

            if (xferBenchConfig::isStorageBackend() ||
                xferBenchConfig::backend == XFERBENCH_BACKEND_GPUNETIO) {
                if (xferBenchConfig::op_type == XFERBENCH_OP_READ) {
                    if (xferBenchConfig::initiator_seg_type == XFERBENCH_SEG_TYPE_VRAM) {
#if HAVE_CUDA
                        addr = calloc(1, len);
                        is_allocated = true;
                        CHECK_CUDA_ERROR(
                            cudaMemcpy(addr, (void *)iov.addr, len, cudaMemcpyDeviceToHost),
                            "cudaMemcpy failed");
#else
                        std::cerr << "Failure in consistency check: VRAM segment type not "
                                     "supported without CUDA"
                                  << std::endl;
                        exit(EXIT_FAILURE);
#endif
                    } else {
                        addr = (void *)iov.addr;
                    }
                } else if (xferBenchConfig::op_type == XFERBENCH_OP_WRITE) {
                    // Allocate buffer (handle alignment for direct I/O if requested)
                    if (xferBenchConfig::storage_enable_direct) {
                        void *aligned = nullptr;
                        if (posix_memalign(&aligned, xferBenchConfig::page_size, len) != 0) {
                            std::cerr << "Failed to allocate aligned buffer of size: " << len
                                      << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        addr = aligned;
                    } else {
                        addr = calloc(1, len);
                    }
                    is_allocated = true;
                    if (xferBenchConfig::backend == XFERBENCH_BACKEND_OBJ) {
                        if (!getObjS3(iov.metaInfo)) {
                            std::cerr << "Failed to get S3 object: " << iov.metaInfo << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        int fd = open(iov.metaInfo.c_str(), O_RDONLY);
                        if (fd < 0) {
                            std::cerr << "Failed to open downloaded file: " << iov.metaInfo
                                      << " with error: " << strerror(errno) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        ssize_t rc = pread(fd, addr, len, 0);
                        if (rc < 0) {
                            std::cerr << "Failed to read from file: " << iov.metaInfo
                                      << " with error: " << strerror(errno) << std::endl;
                        }
                        close(fd);
                        unlink(iov.metaInfo.c_str());
                    } else if (xferBenchConfig::backend == XFERBENCH_BACKEND_GUSLI) {
                        // Map device id -> path via device_list and read from the bdev at LBA
                        // offset
                        auto it = std::find_if(
                            gusli_devs.begin(), gusli_devs.end(), [&](const GusliDeviceConfig &m) {
                                return m.device_id == iov.devId;
                            });
                        if (it == gusli_devs.end()) {
                            std::cerr << "Failed to locate GUSLI device id " << iov.devId
                                      << " in device_list. Cannot validate." << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        if (it->device_type != 'F' && it->device_type != 'K') {
                            std::cerr << "GUSLI device type '" << it->device_type
                                      << "' not supported for consistency validation" << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        int oflags = O_RDONLY;
                        if (xferBenchConfig::storage_enable_direct) oflags |= O_DIRECT;
                        int fd = open(it->device_path.c_str(), oflags);
                        if (fd < 0) {
                            std::cerr << "Failed to open GUSLI device path: " << it->device_path
                                      << " with error: " << strerror(errno) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                        ssize_t rc = pread(fd, addr, len, iov.addr);
                        if (rc < 0) {
                            std::cerr << "Failed to read from GUSLI device: " << it->device_path
                                      << " with error: " << strerror(errno) << std::endl;
                            close(fd);
                            exit(EXIT_FAILURE);
                        }
                        close(fd);
                    } else {
                        ssize_t rc = pread(iov.devId, addr, len, iov.addr);
                        if (rc < 0) {
                            std::cerr << "Failed to read from device: " << iov.devId
                                      << " with error: " << strerror(errno) << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    }
                }
            } else {
                // This will be called on target process in case of write and
                // on initiator process in case of read
                if ((xferBenchConfig::op_type == XFERBENCH_OP_WRITE &&
                     xferBenchConfig::target_seg_type == XFERBENCH_SEG_TYPE_VRAM) ||
                    (xferBenchConfig::op_type == XFERBENCH_OP_READ &&
                     xferBenchConfig::initiator_seg_type == XFERBENCH_SEG_TYPE_VRAM)) {
#if HAVE_CUDA
                    addr = calloc(1, len);
                    is_allocated = true;
                    CHECK_CUDA_ERROR(
                        cudaMemcpy(addr, (void *)iov.addr, len, cudaMemcpyDeviceToHost),
                        "cudaMemcpy failed");
#else
                    std::cerr << "Failure in consistency check: VRAM segment type not supported "
                                 "without CUDA"
                              << std::endl;
                    exit(EXIT_FAILURE);
#endif
                } else if ((xferBenchConfig::op_type == XFERBENCH_OP_WRITE &&
                            xferBenchConfig::target_seg_type == XFERBENCH_SEG_TYPE_DRAM) ||
                           (xferBenchConfig::op_type == XFERBENCH_OP_READ &&
                            xferBenchConfig::initiator_seg_type == XFERBENCH_SEG_TYPE_DRAM)) {
                    addr = (void *)iov.addr;
                }
            }

            if ("WRITE" == xferBenchConfig::op_type) {
                check_val = XFERBENCH_INITIATOR_BUFFER_ELEMENT;
            } else if ("READ" == xferBenchConfig::op_type) {
                check_val = XFERBENCH_TARGET_BUFFER_ELEMENT;
            }
            rc = allBytesAre(addr, len, check_val);
            if (true != rc) {
                std::cerr << "Consistency check failed for iov " << i << ":" << j << std::endl;
                pass_check_consistency = false;
            }
            // Free the addr only if is allocated here
            if (is_allocated) {
                free(addr);
            }
            j++;
        }
        i++;
    }
    if (!pass_check_consistency) {
        std::cerr << "Consistency check failed" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void
xferBenchUtils::printStatsHeader() {
    if (IS_PAIRWISE_AND_SG() && rt->getSize() > 2) {
        // clang-format off
        std::cout << std::left
                  << std::setw(20) << "Block Size (B)"
                  << std::setw(15) << "Batch Size"
                  << std::setw(15) << "B/W (GB/Sec)"
                  << std::setw(25) << "Aggregate B/W (GB/Sec)"
                  << std::setw(20) << "Network Util (%)"
                  << std::setw(15) << "Avg Lat. (us)"
                  << std::setw(15) << "Avg Prep (us)"
                  << std::setw(15) << "P99 Prep (us)"
                  << std::setw(15) << "Avg Post (us)"
                  << std::setw(15) << "P99 Post (us)"
                  << std::setw(15) << "Avg Tx (us)"
                  << std::setw(15) << "P99 Tx (us)"
                  << std::endl;
        // clang-format on
    } else {
        // clang-format off
        std::cout << std::left
                  << std::setw(20) << "Block Size (B)"
                  << std::setw(15) << "Batch Size"
                  << std::setw(15) << "B/W (GB/Sec)"
                  << std::setw(15) << "Avg Lat. (us)"
                  << std::setw(15) << "Avg Prep (us)"
                  << std::setw(15) << "P99 Prep (us)"
                  << std::setw(15) << "Avg Post (us)"
                  << std::setw(15) << "P99 Post (us)"
                  << std::setw(15) << "Avg Tx (us)"
                  << std::setw(15) << "P99 Tx (us)"
                  << std::endl;
        // clang-format on
    }
    xferBenchConfig::printSeparator('-');
}

void
xferBenchUtils::printStats(bool is_target,
                           size_t block_size,
                           size_t batch_size,
                           xferBenchStats stats) {
    size_t total_data_transferred = 0;
    double avg_latency = 0, throughput_gb = 0;
    double totalbw = 0;

    int num_iter = xferBenchConfig::num_iter;

    if (block_size > LARGE_BLOCK_SIZE) {
        num_iter /= xferBenchConfig::large_blk_iter_ftr;
    }

    // TODO: We can avoid this by creating a sub-communicator across initiator ranks
    // if (isTarget() && IS_PAIRWISE_AND_SG() && rt->getSize() > 2) { - Fix this isTarget can not be
    // called here
    if (is_target && IS_PAIRWISE_AND_SG() && rt->getSize() > 2) {
        rt->reduceSumDouble(&throughput_gb, &totalbw, 0);
        return;
    }

    double total_duration = stats.total_duration.avg();

    total_data_transferred = ((block_size * batch_size) * num_iter); // In Bytes
    avg_latency = (total_duration / (num_iter * batch_size)); // In microsec
    if (IS_PAIRWISE_AND_MG()) {
        total_data_transferred *= xferBenchConfig::num_initiator_dev; // In Bytes
        avg_latency /= xferBenchConfig::num_initiator_dev; // In microsec
    }

    throughput_gb = (((double)total_data_transferred / (1000 * 1000 * 1000)) /
                     (total_duration / 1e6)); // In GB/Sec

    if (IS_PAIRWISE_AND_SG() && rt->getSize() > 2) {
        rt->reduceSumDouble(&throughput_gb, &totalbw, 0);
    } else {
        totalbw = throughput_gb;
    }

    if (IS_PAIRWISE_AND_SG() && rt->getRank() != 0) {
        return;
    }

    double prepare_duration = stats.prepare_duration.avg();
    double prepare_p99_duration = stats.prepare_duration.p99();
    double post_duration = stats.post_duration.avg();
    double post_p99_duration = stats.post_duration.p99();
    double transfer_duration = stats.transfer_duration.avg();
    double transfer_p99_duration = stats.transfer_duration.p99();

    // Tabulate print with fixed width for each string
    if (IS_PAIRWISE_AND_SG() && rt->getSize() > 2) {
        // clang-format off
        std::cout << std::left << std::fixed << std::setprecision(6)
                  << std::setw(20) << block_size
                  << std::setw(15) << batch_size
                  << std::setw(15) << throughput_gb
                  << std::setw(25) << totalbw
                  << std::setw(20) << (totalbw / (rt->getSize() / 2 * MAXBW)) * 100
                  << std::setprecision(1)
                  << std::setw(15) << avg_latency
                  << std::setw(15) << prepare_duration
                  << std::setw(15) << prepare_p99_duration
                  << std::setw(15) << post_duration
                  << std::setw(15) << post_p99_duration
                  << std::setw(15) << transfer_duration
                  << std::setw(15) << transfer_p99_duration
                  << std::endl;
        // clang-format on
    } else {
        // clang-format off
        std::cout << std::left << std::fixed << std::setprecision(6)
                  << std::setw(20) << block_size
                  << std::setw(15) << batch_size
                  << std::setw(15) << throughput_gb
                  << std::setprecision(1)
                  << std::setw(15) << avg_latency
                  << std::setw(15) << prepare_duration
                  << std::setw(15) << prepare_p99_duration
                  << std::setw(15) << post_duration
                  << std::setw(15) << post_p99_duration
                  << std::setw(15) << transfer_duration
                  << std::setw(15) << transfer_p99_duration
                  << std::endl;
        // clang-format on
    }
}

std::string
xferBenchUtils::buildAwsCredentials() {
    std::string env_setup = "";

    if (!xferBenchConfig::obj_access_key.empty()) {
        env_setup += "AWS_ACCESS_KEY_ID=" + xferBenchConfig::obj_access_key + " ";
    }
    if (!xferBenchConfig::obj_secret_key.empty()) {
        env_setup += "AWS_SECRET_ACCESS_KEY=" + xferBenchConfig::obj_secret_key + " ";
    }
    if (!xferBenchConfig::obj_session_token.empty()) {
        env_setup += "AWS_SESSION_TOKEN=" + xferBenchConfig::obj_session_token + " ";
    }
    if (!xferBenchConfig::obj_region.empty()) {
        env_setup += "AWS_DEFAULT_REGION=" + xferBenchConfig::obj_region + " ";
    }

    return env_setup;
}

bool
xferBenchUtils::putObjS3(size_t buffer_size, const std::string &name) {
    std::string filename = "/tmp/" + name;
    int fd = open(filename.c_str(), O_RDWR | O_CREAT, 0744);
    if (fd < 0) {
        std::cerr << "Failed to open file: " << name << " with error: " << strerror(errno)
                  << std::endl;
        return false;
    }
    // Create buffer filled with XFERBENCH_TARGET_BUFFER_ELEMENT
    void *buf = (void *)malloc(buffer_size);
    if (!buf) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of memory" << std::endl;
        close(fd);
        return false;
    }
    memset(buf, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
    int rc = pwrite(fd, buf, buffer_size, 0);
    if (rc < 0) {
        std::cerr << "Failed to write to file: " << fd << " with error: " << strerror(errno)
                  << std::endl;
        free(buf);
        close(fd);
        return false;
    }
    free(buf);

    std::string bucket_name = xferBenchConfig::obj_bucket_name;
    if (bucket_name.empty()) {
        std::cerr << "Error: Invalid bucket name for S3 object put" << std::endl;
        close(fd);
        unlink(filename.c_str());
        return false;
    }
    std::string aws_cmd = "aws s3 cp " + filename + " s3://" + bucket_name;
    if (!xferBenchConfig::obj_endpoint_override.empty()) {
        aws_cmd += " --endpoint-url " + xferBenchConfig::obj_endpoint_override;
    }

    std::string full_cmd = buildAwsCredentials() + aws_cmd;
    std::cout << "Putting S3 object: " << name << " in bucket: " << bucket_name
              << " (size: " << buffer_size << " bytes)" << std::endl;

    int result = system(full_cmd.c_str());
    if (result != 0) {
        std::cerr << "Failed to put S3 object " << name << " in bucket " << bucket_name
                  << " (exit code: " << result << ")" << std::endl;
        close(fd);
        unlink(filename.c_str());
        return false;
    }

    close(fd);
    unlink(filename.c_str());
    return true;
}

bool
xferBenchUtils::getObjS3(const std::string &name) {
    std::string bucket_name = xferBenchConfig::obj_bucket_name;
    if (bucket_name.empty()) {
        std::cerr << "Error: Invalid bucket name for S3 object get" << std::endl;
        return false;
    }
    std::string aws_cmd = "aws s3 cp s3://" + bucket_name + "/" + name + " " + name;
    if (!xferBenchConfig::obj_endpoint_override.empty()) {
        aws_cmd += " --endpoint-url " + xferBenchConfig::obj_endpoint_override;
    }

    std::string full_cmd = buildAwsCredentials() + aws_cmd;
    std::cout << "Getting S3 object: " << name << " from bucket: " << bucket_name << std::endl;

    int result = system(full_cmd.c_str());
    if (result != 0) {
        std::cerr << "Failed to get S3 object " << name << " from bucket " << bucket_name
                  << " (exit code: " << result << ")" << std::endl;
        return false;
    }

    return true;
}

bool
xferBenchUtils::rmObjS3(const std::string &name) {
    std::string bucket_name = xferBenchConfig::obj_bucket_name;
    if (bucket_name.empty()) {
        std::cerr << "Error: Invalid bucket name for S3 object get" << std::endl;
        return false;
    }

    std::string aws_cmd = "aws s3 rm s3://" + bucket_name + "/" + name;
    if (!xferBenchConfig::obj_endpoint_override.empty()) {
        aws_cmd += " --endpoint-url " + xferBenchConfig::obj_endpoint_override;
    }

    std::string full_cmd = buildAwsCredentials() + aws_cmd;
    std::cout << "Removing S3 object: " << name << " from bucket: " << bucket_name << std::endl;

    int result = system(full_cmd.c_str());
    if (result != 0) {
        std::cerr << "Warning: Failed to remove S3 object " << name << " from bucket "
                  << bucket_name << " (exit code: " << result << ")" << std::endl;
        return false;
    }
    return true;
}

/*
 * xferMetricStats
 */

double
xferMetricStats::min() const {
    if (samples.empty()) return 0;
    return *std::min_element(samples.begin(), samples.end());
}

double
xferMetricStats::max() const {
    if (samples.empty()) return 0;
    return *std::max_element(samples.begin(), samples.end());
}

double
xferMetricStats::avg() const {
    if (samples.empty()) return 0;
    return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
}

double
xferMetricStats::p90() {
    if (samples.empty()) return 0;
    std::sort(samples.begin(), samples.end());
    size_t index = samples.size() * 0.9;
    return samples[std::min(index, samples.size() - 1)];
}

double
xferMetricStats::p95() {
    if (samples.empty()) return 0;
    std::sort(samples.begin(), samples.end());
    size_t index = samples.size() * 0.95;
    return samples[std::min(index, samples.size() - 1)];
}

double
xferMetricStats::p99() {
    if (samples.empty()) return 0;
    std::sort(samples.begin(), samples.end());
    size_t index = samples.size() * 0.99;
    return samples[std::min(index, samples.size() - 1)];
}

void
xferMetricStats::add(double value) {
    samples.push_back(value);
}

void
xferMetricStats::add(const xferMetricStats &other) {
    samples.insert(samples.end(), other.samples.begin(), other.samples.end());
}

void
xferMetricStats::reserve(size_t n) {
    samples.reserve(n);
}

void
xferMetricStats::clear() {
    samples.clear();
}

/*
 * xferBenchStats
 */

void
xferBenchStats::clear() {
    total_duration.clear();
    prepare_duration.clear();
    post_duration.clear();
    transfer_duration.clear();
}

void
xferBenchStats::add(const xferBenchStats &other) {
    total_duration.add(other.total_duration);
    prepare_duration.add(other.prepare_duration);
    post_duration.add(other.post_duration);
    transfer_duration.add(other.transfer_duration);
}

void
xferBenchStats::reserve(size_t n) {
    total_duration.reserve(n);
    prepare_duration.reserve(n);
    post_duration.reserve(n);
    transfer_duration.reserve(n);
}

/*
 * xferBenchTimer
 */

xferBenchTimer::xferBenchTimer() : start_(nixlTime::getUs()) {}

nixlTime::us_t
xferBenchTimer::lap() {
    nixlTime::us_t now = nixlTime::getUs();
    nixlTime::us_t duration = now - start_;
    start_ = now;
    return duration;
}
