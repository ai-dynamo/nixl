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

#include <cstring>
#include <gflags/gflags.h>
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

/**********
 * xferBench Config
 **********/
DEFINE_string(benchmark_group,
              "default",
              "Name of benchmark group. Use different names to run multiple benchmarks in parallel "
              "(Default: default)");
DEFINE_string(runtime_type, XFERBENCH_RT_ETCD, "Runtime type to use for communication [ETCD]");
DEFINE_string(worker_type, XFERBENCH_WORKER_NIXL, "Type of worker [nixl, nvshmem]");
DEFINE_string(backend,
              XFERBENCH_BACKEND_UCX,
              "Name of communication backend [UCX, UCX_MO, GDS, POSIX, GPUNETIO, OBJ] \
              (only used with nixl worker)");
DEFINE_string(initiator_seg_type, XFERBENCH_SEG_TYPE_DRAM, "Type of memory segment for initiator \
              [DRAM, VRAM]");
DEFINE_string(target_seg_type, XFERBENCH_SEG_TYPE_DRAM, "Type of memory segment for target \
              [DRAM, VRAM]");
DEFINE_string(scheme, XFERBENCH_SCHEME_PAIRWISE, "Scheme: pairwise, maytoone, onetomany, tp");
DEFINE_string(mode, XFERBENCH_MODE_SG, "MODE: SG (Single GPU per proc), MG (Multi GPU per proc) [default: SG]");
DEFINE_string(op_type, XFERBENCH_OP_WRITE, "Op type: READ, WRITE");
DEFINE_bool(check_consistency, false, "Enable Consistency Check");
DEFINE_uint64(total_buffer_size, 8LL * 1024 * (1 << 20), "Total buffer \
              size across device for each process (Default: 80 GiB)");
DEFINE_uint64(start_block_size, 4 * (1 << 10), "Max size of block \
              (Default: 4 KiB)");
DEFINE_uint64(max_block_size, 64 * (1 << 20), "Max size of block \
              (Default: 64 MiB)");
DEFINE_uint64(start_batch_size, 1, "Starting size of batch (Default: 1)");
DEFINE_uint64(max_batch_size, 1, "Max size of batch (starts from 1)");
DEFINE_int32(num_iter, 1000, "Max iterations");
DEFINE_int32(large_blk_iter_ftr,
             16,
             "factor to reduce test iteration when testing large block size(>1MB)");
DEFINE_int32(warmup_iter, 100, "Number of warmup iterations before timing");
DEFINE_int32 (
    num_threads,
    1,
    "Number of threads used by benchmark."
    " Num_iter must be greater or equal than num_threads and equally divisible by num_threads."
    " (Default: 1)");
DEFINE_int32(num_initiator_dev, 1, "Number of device in initiator process");
DEFINE_int32(num_target_dev, 1, "Number of device in target process");
DEFINE_bool(enable_pt, false, "Enable Progress Thread (only used with nixl worker)");
DEFINE_bool(enable_vmm, false, "Enable VMM memory allocation when DRAM is requested");

// Storage backend(GDS, POSIX, HF3FS, OBJ) options
DEFINE_string (filepath, "", "File path for storage operations");
DEFINE_int32 (num_files, 1, "Number of files used by benchmark");
DEFINE_bool (storage_enable_direct, false, "Enable direct I/O for storage operations");

// GDS options - only used when backend is GDS
DEFINE_int32(gds_batch_pool_size, 32, "Batch pool size for GDS operations (default: 32, only used with GDS backend)");
DEFINE_int32(gds_batch_limit, 128, "Batch limit for GDS operations (default: 128, only used with GDS backend)");

// TODO: We should take rank wise device list as input to extend support
// <rank>:<device_list>, ...
// For example- 0:mlx5_0,mlx5_1,mlx5_2,1:mlx5_3,mlx5_4, ...
DEFINE_string(device_list, "all", "Comma-separated device name to use for \
		      communication (only used with nixl worker)");
DEFINE_string(etcd_endpoints, "http://localhost:2379", "ETCD server endpoints for communication");

// POSIX options - only used when backend is POSIX
DEFINE_string (posix_api_type,
               XFERBENCH_POSIX_API_AIO,
               "API type for POSIX operations [AIO, URING] (only used with POSIX backend)");

// DOCA GPUNetIO options - only used when backend is DOCA GPUNetIO
DEFINE_string(gpunetio_device_list, "0", "Comma-separated GPU CUDA device id to use for \
		      communication (only used with nixl worker)");

// OBJ options - only used when backend is OBJ
DEFINE_string(obj_access_key, "", "Access key for S3 backend");
DEFINE_string(obj_secret_key, "", "Secret key for S3 backend");
DEFINE_string(obj_session_token, "", "Session token for S3 backend");
DEFINE_string(obj_bucket_name, XFERBENCH_OBJ_BUCKET_NAME_DEFAULT, "Bucket name for S3 backend");
DEFINE_string(obj_scheme, XFERBENCH_OBJ_SCHEME_HTTP, "HTTP scheme for S3 backend [http, https]");
DEFINE_string(obj_region, XFERBENCH_OBJ_REGION_EU_CENTRAL_1, "Region for S3 backend");
DEFINE_bool(obj_use_virtual_addressing, false, "Use virtual addressing for S3 backend");
DEFINE_string(obj_endpoint_override, "", "Endpoint override for S3 backend");
DEFINE_string(obj_req_checksum,
              XFERBENCH_OBJ_REQ_CHECKSUM_SUPPORTED,
              "Required checksum for S3 backend [supported, required]");

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
bool xferBenchConfig::enable_vmm = false;
std::string xferBenchConfig::device_list = "";
std::string xferBenchConfig::etcd_endpoints = "";
std::string xferBenchConfig::benchmark_group = "default";
int xferBenchConfig::gds_batch_pool_size = 0;
int xferBenchConfig::gds_batch_limit = 0;
std::string xferBenchConfig::gpunetio_device_list = "";
std::vector<std::string> devices = { };
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

int
xferBenchConfig::loadFromFlags() {
    benchmark_group = FLAGS_benchmark_group;
    runtime_type = FLAGS_runtime_type;
    worker_type = FLAGS_worker_type;

    // Only load NIXL-specific configurations if using NIXL worker
    if (worker_type == XFERBENCH_WORKER_NIXL) {
        backend = FLAGS_backend;
        enable_pt = FLAGS_enable_pt;
        device_list = FLAGS_device_list;
        enable_vmm = FLAGS_enable_vmm;

#if defined(HAVE_CUDA) && !defined(HAVE_CUDA_FABRIC)
        if (enable_vmm) {
            std::cerr << "VMM is not supported in CUDA version " << CUDA_VERSION << std::endl;
            return -1;
        }
#endif
        // Load GDS-specific configurations if backend is GDS
        if (backend == XFERBENCH_BACKEND_GDS) {
            gds_batch_pool_size = FLAGS_gds_batch_pool_size;
            gds_batch_limit = FLAGS_gds_batch_limit;
            storage_enable_direct = FLAGS_storage_enable_direct;
        }

        // Load POSIX-specific configurations if backend is POSIX
        if (backend == XFERBENCH_BACKEND_POSIX) {
            posix_api_type = FLAGS_posix_api_type;
            storage_enable_direct = FLAGS_storage_enable_direct;

            // Validate POSIX API type
            if (posix_api_type != XFERBENCH_POSIX_API_AIO &&
                posix_api_type != XFERBENCH_POSIX_API_URING) {
                std::cerr << "Invalid POSIX API type: " << posix_api_type
                          << ". Must be one of [AIO, URING]" << std::endl;
                return -1;
            }
        }

        // Load DOCA-specific configurations if backend is DOCA
        if (backend == XFERBENCH_BACKEND_GPUNETIO) {
            gpunetio_device_list = FLAGS_gpunetio_device_list;
        }

        // Load HD3FS-specific configurations if backend is HD3FS
        if (backend == XFERBENCH_BACKEND_HF3FS) {
            storage_enable_direct = FLAGS_storage_enable_direct;
        }

        // Load OBJ-specific configurations if backend is OBJ
        if (backend == XFERBENCH_BACKEND_OBJ) {
            obj_access_key = FLAGS_obj_access_key;
            obj_secret_key = FLAGS_obj_secret_key;
            obj_session_token = FLAGS_obj_session_token;
            obj_bucket_name = FLAGS_obj_bucket_name;
            obj_scheme = FLAGS_obj_scheme;
            obj_region = FLAGS_obj_region;
            obj_use_virtual_addressing = FLAGS_obj_use_virtual_addressing;
            obj_endpoint_override = FLAGS_obj_endpoint_override;
            obj_req_checksum = FLAGS_obj_req_checksum;

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

    initiator_seg_type = FLAGS_initiator_seg_type;
    target_seg_type = FLAGS_target_seg_type;
    scheme = FLAGS_scheme;
    mode = FLAGS_mode;
    op_type = FLAGS_op_type;
    check_consistency = FLAGS_check_consistency;
    total_buffer_size = FLAGS_total_buffer_size;
    num_initiator_dev = FLAGS_num_initiator_dev;
    num_target_dev = FLAGS_num_target_dev;
    start_block_size = FLAGS_start_block_size;
    max_block_size = FLAGS_max_block_size;
    start_batch_size = FLAGS_start_batch_size;
    max_batch_size = FLAGS_max_batch_size;
    num_iter = FLAGS_num_iter;
    large_blk_iter_ftr = FLAGS_large_blk_iter_ftr;
    warmup_iter = FLAGS_warmup_iter;
    num_threads = FLAGS_num_threads;
    etcd_endpoints = FLAGS_etcd_endpoints;
    filepath = FLAGS_filepath;
    num_files = FLAGS_num_files;
    posix_api_type = FLAGS_posix_api_type;
    storage_enable_direct = FLAGS_storage_enable_direct;

    if (worker_type == XFERBENCH_WORKER_NVSHMEM) {
        if (!((XFERBENCH_SEG_TYPE_VRAM == initiator_seg_type) &&
              (XFERBENCH_SEG_TYPE_VRAM == target_seg_type) &&
              (1 == num_threads) &&
              (1 == num_initiator_dev) &&
              (1 == num_target_dev) &&
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
        std::cerr << "Incorrect buffer size configuration for Initiator"
                  << "(max_block_size * max_batch_size) is > (total_buffer_size / num_initiator_dev)"
                  << std::endl;
        return -1;
    }
    if ((max_block_size * max_batch_size) > (total_buffer_size / num_target_dev)) {
        std::cerr << "Incorrect buffer size configuration for Target"
                  << "(max_block_size * max_batch_size) is > (total_buffer_size / num_initiator_dev)"
                  << std::endl;
        return -1;
    }
    if (max_block_size > (total_buffer_size / num_threads)) {
        std::cerr << "Incorrect buffer size configuration" << " max_block_size(" << max_block_size
                  << ") >" << " (total_buffer_size / num_threads)("
                  << (total_buffer_size / num_threads) << ")" << std::endl;
        return -1;
    }

    if (large_blk_iter_ftr == 0 || large_blk_iter_ftr > num_iter) {
        std::cerr << "iter_factor must not be 0 and must be lower than num_iter" << std::endl;
        return -1;
    }

    int partition = (num_threads * large_blk_iter_ftr);
    if (num_iter % partition) {
        num_iter += partition - (num_iter % partition);
        std::cout << "WARNING: Adjusting num_iter to " << num_iter
                  << " to allow equal distribution to " << num_threads << " threads"
                  << std::endl;
    }
    if (warmup_iter % partition) {
        warmup_iter += partition - (warmup_iter % partition);
        std::cout << "WARNING: Adjusting warmup_iter to " << warmup_iter
                  << " to allow equal distribution to " << num_threads << " threads"
                  << std::endl;
    }
    partition = (num_initiator_dev * num_threads);
    if (total_buffer_size % partition) {
        std::cerr << "Total_buffer_size must be divisible by the product of num_threads and num_initiator_dev"
                  << ", next such value is " << total_buffer_size + partition - (total_buffer_size % partition)
                  << std::endl;
        return -1;
    }
    partition = (num_target_dev * num_threads);
    if (total_buffer_size % partition) {
        std::cerr << "Total_buffer_size must be divisible by the product of num_threads and num_target_dev"
                  << ", next such value is " << total_buffer_size + partition - (total_buffer_size % partition)
                  << std::endl;
        return -1;
    }

    return 0;
}

void
xferBenchConfig::printOption (const std::string &desc, const std::string &value) {
    std::cout << std::left << std::setw (60) << desc << ": " << value << std::endl;
}

void xferBenchConfig::printConfig() {
    std::cout << std::string(70, '*') << std::endl;
    std::cout << "NIXLBench Configuration" << std::endl;
    std::cout << std::string(70, '*') << std::endl;
    printOption ("Runtime (--runtime_type=[etcd])", runtime_type);
    if (runtime_type == XFERBENCH_RT_ETCD) {
        printOption ("ETCD Endpoint ", etcd_endpoints);
    }
    printOption ("Worker type (--worker_type=[nixl,nvshmem])", worker_type);
    if (worker_type == XFERBENCH_WORKER_NIXL) {
        printOption("Backend (--backend=[UCX,UCX_MO,GDS,POSIX,OBJ])", backend);
        printOption ("Enable pt (--enable_pt=[0,1])", std::to_string (enable_pt));
        printOption ("Device list (--device_list=dev1,dev2,...)", device_list);
        printOption ("Enable VMM (--enable_vmm=[0,1])", std::to_string (enable_vmm));

        // Print GDS options if backend is GDS
        if (backend == XFERBENCH_BACKEND_GDS) {
            printOption ("GDS batch pool size (--gds_batch_pool_size=N)",
                         std::to_string (gds_batch_pool_size));
            printOption ("GDS batch limit (--gds_batch_limit=N)", std::to_string (gds_batch_limit));
        }

        // Print POSIX options if backend is POSIX
        if (backend == XFERBENCH_BACKEND_POSIX) {
            printOption ("POSIX API type (--posix_api_type=[AIO,URING])", posix_api_type);
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
        }

        if (xferBenchConfig::isStorageBackend()) {
            printOption ("filepath (--filepath=path)", filepath);
            printOption ("Number of files (--num_files=N)", std::to_string (num_files));
            printOption ("Storage enable direct (--storage_enable_direct=[0,1])",
                         std::to_string (storage_enable_direct));
        }

        // Print DOCA GPUNetIO options if backend is DOCA GPUNetIO
        if (backend == XFERBENCH_BACKEND_GPUNETIO) {
            printOption ("GPU CUDA Device id list (--device_list=dev1,dev2,...)",
                         gpunetio_device_list);
        }
    }
    printOption ("Initiator seg type (--initiator_seg_type=[DRAM,VRAM])", initiator_seg_type);
    printOption ("Target seg type (--target_seg_type=[DRAM,VRAM])", target_seg_type);
    printOption ("Scheme (--scheme=[pairwise,manytoone,onetomany,tp])", scheme);
    printOption ("Mode (--mode=[SG,MG])", mode);
    printOption ("Op type (--op_type=[READ,WRITE])", op_type);
    printOption ("Check consistency (--check_consistency=[0,1])",
                 std::to_string (check_consistency));
    printOption ("Total buffer size (--total_buffer_size=N)", std::to_string (total_buffer_size));
    printOption ("Num initiator dev (--num_initiator_dev=N)", std::to_string (num_initiator_dev));
    printOption ("Num target dev (--num_target_dev=N)", std::to_string (num_target_dev));
    printOption ("Start block size (--start_block_size=N)", std::to_string (start_block_size));
    printOption ("Max block size (--max_block_size=N)", std::to_string (max_block_size));
    printOption ("Start batch size (--start_batch_size=N)", std::to_string (start_batch_size));
    printOption ("Max batch size (--max_batch_size=N)", std::to_string (max_batch_size));
    printOption ("Num iter (--num_iter=N)", std::to_string (num_iter));
    printOption ("Warmup iter (--warmup_iter=N)", std::to_string (warmup_iter));
    printOption("Large block iter factor (--large_blk_iter_ftr=N)",
                std::to_string(large_blk_iter_ftr));
    printOption ("Num threads (--num_threads=N)", std::to_string (num_threads));
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::endl;
}

std::vector<std::string> xferBenchConfig::parseDeviceList() {
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
                      << " provided for pairwise scheme " << devices.size()
                      << "# devices" << std::endl;
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
            XFERBENCH_BACKEND_HF3FS == xferBenchConfig::backend ||
            XFERBENCH_BACKEND_POSIX == xferBenchConfig::backend ||
            XFERBENCH_BACKEND_OBJ == xferBenchConfig::backend);
}
/**********
 * xferBench Utils
 **********/
xferBenchRT *xferBenchUtils::rt = nullptr;
std::string xferBenchUtils::dev_to_use = "";

void xferBenchUtils::setRT(xferBenchRT *rt) {
    xferBenchUtils::rt = rt;
}

void xferBenchUtils::setDevToUse(std::string dev) {
    dev_to_use = dev;
}

std::string xferBenchUtils::getDevToUse() {
    return dev_to_use;
}

static bool allBytesAre(void* buffer, size_t size, uint8_t value) {
    uint8_t* byte_buffer = static_cast<uint8_t*>(buffer);

    // Iterate over each byte in the buffer
    for (size_t i = 0; i < size; ++i) {
        if (byte_buffer[i] != value) {
            return false; // Return false if any byte doesn't match the value
        }
    }
    return true; // All bytes match the value
}

void xferBenchUtils::checkConsistency(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    for (const auto &iov_list: iov_lists) {
        for(const auto &iov: iov_list) {
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
                        CHECK_CUDA_ERROR(cudaMemcpy(addr, (void *)iov.addr, len,
                                                    cudaMemcpyDeviceToHost), "cudaMemcpy failed");
#else
                        std::cerr << "Failure in consistency check: VRAM segment type not supported without CUDA"
                                  << std::endl;
                        exit(EXIT_FAILURE);
#endif
                    } else {
                        addr = (void *)iov.addr;
                    }
                } else if (xferBenchConfig::op_type == XFERBENCH_OP_WRITE) {
                    addr = calloc(1, len);
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
                    CHECK_CUDA_ERROR(cudaMemcpy(addr, (void *)iov.addr, len,
                                                cudaMemcpyDeviceToHost), "cudaMemcpy failed");
#else
                    std::cerr << "Failure in consistency check: VRAM segment type not supported without CUDA"
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

            if("WRITE" == xferBenchConfig::op_type) {
                check_val = XFERBENCH_INITIATOR_BUFFER_ELEMENT;
            } else if("READ" == xferBenchConfig::op_type) {
                check_val = XFERBENCH_TARGET_BUFFER_ELEMENT;
            }
            rc = allBytesAre(addr, len, check_val);
            if (true != rc) {
                std::cerr << "Consistency check failed\n" << std::flush;
            }
            // Free the addr only if is allocated here
            if (is_allocated) {
                free(addr);
            }
        }
    }
}

void xferBenchUtils::printStatsHeader() {
    if (IS_PAIRWISE_AND_SG() && rt->getSize() > 2) {
        std::cout << std::left << std::setw(20) << "Block Size (B)"
                  << std::setw(15) << "Batch Size"
                  << std::setw(15) << "Avg Lat. (us)"
                  << std::setw(15) << "B/W (MiB/Sec)"
                  << std::setw(15) << "B/W (GiB/Sec)"
                  << std::setw(15) << "B/W (GB/Sec)"
                  << std::setw(25) << "Aggregate B/W (GB/Sec)"
                  << std::setw(20) << "Network Util (%)"
                  << std::endl;
    } else {
        std::cout << std::left << std::setw(20) << "Block Size (B)"
                  << std::setw(15) << "Batch Size"
                  << std::setw(15) << "Avg Lat. (us)"
                  << std::setw(15) << "B/W (MiB/Sec)"
                  << std::setw(15) << "B/W (GiB/Sec)"
                  << std::setw(15) << "B/W (GB/Sec)"
                  << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;
}

void xferBenchUtils::printStats(bool is_target, size_t block_size, size_t batch_size, double total_duration) {
    size_t total_data_transferred = 0;
    double avg_latency = 0, throughput = 0, throughput_gib = 0, throughput_gb = 0;
    double totalbw = 0;

    int num_iter = xferBenchConfig::num_iter;

    if (block_size > LARGE_BLOCK_SIZE) {
        num_iter /= xferBenchConfig::large_blk_iter_ftr;
    }

    // TODO: We can avoid this by creating a sub-communicator across initiator ranks
    // if (isTarget() && IS_PAIRWISE_AND_SG() && rt->getSize() > 2) { - Fix this isTarget can not be called here
    if (is_target && IS_PAIRWISE_AND_SG() && rt->getSize() > 2) {
        rt->reduceSumDouble(&throughput_gb, &totalbw, 0);
        return;
    }

    total_data_transferred = ((block_size * batch_size) * num_iter); // In Bytes
    avg_latency = (total_duration / (num_iter * batch_size)); // In microsec
    if (IS_PAIRWISE_AND_MG()) {
        total_data_transferred *= xferBenchConfig::num_initiator_dev; // In Bytes
        avg_latency /= xferBenchConfig::num_initiator_dev; // In microsec
    }

    throughput = (((double) total_data_transferred / (1024 * 1024)) /
                   (total_duration / 1e6));   // In MiB/Sec
    throughput_gib = (throughput / 1024);   // In GiB/Sec
    throughput_gb = (((double) total_data_transferred / (1000 * 1000 * 1000)) /
                   (total_duration / 1e6));   // In GB/Sec

    if (IS_PAIRWISE_AND_SG() && rt->getSize() > 2) {
        rt->reduceSumDouble(&throughput_gb, &totalbw, 0);
    } else {
        totalbw = throughput_gb;
    }

    if (IS_PAIRWISE_AND_SG() && rt->getRank() != 0) {
        return;
    }

    // Tabulate print with fixed width for each string
    if (IS_PAIRWISE_AND_SG() && rt->getSize() > 2) {
        std::cout << std::left << std::setw(20) << block_size
                  << std::setw(15) << batch_size
                  << std::setw(15) << avg_latency
                  << std::setw(15) << throughput
                  << std::setw(15) << throughput_gib
                  << std::setw(15) << throughput_gb
                  << std::setw(25) << totalbw
                  << std::setw(20) << (totalbw / (rt->getSize()/2 * MAXBW))*100
                  << std::endl;
    } else {
        std::cout << std::left << std::setw(20) << block_size
                  << std::setw(15) << batch_size
                  << std::setw(15) << avg_latency
                  << std::setw(15) << throughput
                  << std::setw(15) << throughput_gib
                  << std::setw(15) << throughput_gb
                  << std::endl;
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
