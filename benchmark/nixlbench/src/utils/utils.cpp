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

#include "runtime/etcd/etcd_rt.h"
#include "utils/utils.h"


/**********
 * xferBench Config
 **********/
DEFINE_string(runtime_type, XFERBENCH_RT_ETCD, "Runtime type to use for communication [ETCD]");
DEFINE_string(worker_type, XFERBENCH_WORKER_NIXL, "Type of worker [nixl, nvshmem]");
DEFINE_string(backend, XFERBENCH_BACKEND_UCX, "Name of communication backend [UCX, UCX_MO, GDS, POSIX] \
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
DEFINE_int32(warmup_iter, 100, "Number of warmup iterations before timing");
DEFINE_int32(num_threads, 1,
             "Number of threads used by benchmark."
             " Num_iter must be greater or equal than num_threads and equally divisible by num_threads."
             " (Default: 1)");
DEFINE_int32(num_files, 1, "Number of files used by benchmark");
DEFINE_int32(num_initiator_dev, 1, "Number of device in initiator process");
DEFINE_int32(num_target_dev, 1, "Number of device in target process");
DEFINE_bool(enable_pt, false, "Enable Progress Thread (only used with nixl worker)");
DEFINE_bool(enable_vmm, false, "Enable VMM memory allocation when DRAM is requested");
// GDS options - only used when backend is GDS
DEFINE_string(gds_filepath, "", "File path for GDS operations (only used with GDS backend)");
DEFINE_int32(gds_batch_pool_size, 32, "Batch pool size for GDS operations (default: 32, only used with GDS backend)");
DEFINE_int32(gds_batch_limit, 128, "Batch limit for GDS operations (default: 128, only used with GDS backend)");

// TODO: We should take rank wise device list as input to extend support
// <rank>:<device_list>, ...
// For example- 0:mlx5_0,mlx5_1,mlx5_2,1:mlx5_3,mlx5_4, ...
DEFINE_string(device_list, "all", "Comma-separated device name to use for \
		      communication (only used with nixl worker)");
DEFINE_string(etcd_endpoints, "http://localhost:2379", "ETCD server endpoints for communication");

// POSIX options - only used when backend is POSIX
DEFINE_string(posix_api_type, XFERBENCH_POSIX_API_AIO, "API type for POSIX operations [AIO, URING] (only used with POSIX backend)");
DEFINE_string(posix_filepath, "", "File path for POSIX operations (only used with POSIX backend)");
DEFINE_bool(storage_enable_direct, false, "Enable direct I/O for storage operations (only used with POSIX backend)");

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
int xferBenchConfig::warmup_iter = 0;
int xferBenchConfig::num_threads = 0;
bool xferBenchConfig::enable_pt = false;
bool xferBenchConfig::enable_vmm = false;
std::string xferBenchConfig::device_list = "";
std::string xferBenchConfig::etcd_endpoints = "";
std::string xferBenchConfig::gds_filepath = "";
int xferBenchConfig::gds_batch_pool_size = 0;
int xferBenchConfig::gds_batch_limit = 0;
std::vector<std::string> devices = { };
int xferBenchConfig::num_files = 0;
std::string xferBenchConfig::posix_api_type = "";
std::string xferBenchConfig::posix_filepath = "";
bool xferBenchConfig::storage_enable_direct = false;

int xferBenchConfig::loadFromFlags() {
    runtime_type = FLAGS_runtime_type;
    worker_type = FLAGS_worker_type;

    // Only load NIXL-specific configurations if using NIXL worker
    if (worker_type == XFERBENCH_WORKER_NIXL) {
        backend = FLAGS_backend;
        enable_pt = FLAGS_enable_pt;
        device_list = FLAGS_device_list;
        enable_vmm = FLAGS_enable_vmm;

#if !HAVE_CUDA_FABRIC
        if (enable_vmm) {
            std::cerr << "VMM is not supported in CUDA version " << CUDA_VERSION << std::endl;
            return -1;
        }
#endif

        // Load GDS-specific configurations if backend is GDS
        if (backend == XFERBENCH_BACKEND_GDS) {
            gds_filepath = FLAGS_gds_filepath;
            gds_batch_pool_size = FLAGS_gds_batch_pool_size;
            gds_batch_limit = FLAGS_gds_batch_limit;
            num_files = FLAGS_num_files;
            storage_enable_direct = FLAGS_storage_enable_direct;
        }

        // Load POSIX-specific configurations if backend is POSIX
        if (backend == XFERBENCH_BACKEND_POSIX) {
            posix_api_type = FLAGS_posix_api_type;
            posix_filepath = FLAGS_posix_filepath;
            storage_enable_direct = FLAGS_storage_enable_direct;
            num_files = FLAGS_num_files;

            // Validate POSIX API type
            if (posix_api_type != XFERBENCH_POSIX_API_AIO &&
                posix_api_type != XFERBENCH_POSIX_API_URING) {
                std::cerr << "Invalid POSIX API type: " << posix_api_type
                          << ". Must be one of [AIO, URING]" << std::endl;
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
    warmup_iter = FLAGS_warmup_iter;
    num_threads = FLAGS_num_threads;
    etcd_endpoints = FLAGS_etcd_endpoints;
    num_files = FLAGS_num_files;
    posix_api_type = FLAGS_posix_api_type;
    posix_filepath = FLAGS_posix_filepath;
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

    int partition = (num_threads * LARGE_BLOCK_SIZE_ITER_FACTOR);
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

void xferBenchConfig::printConfig() {
    std::cout << std::string(70, '*') << std::endl;
    std::cout << "NIXLBench Configuration" << std::endl;
    std::cout << std::string(70, '*') << std::endl;
    std::cout << std::left << std::setw(60) << "Runtime (--runtime_type=[etcd])" << ": "
              << runtime_type << std::endl;
    if (runtime_type == XFERBENCH_RT_ETCD) {
        std::cout << std::left << std::setw(60) << "ETCD Endpoint " << ": "
	          << etcd_endpoints << std::endl;
    }
    std::cout << std::left << std::setw(60) << "Worker type (--worker_type=[nixl,nvshmem])" << ": "
              << worker_type << std::endl;
    if (worker_type == XFERBENCH_WORKER_NIXL) {
        std::cout << std::left << std::setw(60) << "Backend (--backend=[UCX,UCX_MO,GDS,POSIX])" << ": "
                  << backend << std::endl;
        std::cout << std::left << std::setw(60) << "Enable pt (--enable_pt=[0,1])" << ": "
                  << enable_pt << std::endl;
        std::cout << std::left << std::setw(60) << "Device list (--device_list=dev1,dev2,...)" << ": "
                  << device_list << std::endl;
        std::cout << std::left << std::setw(60) << "Enable VMM (--enable_vmm=[0,1])" << ": "
                  << enable_vmm << std::endl;

        // Print GDS options if backend is GDS
        if (backend == XFERBENCH_BACKEND_GDS) {
            std::cout << std::left << std::setw(60) << "GDS filepath (--gds_filepath=path)" << ": "
                      << gds_filepath << std::endl;
            std::cout << std::left << std::setw(60) << "GDS batch pool size (--gds_batch_pool_size=N)" << ": "
                      << gds_batch_pool_size << std::endl;
            std::cout << std::left << std::setw(60) << "GDS batch limit (--gds_batch_limit=N)" << ": "
                      << gds_batch_limit << std::endl;
            std::cout << std::left << std::setw(60) << "GDS enable direct (--gds_enable_direct=[0,1])" << ": "
                      << storage_enable_direct << std::endl;
            std::cout << std::left << std::setw(60) << "Number of files (--num_files=N)" << ": "
                      << num_files << std::endl;
        }

        // Print POSIX options if backend is POSIX
        if (backend == XFERBENCH_BACKEND_POSIX) {
            std::cout << std::left << std::setw(60) << "POSIX API type (--posix_api_type=[AIO,URING])" << ": "
                      << posix_api_type << std::endl;
            std::cout << std::left << std::setw(60) << "POSIX filepath (--posix_filepath=path)" << ": "
                      << posix_filepath << std::endl;
            std::cout << std::left << std::setw(60) << "POSIX enable direct (--storage_enable_direct=[0,1])" << ": "
                      << storage_enable_direct << std::endl;
            std::cout << std::left << std::setw(60) << "Number of files (--num_files=N)" << ": "
                      << num_files << std::endl;
        }
    }
    std::cout << std::left << std::setw(60) << "Initiator seg type (--initiator_seg_type=[DRAM,VRAM])" << ": "
              << initiator_seg_type << std::endl;
    std::cout << std::left << std::setw(60) << "Target seg type (--target_seg_type=[DRAM,VRAM])" << ": "
              << target_seg_type << std::endl;
    std::cout << std::left << std::setw(60) << "Scheme (--scheme=[pairwise,manytoone,onetomany,tp])" << ": "
              << scheme << std::endl;
    std::cout << std::left << std::setw(60) << "Mode (--mode=[SG,MG])" << ": "
              << mode << std::endl;
    std::cout << std::left << std::setw(60) << "Op type (--op_type=[READ,WRITE])" << ": "
              << op_type << std::endl;
    std::cout << std::left << std::setw(60) << "Check consistency (--check_consistency=[0,1])" << ": "
              << check_consistency << std::endl;
    std::cout << std::left << std::setw(60) << "Total buffer size (--total_buffer_size=N)" << ": "
              << total_buffer_size << std::endl;
    std::cout << std::left << std::setw(60) << "Num initiator dev (--num_initiator_dev=N)" << ": "
              << num_initiator_dev << std::endl;
    std::cout << std::left << std::setw(60) << "Num target dev (--num_target_dev=N)" << ": "
              << num_target_dev << std::endl;
    std::cout << std::left << std::setw(60) << "Start block size (--start_block_size=N)" << ": "
              << start_block_size << std::endl;
    std::cout << std::left << std::setw(60) << "Max block size (--max_block_size=N)" << ": "
              << max_block_size << std::endl;
    std::cout << std::left << std::setw(60) << "Start batch size (--start_batch_size=N)" << ": "
              << start_batch_size << std::endl;
    std::cout << std::left << std::setw(60) << "Max batch size (--max_batch_size=N)" << ": "
              << max_batch_size << std::endl;
    std::cout << std::left << std::setw(60) << "Num iter (--num_iter=N)" << ": "
              << num_iter << std::endl;
    std::cout << std::left << std::setw(60) << "Warmup iter (--warmup_iter=N)" << ": "
              << warmup_iter << std::endl;
    std::cout << std::left << std::setw(60) << "Num threads (--num_threads=N)" << ": "
              << num_threads << std::endl;
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

            if ((xferBenchConfig::backend == XFERBENCH_BACKEND_GDS) ||
                (xferBenchConfig::backend == XFERBENCH_BACKEND_POSIX)) {
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
                    ssize_t rc = pread(iov.devId, addr, len, iov.addr);
                    if (rc < 0) {
                        std::cerr << "Failed to read from device: " << iov.devId
                                  << " with error: " << strerror(errno) << std::endl;
                        exit(EXIT_FAILURE);
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
        num_iter /= LARGE_BLOCK_SIZE_ITER_FACTOR;
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
