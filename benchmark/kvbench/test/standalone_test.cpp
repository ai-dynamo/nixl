/*
 * Standalone C++ test to isolate whether the Python process environment
 * causes the 37 vs 48 GB/s gap. Uses the exact same NIXL API calls as
 * nixlbench and our binding, but without Python.
 *
 * Build:  g++ -O3 -DNDEBUG -std=c++17 -pthread -fopenmp \
 *           -I/usr/local/nixl/include -I/usr/local/nixl/include/nixl \
 *           standalone_test.cpp -L/usr/local/nixl/lib/x86_64-linux-gnu \
 *           -lnixl -lgomp -o standalone_test
 *
 * Run:    ./standalone_test /mnt/vast/mtennenhaus/nixlbench_fresh
 */
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "nixl.h"

struct IOV {
    uintptr_t addr;
    size_t len;
    int devId;
};

static void iovToXferDlist(const std::vector<IOV> &iovs, nixl_xfer_dlist_t &dlist) {
    nixlBasicDesc desc;
    for (auto &iov : iovs) {
        desc.addr = iov.addr; desc.len = iov.len; desc.devId = iov.devId;
        dlist.addDesc(desc);
    }
}

static inline nixl_status_t execSingle(nixlAgent *agent, nixlXferReqH *req) {
    nixl_status_t rc = agent->postXferReq(req);
    while (NIXL_IN_PROG == rc) rc = agent->getXferStatus(req);
    return rc;
}

static int execIterations(nixlAgent *agent, nixl_xfer_op_t op,
                          nixl_xfer_dlist_t &local_desc, nixl_xfer_dlist_t &remote_desc,
                          const std::string &target, nixl_opt_args_t &params, int num_iter) {
    nixlXferReqH *req = nullptr;
    if (agent->createXferReq(op, local_desc, remote_desc, target, req, &params) != NIXL_SUCCESS)
        return -1;
    for (int i = 0; i < num_iter; ++i) {
        nixl_status_t rc = execSingle(agent, req);
        if (__builtin_expect(rc != NIXL_SUCCESS, 0)) { agent->releaseXferReq(req); return -1; }
    }
    if (agent->releaseXferReq(req) != NIXL_SUCCESS) return -1;
    return 0;
}

static int64_t execTransfer(nixlAgent *agent,
                            const std::vector<std::vector<IOV>> &local_iovs,
                            const std::vector<std::vector<IOV>> &remote_iovs,
                            const std::string &agent_name, nixl_xfer_op_t op,
                            int num_iter, int num_threads) {
    int ret = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        nixl_xfer_dlist_t local_desc(DRAM_SEG), remote_desc(FILE_SEG);
        iovToXferDlist(local_iovs[tid], local_desc);
        iovToXferDlist(remote_iovs[tid], remote_desc);
        nixl_opt_args_t params;
        int r = execIterations(agent, op, local_desc, remote_desc, agent_name, params, num_iter);
        if (__builtin_expect(r != 0, 0)) ret = r;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ret != 0) return -1;
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <storage_path>\n"; return 1; }
    std::string storage_path = argv[1];

    const int NUM_FILES = 16, NUM_THREADS = 16, BATCH_SIZE = 256;
    const size_t BLOCK_SIZE = 1048576, FILE_SIZE = 1073741824;
    const int NUM_ITERS = 112, WARMUP_ITERS = 16;
    long page_size = sysconf(_SC_PAGESIZE);

    std::cout << "Standalone NIXL storage test\n"
              << "  files=" << NUM_FILES << " threads=" << NUM_THREADS
              << " batch=" << BATCH_SIZE << " block=1M\n"
              << "  file_size=1GB iters=" << NUM_ITERS << " warmup=" << WARMUP_ITERS << "\n";

    nixlAgentConfig cfg(false, false, 0, nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
    nixlAgent agent("initiator", cfg);

    nixlBackendH *backend = nullptr;
    nixl_b_params_t bp;
    bp["use_uring"] = "true";
    bp["use_aio"] = "false";
    bp["use_posix_aio"] = "false";
    agent.createBackend("POSIX", bp, backend);
    std::cout << "  backend created\n";

    nixl_opt_args_t reg_opt;
    reg_opt.backends.push_back(backend);

    // Open files
    std::vector<int> fds;
    for (int i = 0; i < NUM_FILES; i++) {
        std::string fp = storage_path + "/nixlbench_posix_test_file_initiator_" + std::to_string(i);
        int fd = open(fp.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) { std::cerr << "Failed to open " << fp << "\n"; return 1; }
        fds.push_back(fd);

        nixl_reg_dlist_t freg(FILE_SEG);
        nixlBlobDesc fdesc;
        fdesc.addr = 0; fdesc.len = FILE_SIZE; fdesc.devId = fd;
        freg.addDesc(fdesc);
        agent.registerMem(freg, &reg_opt);
    }
    std::cout << "  " << NUM_FILES << " files opened + registered\n";

    // Allocate DRAM buffers
    size_t buf_size = ((BATCH_SIZE * BLOCK_SIZE + page_size - 1) / page_size) * page_size;
    std::vector<void*> bufs(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        posix_memalign(&bufs[i], page_size, buf_size);
        memset(bufs[i], 0, buf_size);
        nixl_reg_dlist_t dreg(DRAM_SEG);
        nixlBlobDesc ddesc;
        ddesc.addr = (uintptr_t)bufs[i]; ddesc.len = buf_size; ddesc.devId = 0;
        dreg.addDesc(ddesc);
        agent.registerMem(dreg, &reg_opt);
    }
    std::cout << "  " << NUM_THREADS << " x " << buf_size/(1024*1024) << "MB DRAM allocated + registered\n";

    // Build IOV lists (round-robin, matching nixlbench exchangeIOV)
    std::vector<std::vector<IOV>> local_iovs(NUM_THREADS), remote_iovs(NUM_THREADS);
    int fd_idx = 0;
    uint64_t file_offset = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        size_t buf_offset = 0;
        for (int j = 0; j < BATCH_SIZE; j++) {
            local_iovs[t].push_back({(uintptr_t)bufs[t] + buf_offset, BLOCK_SIZE, 0});
            remote_iovs[t].push_back({file_offset, BLOCK_SIZE, fds[fd_idx]});
            buf_offset += BLOCK_SIZE;
            fd_idx++;
            if (fd_idx >= NUM_FILES) {
                file_offset += BLOCK_SIZE;
                fd_idx = 0;
            }
        }
        file_offset += BLOCK_SIZE;
    }
    std::cout << "  IOV lists built (round-robin)\n";

    // Warmup
    std::cout << "  warmup (" << WARMUP_ITERS << " iters)...\n";
    execTransfer(&agent, local_iovs, remote_iovs, "initiator", NIXL_READ, WARMUP_ITERS, NUM_THREADS);

    // Timed
    std::cout << "  timed (" << NUM_ITERS << " iters)...\n";
    int64_t us = execTransfer(&agent, local_iovs, remote_iovs, "initiator", NIXL_READ, NUM_ITERS, NUM_THREADS);

    size_t total = (size_t)NUM_THREADS * BATCH_SIZE * BLOCK_SIZE * NUM_ITERS;
    double bw = (double)total / ((double)us / 1e6) / (1024.0*1024.0*1024.0);
    std::cout << "\n>>> BW = " << bw << " GB/s <<<\n";
    std::cout << "  wall_time = " << us/1000.0 << " ms\n";
    std::cout << "  total_data = " << total/(1024*1024*1024) << " GB\n";

    for (int fd : fds) close(fd);
    for (void *b : bufs) free(b);
    return 0;
}
