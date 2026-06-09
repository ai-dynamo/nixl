/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Shared smoke harness: 2-file register -> WRITE/READ/verify -> dereg with fd-leak check.

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "nixl.h"
#include "nixl_descriptors.h"
#include "nixl_params.h"
#include "nixl_types.h"

namespace nixl_test {

inline size_t
countOpenFds() {
    size_t n = 0;
    for ([[maybe_unused]] auto &e : std::filesystem::directory_iterator("/proc/self/fd")) {
        ++n;
    }
    return n;
}

inline void *
allocDram(size_t s) {
    void *p = nullptr;
    return posix_memalign(&p, sysconf(_SC_PAGESIZE), s) == 0 ? p : nullptr;
}

inline bool
verifyDram(const void *p, unsigned char v, size_t s) {
    auto *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < s; ++i) {
        if (b[i] != v) {
            return false;
        }
    }
    return true;
}

// One register/WRITE/READ/verify/deregister cycle; baseline non-null asserts net-zero fd delta.
inline bool
doRoundtrip(nixlAgent &agent,
            const char *agent_name,
            const std::string &path_a,
            const std::string &path_b,
            size_t size,
            const size_t *baseline) {
    constexpr unsigned char kPatternA = 0xA5;
    constexpr unsigned char kPatternB = 0x5A;

    nixl_reg_dlist_t file_descs(FILE_SEG);
    uint64_t file_devid = 0;
    for (const std::string &p : {path_a, path_b}) {
        nixlBlobDesc fd;
        fd.addr = 0;
        fd.len = size;
        fd.devId = file_devid++; // distinct so xfer-time lookup disambiguates
        fd.metaInfo = std::string("rw:") + p;
        file_descs.addDesc(fd);
    }
    if (agent.registerMem(file_descs) != NIXL_SUCCESS) {
        return false;
    }

    void *buf_a = allocDram(size);
    void *buf_b = allocDram(size);
    nixl_reg_dlist_t buf_descs(DRAM_SEG);
    for (void *b : {buf_a, buf_b}) {
        nixlBlobDesc bd;
        bd.addr = reinterpret_cast<uintptr_t>(b);
        bd.len = size;
        bd.devId = 0; // buf addrs already disambiguate
        buf_descs.addDesc(bd);
    }

    bool ok = false;
    if (buf_a && buf_b && agent.registerMem(buf_descs) == NIXL_SUCCESS) {
        std::memset(buf_a, kPatternA, size);
        std::memset(buf_b, kPatternB, size);
        nixl_xfer_dlist_t bx = buf_descs.trim();
        nixl_xfer_dlist_t fx = file_descs.trim();
        auto xfer = [&](nixl_xfer_op_t op) {
            nixlXferReqH *req = nullptr;
            nixl_status_t r = agent.createXferReq(op, bx, fx, agent_name, req);
            if (r == NIXL_SUCCESS) {
                r = agent.postXferReq(req);
                while (r == NIXL_IN_PROG) {
                    r = agent.getXferStatus(req);
                }
                agent.releaseXferReq(req);
            }
            return r;
        };
        if (xfer(NIXL_WRITE) == NIXL_SUCCESS) {
            std::memset(buf_a, 0, size);
            std::memset(buf_b, 0, size);
            if (xfer(NIXL_READ) == NIXL_SUCCESS && verifyDram(buf_a, kPatternA, size) &&
                verifyDram(buf_b, kPatternB, size)) {
                ok = true;
            }
        }
        agent.deregisterMem(buf_descs);
    }
    if (buf_a) {
        free(buf_a);
    }
    if (buf_b) {
        free(buf_b);
    }
    if (agent.deregisterMem(file_descs) != NIXL_SUCCESS) {
        ok = false;
    }
    if (baseline && ok && countOpenFds() != *baseline) {
        ok = false;
    }
    return ok;
}

// A ro: registration of a missing `path` must fail gracefully, not crash.
inline bool
checkMissingFileRejected(nixlAgent &agent, const std::string &path) {
    nixl_reg_dlist_t file_descs(FILE_SEG);
    nixlBlobDesc fd;
    fd.addr = 0;
    fd.len = 4096;
    fd.devId = 0;
    fd.metaInfo = std::string("ro:") + path;
    file_descs.addDesc(fd);
    if (agent.registerMem(file_descs) == NIXL_SUCCESS) {
        agent.deregisterMem(file_descs);
        return false;
    }
    return true;
}

// 0 on success or SKIP (backend unavailable), 1 on hard failure.
inline int
runPathModeSmoke(const char *agent_name,
                 const char *backend_name,
                 const char *file_path,
                 size_t size) {
    const std::string path_a = file_path;
    const std::string path_b = std::string(file_path) + ".b";

    for (const std::string &p : {path_a, path_b}) {
        if (auto *f = std::fopen(p.c_str(), "wb")) {
            std::fseek(f, size - 1, SEEK_SET);
            std::fputc(0, f);
            std::fclose(f);
        } else {
            return 1;
        }
    }

    nixlAgentConfig cfg;
    nixlAgent agent(agent_name, cfg);
    nixl_b_params_t params;
    nixlBackendH *be = nullptr;
    if (agent.createBackend(backend_name, params, be) != NIXL_SUCCESS || !be) {
        std::cout << "SKIP: " << backend_name << " createBackend failed" << std::endl;
        std::remove(path_a.c_str());
        std::remove(path_b.c_str());
        return 0;
    }

    // Warm-up absorbs backends' lazy-init fds (e.g. libhf3fs's hf3fs_iorcreate) into baseline.
    if (!doRoundtrip(agent, agent_name, path_a, path_b, size, nullptr)) {
        std::cerr << backend_name << " path-mode warm-up FAILED" << std::endl;
        std::remove(path_a.c_str());
        std::remove(path_b.c_str());
        return 1;
    }

    const std::string missing = std::string(file_path) + ".missing";
    std::remove(missing.c_str());
    if (!checkMissingFileRejected(agent, missing)) {
        std::cerr << backend_name << " path-mode missing-file check FAILED" << std::endl;
        std::remove(path_a.c_str());
        std::remove(path_b.c_str());
        return 1;
    }

    const size_t baseline = countOpenFds();
    bool ok = doRoundtrip(agent, agent_name, path_a, path_b, size, &baseline);
    std::remove(path_a.c_str());
    std::remove(path_b.c_str());
    if (ok) {
        std::cout << backend_name << " path-mode roundtrip OK (baseline=" << baseline << ")"
                  << std::endl;
        return 0;
    }
    std::cerr << backend_name << " path-mode roundtrip FAILED" << std::endl;
    return 1;
}

} // namespace nixl_test
