// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Test double: ownership/lifetime validation only, not NIXL implementation qualification.
#pragma once
#include <stdexcept>
#include <map>
#include <string>
#include <vector>
#include <atomic>
#include <set>

inline void
mockCheck(bool value) {
    if (!value) {
        throw std::runtime_error("mock NIXL ownership contract violated");
    }
}
using nixl_status_t = int;
inline constexpr int NIXL_SUCCESS = 0, NIXL_IN_PROG = 1, NIXL_ERR_UNKNOWN = -1,
                     NIXL_ERR_REPOST_ACTIVE = -2;
enum class nixl_thread_sync_t { NIXL_THREAD_SYNC_RW, NIXL_THREAD_SYNC_NONE };

struct nixlAgentConfig {
    bool useProgThread = true;
    nixl_thread_sync_t syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE;
};

struct nixlBackendH {
    int owner;
};

struct nixlXferReqH {
    int owner;
    int polls = 2;
};

struct nixlDlistH {
    int owner;
};

using nixl_reg_dlist_t = std::vector<int>;
using nixl_b_params_t = std::map<std::string, std::string>;
using nixl_mem_list_t = std::vector<int>;

struct nixl_opt_args_t {
    std::vector<nixlBackendH *> backends;
};
#ifdef NIXLBENCH_LOOP_TEST
void
completedRequest(nixlXferReqH *);
#endif
struct nixlAgent {
    inline static std::set<int> alive;
    inline static int next = 0, live = 0, backends = 0, registered = 0;
    inline static int fail_backend = -1, fail_registration = -1, register_calls = 0;
    inline static std::atomic<int> releases{0}, lists{0};
    inline static bool fail_release = false, fail_list = false;
    int id = next++;
    nixlBackendH *backend = nullptr;

    nixlAgent(const std::string &, const nixlAgentConfig &config) {
        mockCheck(!config.useProgThread);
        mockCheck(config.syncMode == nixl_thread_sync_t::NIXL_THREAD_SYNC_RW);
        ++live;
        alive.insert(id);
    }

    ~nixlAgent() {
        if (backend) {
            delete backend;
            --backends;
        }
        --live;
        alive.erase(id);
    }

    int
    getPluginParams(const std::string &, nixl_mem_list_t &, nixl_b_params_t &) {
        return 0;
    }

    int
    createBackend(const std::string &, const nixl_b_params_t &, nixlBackendH *&out) {
        if (id == fail_backend) {
            return -1;
        }
        out = backend = new nixlBackendH{id};
        ++backends;
        return 0;
    }

    int
    registerMem(nixl_reg_dlist_t &, nixl_opt_args_t *options) {
        mockCheck(options->backends.at(0) == backend);
        if (register_calls++ == fail_registration) {
            return -1;
        }
        ++registered;
        return 0;
    }

    int
    deregisterMem(nixl_reg_dlist_t &, nixl_opt_args_t *options) {
        mockCheck(options->backends.at(0) == backend);
        --registered;
        return 0;
    }

    int
    getXferStatus(nixlXferReqH *request) {
        mockCheck(request->owner == id);
        if (request->polls-- > 0) {
            return NIXL_IN_PROG;
        }
#ifdef NIXLBENCH_LOOP_TEST
        completedRequest(request);
#endif
        return NIXL_SUCCESS;
    }

    int
    releaseXferReq(nixlXferReqH *request) {
        mockCheck(request->owner == id);
        if (fail_release) {
            return NIXL_ERR_UNKNOWN;
        }
        if (request->polls >= 0) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
        delete request;
        ++releases;
        return 0;
    }

    int
    releasedDlistH(nixlDlistH *list) {
        mockCheck(list->owner == id);
        if (fail_list) {
            return NIXL_ERR_UNKNOWN;
        }
        delete list;
        ++lists;
        return 0;
    }
};
