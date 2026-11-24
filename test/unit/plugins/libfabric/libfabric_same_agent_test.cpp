/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Regression test: a same-agent (loopback) LIBFABRIC write posts real rail writes, so it must
// report done only after every one of them completed. postXfer used to overwrite the expected
// request count with 0 for same-agent transfers, so a transfer either reported done before its
// writes landed (and its handle was freed under in-flight writes), or, when a write completed
// before the overwrite, never reported done at all.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include "nixl.h"
#include "test_utils.h"

using namespace std;
using namespace std::chrono;

static const string kAgent("SameAgent");
// Many descriptors per transfer, so writes are still being posted when the first ones complete.
static constexpr size_t kDescs = 64;
static constexpr size_t kDescSize = 1 << 20;
static constexpr size_t kTotal = kDescs * kDescSize;
// Back-to-back transfers, released right away, so a handle address gets reused.
static constexpr int kIters = 20;
static const seconds kDeadline(10);

static void *
allocBuffer(size_t len) {
    void *ptr = nullptr;
    nixl_exit_on_failure(posix_memalign(&ptr, 4096, len) == 0, "posix_memalign failed", kAgent);
    return ptr;
}

static bool
allBytesAre(const uint8_t *buf, size_t len, uint8_t value) {
    for (size_t i = 0; i < len; i++) {
        if (buf[i] != value) {
            return false;
        }
    }
    return true;
}

int
main() {
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    nixlAgent agent(kAgent, cfg);

    nixl_b_params_t params;
    nixl_mem_list_t mems;
    nixl_exit_on_failure(agent.getPluginParams("LIBFABRIC", mems, params),
                         "Failed to get LIBFABRIC plugin params",
                         kAgent);
    nixlBackendH *backend = nullptr;
    nixl_exit_on_failure(agent.createBackend("LIBFABRIC", params, backend),
                         "Failed to create LIBFABRIC backend",
                         kAgent);
    nixl_opt_args_t extra;
    extra.backends.push_back(backend);

    // src and dst are the two halves of one registration, so both get the same rails. Registered
    // separately, the DRAM rail policy can put them on different rails, and every write would
    // then cross from one EFA device to another on this host: a libfabric issue unrelated to the
    // completion count tested here.
    auto *buf = static_cast<uint8_t *>(allocBuffer(2 * kTotal));
    uint8_t *src = buf;
    uint8_t *dst = buf + kTotal;

    nixl_reg_dlist_t reg(DRAM_SEG);
    reg.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buf), 2 * kTotal, 0, ""));
    nixl_exit_on_failure(agent.registerMem(reg, &extra), "Failed to register memory", kAgent);

    for (int iter = 0; iter < kIters; iter++) {
        const auto pattern = static_cast<uint8_t>(iter % 250 + 1);
        memset(src, pattern, kTotal);
        memset(dst, 0, kTotal);

        nixl_xfer_dlist_t src_descs(DRAM_SEG), dst_descs(DRAM_SEG);
        for (size_t k = 0; k < kDescs; k++) {
            src_descs.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(src + k * kDescSize), kDescSize, 0));
            dst_descs.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(dst + k * kDescSize), kDescSize, 0));
        }

        const string notif = "same-agent-" + to_string(iter);
        extra.notif = notif;
        nixlXferReqH *req = nullptr;
        nixl_exit_on_failure(
            agent.createXferReq(NIXL_WRITE, src_descs, dst_descs, kAgent, req, &extra),
            "Failed to create same-agent transfer",
            kAgent);

        nixl_status_t status = agent.postXferReq(req);
        const auto t0 = steady_clock::now();
        while (status == NIXL_IN_PROG && steady_clock::now() - t0 < kDeadline) {
            this_thread::sleep_for(microseconds(100));
            status = agent.getXferStatus(req);
        }
        nixl_exit_on_failure(status != NIXL_IN_PROG,
                             "Same-agent transfer " + to_string(iter) +
                                 " never reported done: its completion count does not match "
                                 "the writes it posted",
                             kAgent);
        nixl_exit_on_failure(status, "Same-agent transfer " + to_string(iter) + " failed", kAgent);

        // Done must mean every write landed.
        nixl_exit_on_failure(allBytesAre(dst, kTotal, pattern),
                             "Same-agent transfer " + to_string(iter) +
                                 " reported done before all of its writes landed",
                             kAgent);

        nixl_notifs_t notifs;
        bool got_notif = false;
        while (!got_notif && steady_clock::now() - t0 < kDeadline) {
            nixl_exit_on_failure(agent.getNotifs(notifs), "Failed to get notifications", kAgent);
            for (const auto &msg : notifs[kAgent]) {
                got_notif = got_notif || msg == notif;
            }
            if (!got_notif) {
                this_thread::sleep_for(microseconds(100));
            }
        }
        nixl_exit_on_failure(got_notif,
                             "Notification for same-agent transfer " + to_string(iter) +
                                 " never arrived",
                             kAgent);

        // Release right away: the next transfer's handle may reuse this address.
        nixl_exit_on_failure(agent.releaseXferReq(req), "Failed to release transfer", kAgent);
    }

    nixl_exit_on_failure(agent.deregisterMem(reg, &extra), "Failed to deregister memory", kAgent);
    free(buf);

    cout << "libfabric_same_agent_test OK: " << kIters << " same-agent transfers of " << kDescs
         << " x " << kDescSize << " bytes" << endl;
    return 0;
}
