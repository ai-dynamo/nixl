/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "worker.h"
#include "runtime/asio_runtime.h"
#include "runtime/etcd/etcd_rt.h"
#include "utils/utils.h"

#include <unistd.h>

// Null runtime for storage backends that don't need ETCD
class xferBenchNullRT : public xferBenchRT {
public:
    xferBenchNullRT() {
        setSize(1);
        setRank(0);
    }

    virtual ~xferBenchNullRT() {}

    virtual int
    sendInt(int *buffer, int dest_rank) override {
        return 0;
    }

    virtual int
    recvInt(int *buffer, int src_rank) override {
        return 0;
    }

    virtual int
    broadcastInt(int *buffer, size_t count, int root_rank) override {
        return 0;
    }

    virtual int
    sendChar(char *buffer, size_t count, int dest_rank) override {
        return 0;
    }

    virtual int
    recvChar(char *buffer, size_t count, int src_rank) override {
        return 0;
    }

    virtual int
    reduceSumDouble(double *local_value, double *global_value, int dest_rank) override {
        *global_value = *local_value;
        return 0;
    }

    virtual int
    barrier(const std::string &barrier_id) override {
        return 0;
    }
};

namespace {

[[nodiscard]] int
determineTotal() {
    int total = 2;
    if (XFERBENCH_MODE_SG == xferBenchConfig::mode) {
        total = xferBenchConfig::num_initiator_dev + xferBenchConfig::num_target_dev;
    }
    if (xferBenchConfig::isStorageBackend()) {
        total = 1;
    }
    return total;
}

static xferBenchRT *
createRT(std::atomic<int> *terminate) {
    // For storage backends without ETCD endpoints, use null runtime
    if (xferBenchConfig::isStorageBackend() && xferBenchConfig::etcd_endpoints.empty()) {
        std::cout << "Using null runtime for storage backend without ETCD" << std::endl;
        return new xferBenchNullRT();
    }

    const int total = determineTotal();

#if HAVE_ETCD
    if (XFERBENCH_RT_ETCD == xferBenchConfig::runtime_type) {
        xferBenchEtcdRT *etcd_rt = new xferBenchEtcdRT(
            xferBenchConfig::benchmark_group, xferBenchConfig::etcd_endpoints, total, terminate);
        if (etcd_rt->setup() != 0) {
            std::cerr << "Failed to setup ETCD runtime" << std::endl;
            delete etcd_rt;
            exit (EXIT_FAILURE);
        }
        return etcd_rt;
    }
#endif

    if (xferBenchConfig::runtime_type == XFERBENCH_RT_ASIO) {
        if (total != 2) {
            std::cerr << "Invalid total " << total << " for ASIO runtime -- supports only 2"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        return new xferBenchAsioRT(xferBenchConfig::asio_address, xferBenchConfig::asio_port);
    }

    std::cerr << "Invalid runtime: " << xferBenchConfig::runtime_type << std::endl;
    exit(EXIT_FAILURE);
}

} // namespace

int
xferBenchWorker::synchronize() {
    if (rt->barrier("sync") != 0) {
        std::cerr << "Failed to synchronize" << std::endl;
        // Best-effort cleanup of non-leased etcd keys (e.g. the "size" key)
        // so they don't poison subsequent runs in the same benchmark_group.
        rt->cleanupForExit();
        // Use _Exit() instead of exit() to bypass atexit handlers (e.g. gRPC shutdown).
        // exit() would deadlock with the etcd KeepAlive background thread: gRPC shutdown
        // waits for open streams to close, but the KeepAlive thread keeps renewing the
        // lease stream indefinitely. _Exit() kills all threads immediately, which closes
        // the gRPC stream and lets the lease expire on the etcd server side.
        std::_Exit(EXIT_FAILURE);
    }

    return 0;
}

xferBenchWorker::xferBenchWorker() {
    terminate = 0;

    rt = createRT(&terminate);
    if (!rt) {
        std::cerr << "Failed to create runtime object" << std::endl;
        exit(EXIT_FAILURE);
    }

    int rank = rt->getRank();

    // For storage backends without ETCD endpoints always act as initiator
    if (xferBenchConfig::isStorageBackend() && xferBenchConfig::etcd_endpoints.empty() &&
        (xferBenchConfig::runtime_type == XFERBENCH_RT_ETCD)) {
        name = "initiator";
    } else if (XFERBENCH_MODE_SG == xferBenchConfig::mode) {
        if (rank >= 0 && rank < xferBenchConfig::num_initiator_dev) {
            name = "initiator";
        } else {
            name = "target";
        }
    } else if (XFERBENCH_MODE_MG == xferBenchConfig::mode) {
        if (0 == rank) {
            name = "initiator";
        } else {
            name = "target";
        }
    }

    // Set the RT for utils
    xferBenchUtils::setRT(rt);
}

xferBenchWorker::~xferBenchWorker() {
    delete rt;
}

std::string xferBenchWorker::getName() const {
    return name;
}

bool xferBenchWorker::isMasterRank() {
    return (0 == rt->getRank());
}

bool xferBenchWorker::isInitiator() {
    return ("initiator" == name);
}

bool xferBenchWorker::isTarget() {
    return ("target" == name);
}

static_assert(std::atomic<int>::is_always_lock_free,
              "xferBenchWorker::terminate must be lock-free for safe use in signal handlers");

std::atomic<int> xferBenchWorker::terminate = 0;

void xferBenchWorker::signalHandler(int signal) {
    static const char msg[] = "Ctrl-C received, exiting...\n";
    constexpr int stdout_fd = 1;
    constexpr int max_count = 1;
    auto size = write(stdout_fd, msg, sizeof(msg) - 1);
    (void)size;

    if (terminate.fetch_add(1) >= max_count) {
        std::_Exit(EXIT_FAILURE);
    }
}
