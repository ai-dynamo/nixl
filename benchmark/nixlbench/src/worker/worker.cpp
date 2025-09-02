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

#include "worker.h"
#include "runtime/etcd/etcd_rt.h"
#include "utils/utils.h"

#include <unistd.h>

static xferBenchRT *createRT(int *terminate) {
    if (XFERBENCH_RT_ETCD == xferBenchConfig::runtime_type) {
        int total = 2;
        if (XFERBENCH_MODE_SG == xferBenchConfig::mode) {
            total = xferBenchConfig::num_initiator_dev +
                xferBenchConfig::num_target_dev;
        }
        if (xferBenchConfig::isStorageBackend()) {
            total = 1;
        }
        xferBenchEtcdRT *etcd_rt = new xferBenchEtcdRT(
            xferBenchConfig::benchmark_group, xferBenchConfig::etcd_endpoints, total, terminate);
        if (etcd_rt->setup() != 0) {
            std::cerr << "Failed to setup ETCD runtime" << std::endl;
            delete etcd_rt;
            exit (EXIT_FAILURE);
        }
        return etcd_rt;
    }

    std::cerr << "Invalid runtime: " << xferBenchConfig::runtime_type << std::endl;
    exit(EXIT_FAILURE);
}

int xferBenchWorker::synchronize() {
    if (rt->barrier("sync") != 0) {
        std::cerr << "Failed to synchronize" << std::endl;
        // assuming this is a fatal error, continue benchmarking after synchronization failure does
        // not make sense
        exit(EXIT_FAILURE);
    }

    return 0;
}

xferBenchWorker::xferBenchWorker(int *argc, char ***argv) {
    terminate = 0;

    rt = createRT(&terminate);
    if (!rt) {
        std::cerr << "Failed to create runtime object" << std::endl;
        exit(EXIT_FAILURE);
    }

    int rank = rt->getRank();

    if (XFERBENCH_MODE_SG == xferBenchConfig::mode) {
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

int xferBenchWorker::terminate = 0;

void xferBenchWorker::signalHandler(int signal) {
    static const char msg[] = "Ctrl-C received, exiting...\n";
    constexpr int stdout_fd = 1;
    constexpr int max_count = 1;
    auto size = write(stdout_fd, msg, sizeof(msg) - 1);
    (void)size;

    if (++terminate > max_count) {
        std::_Exit(EXIT_FAILURE);
    }
}
