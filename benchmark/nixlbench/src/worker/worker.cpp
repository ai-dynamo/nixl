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

#include <sstream>
#include <unistd.h>

namespace nixlbench {

bool
usesNullRuntime(const benchmarkConfig &config) {
    return isStorageBackend(config.backend) && config.runtime.etcd_endpoints.empty();
}

int
runtimeWorldSize(const benchmarkConfig &config) {
    if (isStorageBackend(config.backend)) {
        return 1;
    }
    if (config.transfer.mode == XFERBENCH_MODE_SG) {
        return config.worker.num_initiator_dev + config.worker.num_target_dev;
    }
    return 2;
}

std::string
rankRoleName(const benchmarkConfig &config, int rank) {
    if (usesNullRuntime(config)) {
        return "initiator";
    }
    if (config.transfer.mode == XFERBENCH_MODE_SG) {
        if (rank >= 0 && rank < config.worker.num_initiator_dev) {
            return "initiator";
        }
        return "target";
    }
    if (config.transfer.mode == XFERBENCH_MODE_MG) {
        if (0 == rank) {
            return "initiator";
        }
        return "target";
    }
    return "";
}

std::vector<std::string>
parseWorkerDeviceList(const benchmarkConfig &config) {
    std::vector<std::string> devices;
    std::string dev;
    std::stringstream ss(config.worker.device_list);

    // TODO: Add support for other schemes
    if (config.transfer.scheme == XFERBENCH_SCHEME_PAIRWISE && config.worker.device_list != "all") {
        while (std::getline(ss, dev, ',')) {
            devices.push_back(dev);
        }

        if ((int)devices.size() != config.worker.num_initiator_dev ||
            (int)devices.size() != config.worker.num_target_dev) {
            std::cerr << "Incorrect device list " << config.worker.device_list
                      << " provided for pairwise scheme " << devices.size() << "# devices"
                      << std::endl;
            return {};
        }
    } else {
        devices.push_back("all");
    }

    return devices;
}

} // namespace nixlbench

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

static xferBenchRT *createRT(const nixlbench::benchmarkConfig &config, int *terminate) {
    // For storage backends without ETCD endpoints, use null runtime
    if (nixlbench::usesNullRuntime(config)) {
        std::cout << "Using null runtime for storage backend without ETCD" << std::endl;
        return new xferBenchNullRT();
    }

#if HAVE_ETCD
    if (XFERBENCH_RT_ETCD == config.runtime.type) {
        int total = nixlbench::runtimeWorldSize(config);
        xferBenchEtcdRT *etcd_rt = new xferBenchEtcdRT(
            config.common.benchmark_group, config.runtime.etcd_endpoints, total, terminate);
        if (etcd_rt->setup() != 0) {
            std::cerr << "Failed to setup ETCD runtime" << std::endl;
            delete etcd_rt;
            exit (EXIT_FAILURE);
        }
        return etcd_rt;
    }
#endif

    std::cerr << "Invalid runtime: " << config.runtime.type << std::endl;
    exit(EXIT_FAILURE);
}

int xferBenchWorker::synchronize() {
    // For storage backends without ETCD, no synchronization needed
    if (nixlbench::usesNullRuntime(benchmark_config)) {
        return 0;
    }

    if (rt->barrier("sync") != 0) {
        std::cerr << "Failed to synchronize" << std::endl;
        // assuming this is a fatal error, continue benchmarking after synchronization failure does
        // not make sense
        exit(EXIT_FAILURE);
    }

    return 0;
}

xferBenchWorker::xferBenchWorker(const nixlbench::benchmarkConfig &config)
    : benchmark_config(config), config(nixlbench::makeLegacyConfigFromBenchmarkConfig(config)) {
    terminate = 0;

    rt = createRT(config, &terminate);
    if (!rt) {
        std::cerr << "Failed to create runtime object" << std::endl;
        exit(EXIT_FAILURE);
    }

    int rank = rt->getRank();

    name = nixlbench::rankRoleName(config, rank);

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
