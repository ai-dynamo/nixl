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

static xferBenchRT *createRT(int *argc, char ***argv) {
    if (XFERBENCH_RT_ETCD == xferBenchConfig::runtime_type) {
        int total = 2;
        if (XFERBENCH_MODE_SG == xferBenchConfig::mode) {
            total = xferBenchConfig::num_initiator_dev +
                xferBenchConfig::num_target_dev;
        }
        if (XFERBENCH_BACKEND_GDS == xferBenchConfig::backend) {
            total = 1;
        }
        return new xferBenchEtcdRT(xferBenchConfig::etcd_endpoints, total);
    }
    std::cerr << "Invalid runtime: " << xferBenchConfig::runtime_type << std::endl;
    exit(EXIT_FAILURE);
    return nullptr;
}

xferBenchWorker::xferBenchWorker(int *argc, char ***argv) {
    int rank;

    // Create the RT
    rt = createRT(argc, argv);

    if (!rt) {
	std::cerr << "Failed to create runtime object" << std::endl;
	exit(EXIT_FAILURE);
    }

    rank = rt->getRank();

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

bool xferBenchWorker::isInitiator() {
    return ("initiator" == name);
}

bool xferBenchWorker::isTarget() {
    return ("target" == name);
}
