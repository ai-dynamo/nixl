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
/**
 * @file nixl_md_config.h
 * @brief Temporary carrier for the backend settings that still live in nixlAgentConfig.
 */
#ifndef NIXL_SRC_CORE_NIXL_MD_CONFIG_H
#define NIXL_SRC_CORE_NIXL_MD_CONFIG_H

#include <chrono>
#include <cstdint>

/**
 * @struct nixlMDConfig
 * @brief Backend-specific settings carved out of nixlAgentConfig so a backend
 *        never reaches the public nixl_params.h.
 *
 * Temporary. Every field here is backend-specific and only travels through the
 * manager because it arrives as a public agent setting; the direction is for a
 * backend to read its own parameters from utils/common/configuration.h and for
 * these to leave nixlAgentConfig on the next ABI/API breaking update, at which
 * point this struct goes away. Only the three current backends (P2P, ETCD,
 * TCPStore) take it; the nixlMetadataBackend contract does not.
 *
 * Passed to the manager and on to each backend at construction; every field is
 * fixed for the life of the agent.
 */
struct nixlMDConfig {
    /** P2P: listen for inbound peers. */
    bool useListenThread = false;
    /** P2P: port the listener binds. */
    std::uint16_t listenPort = 0;
    /** ETCD: how long a fetch waits on a watch for a key to appear. */
    std::chrono::microseconds etcdWatchTimeout{0};
    /** How long a backend worker waits for work before polling anyway. */
    std::chrono::microseconds workerDelay{0};
};

#endif // NIXL_SRC_CORE_NIXL_MD_CONFIG_H
