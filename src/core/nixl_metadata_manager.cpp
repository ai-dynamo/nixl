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
#include "nixl_metadata_manager.h"

#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

#include "agent_data.h"
#include "backends/etcd_metadata_backend.h"
#include "backends/p2p_metadata_backend.h"
#include "common/nixl_log.h"
#include "common/nixl_time.h"
#include "nixl.h"

const std::string default_metadata_label = "metadata";

nixlMetadataManager::nixlMetadataManager(nixlAgentData &agent_data) : agent_data_(agent_data) {
    const nixlAgentConfig &config = agent_data_.config();
    if (config.useListenThread) {
        auto p2p = std::make_unique<nixlP2PMetadataBackend>(config.listenPort, agent_data_.name());
        p2p_ = p2p.get();
        p2p_->setupListener();
        backends_.push_back(std::move(p2p));
    }
#if HAVE_ETCD
    if (agent_data_.useEtcd()) {
        auto etcd =
            std::make_unique<nixlEtcdMetadataBackend>(agent_data_.name(), config.etcdWatchTimeout);
        etcd_ = etcd.get();
        backends_.push_back(std::move(etcd));
    }
#endif
}

nixlMetadataManager::~nixlMetadataManager() {
    // Idempotent: if the owning agent didn't call stop() (e.g. it threw out
    // of its constructor before start()), the worker thread isn't joinable
    // and stop() returns immediately.
    stop();
    backends_.clear();
}

void
nixlMetadataManager::start(nixlAgent &agent) {
    stop_ = false;
    shutdown_ = false;
    worker_failed_ = false;
    worker_exception_ = nullptr;
    worker_ = std::thread(&nixlMetadataManager::runLoopNoexcept, this, std::ref(agent));
}

void
nixlMetadataManager::stop() {
    if (!worker_.joinable()) {
        return;
    }

    shutdown_ = true;

    while (!queueEmpty() && !worker_failed_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    stop_ = true;
    worker_.join();

    try {
        if (worker_exception_) {
            std::rethrow_exception(worker_exception_);
        }
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Metadata manager worker thread terminated with an exception: " << e.what();
    }
}

void
nixlMetadataManager::enqueue(nixl::md::request req) {
    if (shutdown_) {
        NIXL_WARN << "Agent shutting down, unable to accept new requests";
        return;
    }
    const std::lock_guard lock(queue_lock_);
    queue_.push_back(std::move(req));
}

void
nixlMetadataManager::drainInto(std::vector<nixl::md::request> &out) {
    const std::lock_guard lock(queue_lock_);
    out = std::move(queue_);
    queue_.clear();
}

bool
nixlMetadataManager::queueEmpty() const {
    const std::lock_guard lock(queue_lock_);
    return queue_.empty();
}

void
nixlMetadataManager::runLoopNoexcept(nixlAgent &agent) noexcept {
    try {
        runLoop(agent);
    }
    catch (...) {
        worker_exception_ = std::current_exception();
        worker_failed_ = true;
    }
}

void
nixlMetadataManager::recordPeerUuid(const std::string &agent_name, const std::string &agent_uuid) {
    const std::lock_guard lock(cache_lock_);
    name_to_uuid_[agent_name] = agent_uuid;
}

void
nixlMetadataManager::runLoop(nixlAgent &agent) {
    while (!stop_) {
        std::vector<nixl::md::request> work;

        drainInto(work);

        for (const auto &req : work) {
            switch (req.kind) {
            case nixl::md::request_kind::p2p_send:
                if (p2p_) {
                    const nixl_status_t ret = p2p_->sendToPeer(req.str1, req.port, req.blob);
                    if (ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "Failed to send P2P metadata to " << req.str1 << ":"
                                   << req.port << ": " << ret;
                    }
                }
                break;
            case nixl::md::request_kind::p2p_fetch:
                if (p2p_) {
                    const nixl_status_t ret = p2p_->requestMetadataFromPeer(req.str1, req.port);
                    if (ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "Failed to request P2P metadata from " << req.str1 << ":"
                                   << req.port << ": " << ret;
                    }
                }
                break;
            case nixl::md::request_kind::p2p_inval:
                if (p2p_) {
                    const nixl_status_t ret =
                        p2p_->invalidatePeerMetadata(req.str1, req.port, agent_data_.name());
                    if (ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "Failed to invalidate P2P metadata on " << req.str1 << ":"
                                   << req.port << ": " << ret;
                    }
                }
                break;
#if HAVE_ETCD
            case nixl::md::request_kind::etcd_send: {
                if (!etcd_) {
                    NIXL_ERROR << "Dropping etcd_send: ETCD backend not enabled";
                    break;
                }
                const std::string key = etcd_->legacyKey(agent_data_.name(), req.str1);
                const nixl_status_t ret = etcd_->publish(key, req.blob);
                if (ret != NIXL_SUCCESS) {
                    NIXL_ERROR << "Failed to store metadata in etcd: " << ret;
                }
                break;
            }
            case nixl::md::request_kind::etcd_fetch: {
                if (!etcd_) {
                    NIXL_ERROR << "Dropping etcd_fetch: ETCD backend not enabled";
                    break;
                }
                const std::string &metadata_label = req.str1;
                const std::string &remote_agent = req.blob;

                nixl_blob_t remote_metadata;
                nixl_status_t ret =
                    etcd_->fetchOrWait(remote_agent, metadata_label, remote_metadata);
                if (ret != NIXL_SUCCESS) {
                    NIXL_ERROR << "Failed to fetch metadata from etcd: " << ret;
                    break;
                }

                std::string remote_agent_from_md;
                ret = agent.loadRemoteMD(remote_metadata, remote_agent_from_md);
                if (ret != NIXL_SUCCESS) {
                    NIXL_ERROR << "Failed to load remote metadata: " << ret;
                    break;
                }
                if (remote_agent_from_md != remote_agent) {
                    NIXL_ERROR << "Metadata mismatch for agent: " << remote_agent
                               << " from md: " << remote_agent_from_md;
                    break;
                }
                NIXL_DEBUG << "Successfully loaded metadata for agent: " << remote_agent;
                etcd_->setupAgentInvalWatcher(remote_agent);
                break;
            }
            case nixl::md::request_kind::etcd_inval: {
                if (!etcd_) {
                    NIXL_ERROR << "Dropping etcd_inval: ETCD backend not enabled";
                    break;
                }
                const std::string agent_prefix = etcd_->legacyKey(agent_data_.name(), "");
                const nixl_status_t ret = etcd_->remove(agent_prefix);
                if (ret != NIXL_SUCCESS) {
                    NIXL_ERROR << "Failed to invalidate metadata in etcd: " << ret;
                }
                break;
            }
#else
            case nixl::md::request_kind::etcd_send:
            case nixl::md::request_kind::etcd_fetch:
            case nixl::md::request_kind::etcd_inval:
                NIXL_ERROR << "Dropping ETCD request: ETCD support is not compiled in";
                break;
#endif // HAVE_ETCD
            }
        }

        if (p2p_) {
            p2p_->processOnce(agent);
        }

#if HAVE_ETCD
        if (etcd_) {
            etcd_->processInvalidatedAgents(agent);
        }
#endif

        const nixlTime::us_t start = nixlTime::getUs();
        while ((start + agent_data_.config().lthrDelay) > nixlTime::getUs()) {
            std::this_thread::yield();
        }
    }
}
