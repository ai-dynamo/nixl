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

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

#include "agent_data.h"
#include "common/nixl_log.h"
#include "common/nixl_time.h"
#include "metadata/etcd_metadata_backend.h"
#include "metadata/p2p_metadata_backend.h"
#include "nixl.h"

const std::string default_metadata_label = "metadata";

namespace nixl::metadata {

namespace {

const char *
requestBackendName(request_backend backend) {
    switch (backend) {
    case request_backend::p2p:
        return "P2P";
    case request_backend::etcd:
        return "ETCD";
    }
    return "unknown";
}

const char *
requestOpName(request_op op) {
    switch (op) {
    case request_op::send:
        return "send";
    case request_op::fetch:
        return "fetch";
    case request_op::invalidate:
        return "invalidate";
    }
    return "unknown";
}

} // namespace

class nixlMetadataStrategy {
public:
    virtual ~nixlMetadataStrategy() = default;

    [[nodiscard]] virtual request_backend
    backend() const noexcept = 0;

    [[nodiscard]] virtual nixl_status_t
    send(const request &req, nixlAgent &agent) = 0;

    [[nodiscard]] virtual nixl_status_t
    fetch(const request &req, nixlAgent &agent) = 0;

    [[nodiscard]] virtual nixl_status_t
    invalidate(const request &req, nixlAgent &agent) = 0;

    virtual void
    progress(nixlAgent &agent) = 0;
};

class p2pMetadataStrategy final : public nixlMetadataStrategy {
public:
    explicit p2pMetadataStrategy(nixlAgentData &agent_data)
        : agent_data_(agent_data),
          backend_(agent_data.config().listenPort, agent_data.name()) {}

    [[nodiscard]] request_backend
    backend() const noexcept override {
        return request_backend::p2p;
    }

    [[nodiscard]] nixl_status_t
    send(const request &req, nixlAgent & /*agent*/) override {
        return backend_.sendToPeer(req.target.peer_ip, req.target.peer_port, req.blob);
    }

    [[nodiscard]] nixl_status_t
    fetch(const request &req, nixlAgent & /*agent*/) override {
        return backend_.requestMetadataFromPeer(req.target.peer_ip, req.target.peer_port);
    }

    [[nodiscard]] nixl_status_t
    invalidate(const request &req, nixlAgent & /*agent*/) override {
        return backend_.invalidatePeerMetadata(req.target.peer_ip,
                                              req.target.peer_port,
                                              agent_data_.name());
    }

    void
    progress(nixlAgent &agent) override {
        backend_.processOnce(agent);
    }

private:
    nixlAgentData &agent_data_;
    nixlP2PMetadataBackend backend_;
};

#if HAVE_ETCD
class etcdMetadataStrategy final : public nixlMetadataStrategy {
public:
    explicit etcdMetadataStrategy(nixlAgentData &agent_data)
        : agent_data_(agent_data),
          backend_(agent_data.name(), agent_data.config().etcdWatchTimeout) {}

    [[nodiscard]] request_backend
    backend() const noexcept override {
        return request_backend::etcd;
    }

    [[nodiscard]] nixl_status_t
    send(const request &req, nixlAgent & /*agent*/) override {
        const std::string key =
            backend_.legacyKey(agent_data_.name(), req.target.metadata_label);
        return backend_.publish(key, req.blob);
    }

    [[nodiscard]] nixl_status_t
    fetch(const request &req, nixlAgent &agent) override {
        nixl_blob_t remote_metadata;
        nixl_status_t ret = backend_.fetchOrWait(req.target.remote_agent,
                                                 req.target.metadata_label,
                                                 remote_metadata);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }

        std::string remote_agent_from_md;
        ret = agent.loadRemoteMD(remote_metadata, remote_agent_from_md);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }
        if (remote_agent_from_md != req.target.remote_agent) {
            NIXL_ERROR << "Metadata mismatch for agent: " << req.target.remote_agent
                       << " from md: " << remote_agent_from_md;
            return NIXL_ERR_MISMATCH;
        }

        NIXL_DEBUG << "Successfully loaded metadata for agent: " << req.target.remote_agent;
        backend_.setupAgentInvalWatcher(req.target.remote_agent);
        return NIXL_SUCCESS;
    }

    [[nodiscard]] nixl_status_t
    invalidate(const request & /*req*/, nixlAgent & /*agent*/) override {
        const std::string agent_prefix = backend_.legacyKey(agent_data_.name(), "");
        return backend_.remove(agent_prefix);
    }

    void
    progress(nixlAgent &agent) override {
        backend_.processInvalidatedAgents(agent);
    }

private:
    nixlAgentData &agent_data_;
    nixlEtcdMetadataBackend backend_;
};
#endif

nixlMetadataManager::nixlMetadataManager(nixlAgentData &agent_data) : agent_data_(agent_data) {
    const nixlAgentConfig &config = agent_data_.config();
    if (config.useListenThread) {
        strategies_.push_back(std::make_unique<p2pMetadataStrategy>(agent_data_));
    }
#if HAVE_ETCD
    if (agent_data_.useEtcd()) {
        try {
            strategies_.push_back(std::make_unique<etcdMetadataStrategy>(agent_data_));
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Error creating ETCD metadata backend: " << e.what();
        }
    }
#endif
}

nixlMetadataManager::~nixlMetadataManager() {
    // Idempotent: if the owning agent didn't call stop() (e.g. it threw out
    // of its constructor before start()), the worker thread isn't joinable
    // and stop() returns immediately.
    stop();
    strategies_.clear();
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
nixlMetadataManager::enqueue(request req) {
    if (shutdown_) {
        NIXL_WARN << "Agent shutting down, unable to accept new requests";
        return;
    }
    const std::lock_guard lock(queue_lock_);
    queue_.push_back(std::move(req));
}

void
nixlMetadataManager::drainInto(std::vector<request> &out) {
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
        std::vector<request> work;

        drainInto(work);

        for (const auto &req : work) {
            auto strategy_it = std::find_if(
                strategies_.begin(), strategies_.end(), [&req](const auto &strategy) {
                    return strategy->backend() == req.target.backend;
                });
            if (strategy_it == strategies_.end()) {
                NIXL_ERROR << "Dropping " << requestBackendName(req.target.backend) << " "
                           << requestOpName(req.op) << ": metadata strategy not enabled";
                continue;
            }

            nixl_status_t ret = NIXL_ERR_UNKNOWN;
            switch (req.op) {
            case request_op::send:
                ret = (*strategy_it)->send(req, agent);
                break;
            case request_op::fetch:
                ret = (*strategy_it)->fetch(req, agent);
                break;
            case request_op::invalidate:
                ret = (*strategy_it)->invalidate(req, agent);
                break;
            }
            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to " << requestOpName(req.op) << " "
                           << requestBackendName(req.target.backend) << " metadata: " << ret;
            }
        }

        for (const auto &strategy : strategies_) {
            strategy->progress(agent);
        }

        const nixlTime::us_t start = nixlTime::getUs();
        while ((start + agent_data_.config().lthrDelay) > nixlTime::getUs()) {
            std::this_thread::yield();
        }
    }
}

} // namespace nixl::metadata
