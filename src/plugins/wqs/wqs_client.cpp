// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "wqs_client.h"
#include "wqs_session.h"

#include <shared_client.h>

#include <mutex>
#include <stdexcept>

namespace {
class wqsSharedClient final : public iWqsClient {
public:
    wqsSharedClient(wqs::SharedClient &client, const nixlWqsConfig &config, nixlWqsBufferPool &pool)
        : client_(client) {
        if (client_.setup("", "", 0, 0, "", "", config.unixPath) != 0 ||
            client_.register_buffer_with_fd(pool.data(), pool.size(), pool.fd()) != 0) {
            throw std::runtime_error("WQS setup or staging buffer registration failed");
        }
    }

    int64_t
    getSize(const std::string &key) override {
        return client_.getSize(key);
    }

    std::vector<int>
    read(const std::vector<std::string> &keys,
         const std::vector<void *> &buffers,
         const std::vector<size_t> &sizes) override {
        return client_.batch_get_into(keys, buffers, sizes);
    }

    std::vector<int>
    write(const std::vector<std::string> &keys,
          const std::vector<void *> &buffers,
          const std::vector<size_t> &sizes) override {
        return client_.batch_put_from(keys, buffers, sizes);
    }

private:
    wqs::SharedClient &client_;
};

struct wqsProcessSession {
    std::mutex mutex;
    bool attempted = false;
    std::shared_ptr<nixlWqsBufferPool> pool;
    std::unique_ptr<wqs::SharedClient> sdkClient;
    std::shared_ptr<wqsSharedClient> client;
    std::shared_ptr<nixlWqsSession> session;
};
} // namespace

const std::shared_ptr<nixlWqsSession> &
getWqsSession(const nixlWqsConfig &config) {
    // WQS cannot unregister its buffer or safely reinitialize SPDK. Keep the session until
    // process exit, including failed initialization; NIXL loads plugins with RTLD_NODELETE.
    static auto *process = new wqsProcessSession;
    const std::lock_guard lock(process->mutex);
    if (process->session) {
        if (!config.sameSession(process->session->config())) {
            throw std::invalid_argument("WQS backends in one process must share session settings");
        }
        return process->session;
    }
    if (process->attempted) {
        throw std::runtime_error("WQS initialization previously failed; restart the process");
    }
    process->pool = std::make_shared<nixlWqsBufferPool>(config.stagingPoolBytes);
    process->attempted = true;
    // Retain the SDK even if the wrapper's constructor fails after starting SPDK.
    process->sdkClient = std::make_unique<wqs::SharedClient>();
    process->client =
        std::make_shared<wqsSharedClient>(*process->sdkClient, config, *process->pool);
    process->session = std::make_shared<nixlWqsSession>(config, process->client, process->pool);
    return process->session;
}
