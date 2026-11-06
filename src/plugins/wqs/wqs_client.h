// SPDX-FileCopyrightText: Copyright (c) 2026 Expontech. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef NIXL_SRC_PLUGINS_WQS_WQS_CLIENT_H
#define NIXL_SRC_PLUGINS_WQS_WQS_CLIENT_H

#include "nixl_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct nixlWqsConfig {
    std::string unixPath = "/var/run/wds/wqs";
    size_t stagingPoolBytes = size_t{1} << 30;
    size_t maxInflight = 64;
    size_t maxBatchEntries = 64;
    size_t payloadBytes = 8192;
    std::string keyPrefix;

    [[nodiscard]] static nixlWqsConfig
    fromParams(const nixl_b_params_t &params);
    [[nodiscard]] bool
    sameSession(const nixlWqsConfig &other) const;
};

/** Synchronous WQS calls serialized by the session worker; injectable for tests. */
class iWqsClient {
public:
    iWqsClient() = default;
    virtual ~iWqsClient() = default;
    iWqsClient(const iWqsClient &) = delete;
    iWqsClient &
    operator=(const iWqsClient &) = delete;
    iWqsClient(iWqsClient &&) = delete;
    iWqsClient &
    operator=(iWqsClient &&) = delete;

    [[nodiscard]] virtual int64_t
    getSize(const std::string &key) = 0;
    [[nodiscard]] virtual std::vector<int>
    read(const std::vector<std::string> &keys,
         const std::vector<void *> &buffers,
         const std::vector<size_t> &sizes) = 0;
    [[nodiscard]] virtual std::vector<int>
    write(const std::vector<std::string> &keys,
          const std::vector<void *> &buffers,
          const std::vector<size_t> &sizes) = 0;
};

class nixlWqsSession;
[[nodiscard]] const std::shared_ptr<nixlWqsSession> &
getWqsSession(const nixlWqsConfig &config);

#endif // NIXL_SRC_PLUGINS_WQS_WQS_CLIENT_H
