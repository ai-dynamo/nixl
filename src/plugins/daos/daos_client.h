/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_PLUGINS_DAOS_DAOS_CLIENT_H
#define NIXL_SRC_PLUGINS_DAOS_DAOS_CLIENT_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>

#include "nixl_types.h"

class iDfsObject {
public:
    virtual ~iDfsObject() = default;
};

/**
 * Small interface around libdfs. The concrete implementation owns configurable
 * DAOS event-queue lanes and progress threads; tests can provide a client
 * without a DAOS system.
 */
class iDfsClient {
public:
    using completion_t = std::function<void(int rc, size_t bytes_read)>;

    virtual ~iDfsClient() = default;

    virtual int
    open(std::string_view path, bool write, std::shared_ptr<iDfsObject> &object) = 0;

    /**
     * Submit an asynchronous read. A zero return accepts the operation and
     * must invoke completion exactly once. A nonzero return rejects it and
     * must not invoke completion.
     */
    virtual int
    submitRead(const std::shared_ptr<iDfsObject> &object,
               uintptr_t data_ptr,
               size_t data_len,
               uint64_t offset,
               completion_t completion) = 0;

    /**
     * Submit an asynchronous write with the same single-completion contract
     * as submitRead.
     */
    virtual int
    submitWrite(const std::shared_ptr<iDfsObject> &object,
                uintptr_t data_ptr,
                size_t data_len,
                uint64_t offset,
                completion_t completion) = 0;

    virtual int
    exists(std::string_view path, bool &result) = 0;
};

std::shared_ptr<iDfsClient>
makeLibDfsClient(const nixl_b_params_t *custom_params);

#endif // NIXL_SRC_PLUGINS_DAOS_DAOS_CLIENT_H
