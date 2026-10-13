/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Mark Nelson.
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
#ifndef NIXL_SRC_PLUGINS_RADOS_RADOS_CAPI_H
#define NIXL_SRC_PLUGINS_RADOS_RADOS_CAPI_H

#include <cstdint>

#include <rados/librados.h>

namespace nixl::rados {

/**
 * librados C lifecycle/configuration seam of the rados engine,
 * fake-injectable for the hermetic tests.
 */
class cApi {
public:
    cApi(const cApi &) = delete;
    cApi &
    operator=(const cApi &) = delete;
    virtual ~cApi() = default;

    virtual int
    create2(rados_t *cluster, const char *clustername, const char *name, uint64_t flags) = 0;
    virtual int
    conf_read_file(rados_t cluster, const char *path) = 0;
    virtual int
    conf_set(rados_t cluster, const char *option, const char *value) = 0;
    virtual int
    connect(rados_t cluster) = 0;
    /**
     * Open a pool into a new raw I/O context handle owned by the caller;
     * 0 or negative errno.
     */
    virtual int
    ioctx_create(rados_t cluster, const char *pool_name, rados_ioctx_t *ioctx) = 0;
    /**
     * Destroy the raw handle exactly once, after all I/O drains.
     */
    virtual void
    ioctx_destroy(rados_ioctx_t ioctx) = 0;
    /**
     * Shut down the cluster handle exactly once, after all I/O drains.
     */
    virtual void
    shutdown(rados_t cluster) = 0;

protected:
    cApi() = default;
};

class libRados final : public cApi {
public:
    int
    create2(rados_t *cluster, const char *clustername, const char *name, uint64_t flags) override {
        return rados_create2(cluster, clustername, name, flags);
    }

    int
    conf_read_file(rados_t cluster, const char *path) override {
        return rados_conf_read_file(cluster, path);
    }

    int
    conf_set(rados_t cluster, const char *option, const char *value) override {
        return rados_conf_set(cluster, option, value);
    }

    int
    connect(rados_t cluster) override {
        return rados_connect(cluster);
    }

    int
    ioctx_create(rados_t cluster, const char *pool_name, rados_ioctx_t *ioctx) override {
        return rados_ioctx_create(cluster, pool_name, ioctx);
    }

    void
    ioctx_destroy(rados_ioctx_t ioctx) override {
        rados_ioctx_destroy(ioctx);
    }

    void
    shutdown(rados_t cluster) override {
        rados_shutdown(cluster);
    }
};

} // namespace nixl::rados

#endif
