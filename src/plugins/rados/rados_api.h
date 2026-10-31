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
#ifndef NIXL_SRC_PLUGINS_RADOS_RADOS_API_H
#define NIXL_SRC_PLUGINS_RADOS_RADOS_API_H

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <string>

#include <rados/librados.hpp>

namespace nixl::rados {

class radosApi {
public:
    radosApi(const radosApi &) = delete;
    radosApi &
    operator=(const radosApi &) = delete;
    virtual ~radosApi() = default;

    virtual int
    init2(librados::Rados &cluster, const char *name, const char *clustername, uint64_t flags) = 0;
    virtual int
    conf_read_file(librados::Rados &cluster, const char *path) = 0;
    virtual int
    conf_set(librados::Rados &cluster, const char *option, const char *value) = 0;
    virtual int
    connect(librados::Rados &cluster) = 0;
    virtual int
    ioctx_create(librados::Rados &cluster, const char *pool_name, librados::IoCtx &ioctx) = 0;
    /** Select the namespace for later operations; the engine calls it once per client. */
    virtual void
    set_namespace(librados::IoCtx &ioctx, const char *nspace) = 0;

    /** No-throw mirror of librados::Rados::aio_create_completion(); null on allocation failure. */
    virtual librados::AioCompletion *
    createCompletion() noexcept = 0;
    /** Submit a static caller-DRAM read bufferlist; keep it and DRAM writable until release. */
    virtual int
    aioRead(librados::IoCtx *ctx,
            const char *oid,
            librados::AioCompletion *completion,
            librados::bufferlist *bl,
            size_t len,
            uint64_t off) = 0;
    /** Submit an async range write over bl; keep the non-owning source intact until release. */
    virtual int
    aioWrite(librados::IoCtx *ctx,
             const char *oid,
             librados::AioCompletion *completion,
             const librados::bufferlist &bl,
             size_t len,
             uint64_t off) = 0;
    /** Submit an async stat filling psize, prval, and pbl at retirement; -ENOENT when absent. */
    virtual int
    aioStat(librados::IoCtx *ctx,
            const char *oid,
            librados::AioCompletion *completion,
            std::optional<uint64_t> *psize,
            std::optional<int> *prval,
            std::optional<librados::bufferlist> *pbl) = 0;
    // Non-blocking poll gating harvest and release; 1 once complete and callback-retired, else 0.
    virtual int
    isCompleteAndCb(librados::AioCompletion *completion) = 0;
    /** Result of a retired operation, valid before release: read bytes, write 0, or -errno. */
    virtual int
    getReturnValue(librados::AioCompletion *completion) = 0;
    virtual int
    waitForCompleteAndCb(librados::AioCompletion *completion) = 0;
    /** Release exactly once, only after polled or waited retirement; never reuse. */
    virtual void
    release(librados::AioCompletion *completion) = 0;

protected:
    radosApi() = default;
};

class realRadosApi final : public radosApi {
public:
    int
    init2(librados::Rados &cluster,
          const char *name,
          const char *clustername,
          uint64_t flags) override {
        return cluster.init2(name, clustername, flags);
    }

    int
    conf_read_file(librados::Rados &cluster, const char *path) override {
        return cluster.conf_read_file(path);
    }

    int
    conf_set(librados::Rados &cluster, const char *option, const char *value) override {
        return cluster.conf_set(option, value);
    }

    int
    connect(librados::Rados &cluster) override {
        return cluster.connect();
    }

    int
    ioctx_create(librados::Rados &cluster, const char *pool_name, librados::IoCtx &ioctx) override {
        return cluster.ioctx_create(pool_name, ioctx);
    }

    void
    set_namespace(librados::IoCtx &ioctx, const char *nspace) override {
        ioctx.set_namespace(nspace == nullptr ? std::string{} : std::string{nspace});
    }

    librados::AioCompletion *
    createCompletion() noexcept override {
        try {
            return librados::Rados::aio_create_completion();
        }
        catch (const std::bad_alloc &) {
            // Throwing after allocating can strand an inert librados-side completion upstream.
            return nullptr;
        }
    }

    int
    aioRead(librados::IoCtx *ctx,
            const char *oid,
            librados::AioCompletion *completion,
            librados::bufferlist *bl,
            size_t len,
            uint64_t off) override {
        return ctx->aio_read(oid, completion, bl, len, off);
    }

    int
    aioWrite(librados::IoCtx *ctx,
             const char *oid,
             librados::AioCompletion *completion,
             const librados::bufferlist &bl,
             size_t len,
             uint64_t off) override {
        return ctx->aio_write(oid, completion, bl, len, off);
    }

    int
    aioStat(librados::IoCtx *ctx,
            const char *oid,
            librados::AioCompletion *completion,
            std::optional<uint64_t> *psize,
            std::optional<int> *prval,
            std::optional<librados::bufferlist> *pbl) override {
        if (psize == nullptr || prval == nullptr || pbl == nullptr) {
            return -EINVAL;
        }
        psize->emplace(0);
        prval->emplace(0);
        pbl->emplace();
        librados::ObjectReadOperation op;
        op.stat(&**psize, nullptr, &**prval);
        return ctx->aio_operate(oid, completion, &op, &**pbl);
    }

    int
    isCompleteAndCb(librados::AioCompletion *completion) override {
        return completion->is_complete_and_cb() ? 1 : 0;
    }

    int
    getReturnValue(librados::AioCompletion *completion) override {
        return completion->get_return_value();
    }

    int
    waitForCompleteAndCb(librados::AioCompletion *completion) override {
        return completion->wait_for_complete_and_cb();
    }

    void
    release(librados::AioCompletion *completion) override {
        completion->release();
    }
};

} // namespace nixl::rados

#endif
