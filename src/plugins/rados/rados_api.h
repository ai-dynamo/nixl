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

#include <rados/librados.h>
#include <rados/librados.hpp>

namespace nixl::rados {

/** librados seam: reconciles read output; writes use static non-owning bufferlists. */
class radosApi {
public:
    radosApi(const radosApi &) = delete;
    radosApi &
    operator=(const radosApi &) = delete;
    virtual ~radosApi() = default;

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
     * Point the raw handle at an object namespace for all later operations on
     * it; configuration-only, and the engine calls it exactly once, after
     * ioctx_create and before ioctxFromRaw binds the shared view.
     */
    virtual void
    ioctx_set_namespace(rados_ioctx_t ioctx, const char *nspace) = 0;
    /**
     * Bind ctx as a non-owning view of raw, closed via ioctxClose() before ioctx_destroy().
     */
    virtual void
    ioctxFromRaw(rados_ioctx_t raw, librados::IoCtx *ctx) = 0;
    /**
     * Detach the view only; drain every submission made through it first.
     */
    virtual void
    ioctxClose(librados::IoCtx *ctx) = 0;
    /**
     * Destroy the raw handle exactly once, after its views close.
     */
    virtual void
    ioctx_destroy(rados_ioctx_t ioctx) = 0;
    /**
     * Shut down the cluster handle exactly once, after all I/O drains.
     */
    virtual void
    shutdown(rados_t cluster) = 0;

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
    /**
     * Submit an async range write over bl, a static, non-owning bufferlist: the
     * source must stay allocated, readable, and immutable until its completion
     * is released (see release); librados reads the payload at flush.
     * @return 0 on acceptance; negative errno on rejection, caller releases completion
     */
    virtual int
    aioWrite(librados::IoCtx *ctx,
             const char *oid,
             librados::AioCompletion *completion,
             const librados::bufferlist &bl,
             size_t len,
             uint64_t off) = 0;
    /**
     * Submit an async single-stat read op through ctx; the seam engages all three cells,
     * then librados writes the object size into *psize and the per-op result into *prval
     * at retirement. psize, prval, and pbl are caller-owned and must stay allocated and
     * stable until the completion is released (see release); pbl receives the merged
     * read output, empty for a stat-only op but still dereferenced at completion.
     * On an absent object the completion return value is -ENOENT while the untouched
     * per-op slot surfaces -EIO.
     * @return 0 on acceptance; negative errno on rejection, caller releases completion
     */
    virtual int
    aioStat(librados::IoCtx *ctx,
            const char *oid,
            librados::AioCompletion *completion,
            std::optional<uint64_t> *psize,
            std::optional<int> *prval,
            std::optional<librados::bufferlist> *pbl) = 0;
    /**
     * Non-blocking poll gating harvest and release; 1 once complete and
     * callback-retired, else 0.
     */
    virtual int
    isCompleteAndCb(librados::AioCompletion *completion) = 0;
    /** Result of a retired operation, valid before release: read bytes, write 0, or -errno. */
    virtual int
    getReturnValue(librados::AioCompletion *completion) = 0;
    virtual int
    waitForCompleteAndCb(librados::AioCompletion *completion) = 0;
    /**
     * Release exactly once, only after polled or waited retirement; never reuse.
     */
    virtual void
    release(librados::AioCompletion *completion) = 0;

protected:
    radosApi() = default;
};

class realRadosApi final : public radosApi {
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
    ioctx_set_namespace(rados_ioctx_t ioctx, const char *nspace) override {
        rados_ioctx_set_namespace(ioctx, nspace);
    }

    void
    ioctxFromRaw(rados_ioctx_t raw, librados::IoCtx *ctx) override {
        librados::IoCtx::from_rados_ioctx_t(raw, *ctx);
    }

    void
    ioctxClose(librados::IoCtx *ctx) override {
        ctx->close();
    }

    void
    ioctx_destroy(rados_ioctx_t ioctx) override {
        rados_ioctx_destroy(ioctx);
    }

    void
    shutdown(rados_t cluster) override {
        rados_shutdown(cluster);
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
