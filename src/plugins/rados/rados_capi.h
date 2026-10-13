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

#include <cstddef>
#include <cstdint>

#include <rados/librados.h>
#include <rados/librados.hpp>

namespace nixl::rados {

/**
 * librados seam of the rados engine, fake-injectable: reads land
 * directly in caller memory, writes use static, non-owning bufferlists.
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
     * Point the raw handle at an object namespace for all later operations on
     * it; configuration-only, and the engine calls it exactly once, after
     * ioctx_create and before ioctxFromRaw binds the shared view.
     */
    virtual void
    ioctx_set_namespace(rados_ioctx_t ioctx, const char *nspace) = 0;
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
    /**
     * Bind ctx as a non-owning view of raw, closed via ioCtxClose() before ioctx_destroy().
     */
    virtual void
    ioctxFromRaw(rados_ioctx_t raw, librados::IoCtx *ctx) = 0;
    /**
     * Detach the view only; drain every submission made through it first.
     */
    virtual void
    ioCtxClose(librados::IoCtx *ctx) = 0;

    virtual int
    cCreateCompletion(rados_completion_t *pc) = 0;
    /**
     * Submit an async read landing bytes directly in caller memory; keep buf
     * allocated and writable until its completion is released (see cRelease).
     * @return 0 on acceptance; negative errno on rejection, caller releases completion
     */
    virtual int
    cRead(rados_ioctx_t ioctx,
          const char *oid,
          rados_completion_t completion,
          char *buf,
          size_t len,
          uint64_t off) = 0;
    /**
     * Non-blocking poll gating harvest and release; 1 once complete and
     * callback-retired, else 0.
     */
    virtual int
    cIsCompleteAndCb(rados_completion_t completion) = 0;
    /**
     * Result of a retired operation, valid before release: delivered bytes on
     * success, negative errno on failure.
     */
    virtual int
    cGetReturnValue(rados_completion_t completion) = 0;
    virtual int
    cWaitForCompleteAndCb(rados_completion_t completion) = 0;
    /**
     * Release exactly once, only after polled or waited retirement; never reuse.
     */
    virtual void
    cRelease(rados_completion_t completion) = 0;

    virtual int
    cppCreateCompletion(librados::AioCompletion **pc) = 0;
    /**
     * Submit an async range write over bl, a static, non-owning bufferlist: the
     * source must stay allocated, readable, and immutable until its completion
     * is released (see cppRelease); librados reads the payload at flush.
     * @return 0 on acceptance; negative errno on rejection, caller releases completion
     */
    virtual int
    cppWrite(librados::IoCtx *ctx,
             const char *oid,
             librados::AioCompletion *completion,
             const librados::bufferlist &bl,
             size_t len,
             uint64_t off) = 0;
    /**
     * Non-blocking poll gating harvest and release; 1 once complete and
     * callback-retired, else 0.
     */
    virtual int
    cppIsCompleteAndCb(librados::AioCompletion *completion) = 0;
    /**
     * Result of a retired write, valid before release: 0 on success, negative
     * errno on failure.
     */
    virtual int
    cppGetReturnValue(librados::AioCompletion *completion) = 0;
    virtual int
    cppWaitForCompleteAndCb(librados::AioCompletion *completion) = 0;
    /**
     * Release exactly once, only after polled or waited retirement; never reuse.
     */
    virtual void
    cppRelease(librados::AioCompletion *completion) = 0;

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
    ioctx_set_namespace(rados_ioctx_t ioctx, const char *nspace) override {
        rados_ioctx_set_namespace(ioctx, nspace);
    }

    void
    ioctx_destroy(rados_ioctx_t ioctx) override {
        rados_ioctx_destroy(ioctx);
    }

    void
    shutdown(rados_t cluster) override {
        rados_shutdown(cluster);
    }

    void
    ioctxFromRaw(rados_ioctx_t raw, librados::IoCtx *ctx) override {
        librados::IoCtx::from_rados_ioctx_t(raw, *ctx);
    }

    void
    ioCtxClose(librados::IoCtx *ctx) override {
        ctx->close();
    }

    int
    cCreateCompletion(rados_completion_t *pc) override {
        return rados_aio_create_completion(nullptr, nullptr, nullptr, pc);
    }

    int
    cRead(rados_ioctx_t ioctx,
          const char *oid,
          rados_completion_t completion,
          char *buf,
          size_t len,
          uint64_t off) override {
        return rados_aio_read(ioctx, oid, completion, buf, len, off);
    }

    int
    cIsCompleteAndCb(rados_completion_t completion) override {
        return rados_aio_is_complete_and_cb(completion) ? 1 : 0;
    }

    int
    cGetReturnValue(rados_completion_t completion) override {
        return rados_aio_get_return_value(completion);
    }

    int
    cWaitForCompleteAndCb(rados_completion_t completion) override {
        return rados_aio_wait_for_complete_and_cb(completion);
    }

    void
    cRelease(rados_completion_t completion) override {
        rados_aio_release(completion);
    }

    int
    cppCreateCompletion(librados::AioCompletion **pc) override {
        librados::AioCompletion *created = librados::Rados::aio_create_completion();
        if (created == nullptr) {
            return -ENOMEM;
        }
        *pc = created;
        return 0;
    }

    int
    cppWrite(librados::IoCtx *ctx,
             const char *oid,
             librados::AioCompletion *completion,
             const librados::bufferlist &bl,
             size_t len,
             uint64_t off) override {
        return ctx->aio_write(oid, completion, bl, len, off);
    }

    int
    cppIsCompleteAndCb(librados::AioCompletion *completion) override {
        return completion->is_complete_and_cb() ? 1 : 0;
    }

    int
    cppGetReturnValue(librados::AioCompletion *completion) override {
        return completion->get_return_value();
    }

    int
    cppWaitForCompleteAndCb(librados::AioCompletion *completion) override {
        return completion->wait_for_complete_and_cb();
    }

    void
    cppRelease(librados::AioCompletion *completion) override {
        completion->release();
    }
};

} // namespace nixl::rados

#endif
