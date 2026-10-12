/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rdmaRetry.h"

#include <algorithm>
#include <string>
#include <vector>

#include "common/nixl_log.h"

namespace nixl_obj_rdma {

/**
 * Retry wrappers (token lifecycle + one transient retry). A token-mint failure
 * is itself transient (cuObject NIC selection / registration hiccup), so it is
 * retried rather than aborting on the first attempt.
 */
ssize_t
rdmaPutWithRetry(SharedCuObjClient &rdma,
                 S3RdmaControlPlane &cp,
                 S3RdmaClientCtx &ctx,
                 void *buf,
                 uint64_t size) {
    ssize_t ret = -1;
    for (int attempt = 0; attempt < rdma_max_attempts; ++attempt) {
        char *token = rdma.getToken(buf, size, CUOBJ_PUT);
        if (token == nullptr) {
            ret = -1;
            continue; // transient mint failure: retry
        }
        ret = cp.rdmaPut(ctx, token, size);
        rdma.putToken(token);
        if (ret > 0 || ret == rdma_not_supported) {
            break;
        }
    }
    return ret;
}

ssize_t
rdmaGetWithRetry(SharedCuObjClient &rdma,
                 S3RdmaControlPlane &cp,
                 S3RdmaClientCtx &ctx,
                 void *buf,
                 uint64_t size,
                 uint64_t offset) {
    // Reject a zero-size GET before minting a cuObject token (rdmaGet also
    // guards, but the token is minted here first).
    if (size == 0) {
        NIXL_ERROR << "rdmaGet: zero-size request for key=" << ctx.object;
        return rdma_error;
    }
    ssize_t ret = -1;
    for (int attempt = 0; attempt < rdma_max_attempts; ++attempt) {
        char *token = rdma.getToken(buf, size, CUOBJ_GET);
        if (token == nullptr) {
            ret = -1;
            continue; // transient mint failure: retry
        }
        ret = cp.rdmaGet(ctx, token, size, offset);
        rdma.putToken(token);
        // ret >= 0 is success: 0 is a valid transfer (an empty object/range).
        // Only a negative rdma_error is a transient failure worth retrying; a
        // decline is terminal.
        if (ret >= 0 || ret == rdma_not_supported) {
            break;
        }
    }
    return ret;
}

ssize_t
rdmaPutObject(SharedCuObjClient &rdma,
              S3RdmaControlPlane &cp,
              S3RdmaClientCtx &ctx,
              void *buf,
              uint64_t size) {
    if (size <= rdma_max_request_size) {
        return rdmaPutWithRetry(rdma, cp, ctx, buf, size);
    }
    ctx.uploadId = cp.createMultipartUpload(ctx);
    if (ctx.uploadId.empty()) {
        return rdma_error;
    }

    struct abortUnlessCompleted {
        S3RdmaControlPlane &cp;
        const S3RdmaClientCtx &ctx;
        bool completed = false;

        ~abortUnlessCompleted() {
            if (!completed) {
                cp.abortMultipartUpload(ctx);
            }
        }
    } upload{cp, ctx};

    std::vector<std::string> etags;
    for (uint64_t done = 0; done < size; done += rdma_max_request_size) {
        const uint64_t len = std::min(rdma_max_request_size, size - done);
        ctx.partNumber = static_cast<uint32_t>(etags.size() + 1);
        const ssize_t r = rdmaPutWithRetry(rdma, cp, ctx, static_cast<char *>(buf) + done, len);
        if (r != static_cast<ssize_t>(len)) {
            NIXL_ERROR << "RDMA multipart upload part " << ctx.partNumber << " failed (" << r
                       << " of " << len << ") for key=" << ctx.object;
            return rdma_error;
        }
        etags.push_back(ctx.etag);
    }
    upload.completed = cp.completeMultipartUpload(ctx, etags);
    return upload.completed ? static_cast<ssize_t>(size) : rdma_error;
}

ssize_t
rdmaGetObject(SharedCuObjClient &rdma,
              S3RdmaControlPlane &cp,
              S3RdmaClientCtx &ctx,
              void *buf,
              uint64_t size,
              uint64_t offset) {
    if (size == 0) {
        return rdmaGetWithRetry(rdma, cp, ctx, buf, size, offset);
    }
    uint64_t done = 0;
    while (done < size) {
        const uint64_t len = std::min(rdma_max_request_size, size - done);
        const ssize_t r =
            rdmaGetWithRetry(rdma, cp, ctx, static_cast<char *>(buf) + done, len, offset + done);
        if (r < 0) {
            return r;
        }
        done += static_cast<uint64_t>(r);
        if (r != static_cast<ssize_t>(len)) {
            break;
        }
        ctx.ifMatch = ctx.etag;
    }
    return static_cast<ssize_t>(done);
}

} // namespace nixl_obj_rdma
