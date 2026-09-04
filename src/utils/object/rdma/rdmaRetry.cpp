/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rdmaRetry.h"

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
        char *token = rdma.getToken(buf, size, 0, CUOBJ_PUT);
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
        char *token = rdma.getToken(buf, size, 0, CUOBJ_GET);
        if (token == nullptr) {
            ret = -1;
            continue; // transient mint failure: retry
        }
        ret = cp.rdmaGet(ctx, token, size, offset);
        rdma.putToken(token);
        if (ret > 0 || ret == rdma_not_supported) {
            break;
        }
    }
    return ret;
}

} // namespace nixl_obj_rdma
