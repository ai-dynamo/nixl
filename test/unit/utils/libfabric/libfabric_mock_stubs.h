/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

/*
 * Shared libfabric stub ops for unit tests that need to construct a
 * nixlLibfabricRailManager without real hardware.  Include this header
 * in exactly one translation unit per test executable (it contains
 * definitions, not just declarations).
 */
#ifndef NIXL_TEST_UNIT_UTILS_LIBFABRIC_LIBFABRIC_MOCK_STUBS_H
#define NIXL_TEST_UNIT_UTILS_LIBFABRIC_LIBFABRIC_MOCK_STUBS_H

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>

// helper template for allocating zeroed C structs (free()-able)
template<typename T>
T *
malloc_zero() {
    T *res = (T *)calloc(1, sizeof(T));
    if (res == nullptr) {
        std::stringstream ss;
        ss << "Failed to allocate " << sizeof(T) << " bytes";
        throw std::runtime_error(ss.str());
    }
    return res;
}

// --- AV stubs ---

static int
fi_av_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

static struct fi_ops fi_av_fid_ops_stub{
    .close = fi_av_close_stub,
};

static fi_ops_av av_ops_stub = {};

static int
fi_av_open_stub(struct fid_domain *domain,
                struct fi_av_attr *attr,
                struct fid_av **av,
                void *context) {
    *av = malloc_zero<fid_av>();
    (*av)->fid.ops = &fi_av_fid_ops_stub;
    (*av)->ops = &av_ops_stub;
    return 0;
}

// --- CQ stubs ---

static int
fi_cq_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

static struct fi_ops fi_cq_fid_ops_stub{
    .close = fi_cq_close_stub,
};

static fi_ops_cq cq_ops_stub = {};

static int
fi_cq_open_stub(struct fid_domain *domain,
                struct fi_cq_attr *attr,
                struct fid_cq **cq,
                void *context) {
    *cq = malloc_zero<fid_cq>();
    (*cq)->fid.ops = &fi_cq_fid_ops_stub;
    (*cq)->ops = &cq_ops_stub;
    return 0;
}

// --- EP stubs ---

static int
fi_ep_setopt_stub(fid_t fid, int level, int optname, const void *optval, size_t optlen) {
    return 0;
}

static fi_ops_ep fi_ep_ops_stub = {
    .setopt = fi_ep_setopt_stub,
};

static int
fi_ep_bind_stub(struct fid *fid, struct fid *bfid, uint64_t flags) {
    return 0;
}

static int
fi_ep_control_stub(struct fid *fid, int command, void *arg) {
    return 0;
}

static int
fi_ep_cm_getname_stub(fid_t fid, void *addr, size_t *addrlen) {
    return 0;
}

static int
fi_ep_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

static struct fi_ops fi_ep_fid_ops_stub{
    .close = fi_ep_close_stub,
    .bind = fi_ep_bind_stub,
    .control = fi_ep_control_stub,
};

static struct fi_ops_cm fi_ep_cm_ops_stub{
    .getname = fi_ep_cm_getname_stub,
};

static ssize_t
fi_ep_recvmsg_stub(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags) {
    return 0;
}

static fi_ops_msg fi_ep_msg_ops_stub{
    .recvmsg = fi_ep_recvmsg_stub,
};

static int
fi_endpoint_stub(struct fid_domain *domain,
                 struct fi_info *info,
                 struct fid_ep **ep,
                 void *context) {
    *ep = malloc_zero<fid_ep>();
    (*ep)->ops = &fi_ep_ops_stub;
    (*ep)->fid.ops = &fi_ep_fid_ops_stub;
    (*ep)->cm = &fi_ep_cm_ops_stub;
    (*ep)->msg = &fi_ep_msg_ops_stub;
    return 0;
}

// --- MR stubs ---

static int
fi_mr_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

static fi_ops fi_mr_self_ops_stub = {
    .close = fi_mr_close_stub,
};

static int
fi_mr_reg_stub(struct fid *fid,
               const void *buf,
               size_t len,
               uint64_t access,
               uint64_t offset,
               uint64_t requested_key,
               uint64_t flags,
               struct fid_mr **mr,
               void *context) {
    *mr = malloc_zero<fid_mr>();
    (*mr)->fid.ops = &fi_mr_self_ops_stub;
    return 0;
}

static int
fi_mr_regattr_stub(struct fid *fid,
                   const struct fi_mr_attr *attr,
                   uint64_t flags,
                   struct fid_mr **mr) {
    *mr = malloc_zero<fid_mr>();
    (*mr)->fid.ops = &fi_mr_self_ops_stub;
    return 0;
}

static fi_ops_mr fi_mr_ops_stub{
    .reg = fi_mr_reg_stub,
    .regv = nullptr,
    .regattr = fi_mr_regattr_stub,
};

// --- Domain stubs ---

static fi_ops_domain domain_ops_stub = {.av_open = fi_av_open_stub,
                                        .cq_open = fi_cq_open_stub,
                                        .endpoint = fi_endpoint_stub,
                                        .scalable_ep = nullptr,
                                        .cntr_open = nullptr,
                                        .poll_open = nullptr,
                                        .stx_ctx = nullptr,
                                        .srx_ctx = nullptr,
                                        .query_atomic = nullptr,
                                        .query_collective = nullptr,
                                        .endpoint2 = nullptr};

static int
fi_domain_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

static struct fi_ops fi_domain_ops_stub{
    .close = fi_domain_close_stub,
};

static int
fi_domain_stub(struct fid_fabric *fabric,
               struct fi_info *info,
               struct fid_domain **domain,
               void *context) {
    *domain = malloc_zero<fid_domain>();
    (*domain)->fid.ops = &fi_domain_ops_stub;
    (*domain)->ops = &domain_ops_stub;
    (*domain)->mr = &fi_mr_ops_stub;
    return 0;
}

// --- Fabric stubs ---

static fi_ops_fabric fabric_ops_stub{.domain = fi_domain_stub,
                                     .passive_ep = nullptr,
                                     .eq_open = nullptr,
                                     .wait_open = nullptr,
                                     .trywait = nullptr,
                                     .domain2 = nullptr};

static int
fi_fabric_close_stub(struct fid *fid) {
    free(fid);
    return 0;
}

static struct fi_ops fi_fabric_ops_stub{
    .close = fi_fabric_close_stub,
};

// Helper: build a mock fid_fabric using the stubs above
static struct fid_fabric *
mock_fabric_create() {
    fid_fabric *fabric = malloc_zero<fid_fabric>();
    fabric->fid.ops = &fi_fabric_ops_stub;
    fabric->ops = &fabric_ops_stub;
    return fabric;
}

#endif // NIXL_TEST_UNIT_UTILS_LIBFABRIC_LIBFABRIC_MOCK_STUBS_H
