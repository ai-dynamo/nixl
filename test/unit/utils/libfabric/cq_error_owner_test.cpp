/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NIXL contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "libfabric_mock_stubs.h"

#include <deque>
#include <iostream>
#include <vector>

extern "C" int __wrap_numa_max_node() { return 1; }
extern "C" int __wrap_numa_num_configured_nodes() { return 2; }
extern "C" int __wrap_fi_getinfo(uint32_t, const char *, const char *, uint64_t,
                                 const struct fi_info *, struct fi_info **out) {
    *out = mock_fi_info_chain(1, 100ull * NIXL_LIBFABRIC_GIGA);
    return 0;
}
extern "C" int __wrap_fi_fabric(struct fi_fabric_attr *, struct fid_fabric **out, void *) {
    *out = mock_fabric_create();
    return 0;
}

static std::deque<int> cq_steps;
static void *failed_context = nullptr;
static int post_calls = 0;

static ssize_t read_cq(struct fid_cq *, void *, size_t, fi_addr_t *) {
    if (cq_steps.empty()) return -FI_EAGAIN;
    const int result = cq_steps.front();
    cq_steps.pop_front();
    return result;
}
static ssize_t read_error(struct fid_cq *, struct fi_cq_err_entry *entry, uint64_t) {
    entry->op_context = failed_context;
    entry->err = FI_ECANCELED;
    return 1;
}
static ssize_t send_write(struct fid_ep *, const struct fi_msg_rma *, uint64_t) {
    ++post_calls;
    return post_calls == 1 ? -FI_EAGAIN : 0;
}

int main() {
    cq_ops_stub.size = sizeof(fi_ops_cq);
    cq_ops_stub.readfrom = read_cq;
    cq_ops_stub.readerr = read_error;

    nixlLibfabricRailManager manager(0);
    auto &rail = manager.getRail(0);

    // The CQ failure belongs to a previously posted request, not to the
    // healthy queued write that causes the drain to poll for completions.
    auto *failed = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 11);
    auto *healthy = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 22);
    if (!failed || !healthy) return 1;

    std::vector<nixl_status_t> failed_status;
    std::vector<nixl_status_t> healthy_status;
    failed->completion_callback = [&](nixl_status_t s) { failed_status.push_back(s); };
    healthy->completion_callback = [&](nixl_status_t s) { healthy_status.push_back(s); };

    fi_ops_rma rma = {};
    rma.writemsg = send_write;
    rail.endpoint->rma = &rma;

    nixlLibfabricPostRequest post = {};
    post.type = nixlLibfabricPostRequest::WRITE;
    post.req = healthy;
    post.length = 64;
    failed_context = &failed->ctx;
    cq_steps.push_back(-FI_EAVAIL);
    rail.enqueuePost(post);

    if (rail.drainPostQueue() != NIXL_SUCCESS) {
        std::cerr << "drain failed for unrelated healthy request\n";
        return 1;
    }
    if (failed_status.size() != 1 || failed_status[0] != NIXL_ERR_BACKEND ||
        failed->in_use || !healthy_status.empty() || !healthy->in_use || post_calls != 2) {
        std::cerr << "CQ error did not complete only its owner\n";
        return 1;
    }

    // A duplicate notification must not fire the callback again.
    cq_steps.push_back(-FI_EAVAIL);
    rail.progressCompletionQueue();
    if (failed_status.size() != 1) {
        std::cerr << "duplicate completion for released request\n";
        return 1;
    }
    // The queued write is still in flight and owned by the caller/provider.
    return 0;
}
