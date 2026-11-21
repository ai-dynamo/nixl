/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "libfabric_proxy_engine.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "common/configuration.h"
#include "common/exception.h"
#include "common/nixl_log.h"
#include "device/device_memview.h"
#include "device/device_ops.h"
#include "device/proxy/proxy_config.h"
#include "device/proxy/proxy_runtime.h"
#include "device/proxy/proxy_transport.h"

namespace nixl::libfabric {
class proxyEngine::transport final : public nixl::proxyTransport {
public:
    transport(nixlLibfabricRailManager &rails, const nixl::proxyConfig &config)
        : rails_(rails),
          max_peers_(config.max_peers),
          thread_count_(config.thread_count),
          rings_(config.ringCount()) {
        for (uint32_t thread = 0; thread < config.thread_count; ++thread) {
            const nixl::proxyThreadShare share = config.threadShare(thread);
            for (uint32_t channel = share.first_channel;
                 channel < share.first_channel + share.channel_count;
                 ++channel) {
                for (uint32_t peer = share.peers.first; peer < max_peers_;
                     peer += share.peers.stride) {
                    ringState &ring = rings_[ringIndex(channel, peer)];
                    ring.thread = thread;
                    ring.anchor = channel == share.first_channel && peer == share.peers.first;
                    ring.ops = std::vector<op>(config.ring_depth);
                }
            }
        }
    }

    nixl_status_t
    submit(const nixl::proxyBackendSubmission &submission,
           nixl::proxyBackendRequest &request) override {
        ringState &ring = rings_[ringIndex(submission.channel_id, submission.peer_index)];
        op &record = ring.ops[ring.next_op++ % ring.ops.size()];
        const nixl_status_t status = submission.opcode == nixl_proxy_opcode_t::PUT ?
            postPut(submission, ring, record) :
            postAtomicAdd(submission, ring, record);
        if (status == NIXL_IN_PROG) {
            request.token = reinterpret_cast<uintptr_t>(&record);
        }
        return status;
    }

    nixl_status_t
    checkCompletion(uint32_t, uint32_t, const nixl::proxyBackendRequest &request) override {
        return reinterpret_cast<const op *>(request.token)->status;
    }

    void
    progress(uint32_t channel, uint32_t peer) noexcept override {
        ringState &ring = rings_[ringIndex(channel, peer)];
        if (ring.anchor) {
            pollRails(ring.thread);
        }
        while (!ring.signals.empty() && ring.puts_in_flight.front() == 0) {
            op &record = *ring.signals.front();
            ring.signals.pop_front();
            ring.puts_in_flight.pop_front();
            ++ring.first_epoch;
            const nixl_status_t status = postSignal(ring, record);
            if (status != NIXL_IN_PROG) {
                record.status = status;
            }
        }
    }

    nixl_status_t
    quiesce(uint32_t, uint32_t) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    resolveDirectPtrs(const nixl_remote_meta_dlist_t &dlist,
                      std::vector<void *> &direct_ptrs) override {
        if (nixl::config::checkExistence("NIXL_LIBFABRIC_DISABLE_CUDA_IPC")) {
            return NIXL_SUCCESS;
        }
        for (const auto &desc : dlist) {
            auto *md = static_cast<nixlLibfabricPublicMetadata *>(desc.metadataP);
            char *ptr = nullptr;
            if (md != nullptr) {
                if (!md->cuda_ipc_mapping_) {
                    md->cuda_ipc_mapping_ = LibfabricUtils::cudaIpcImport(md->cuda_ipc_);
                }
                if (md->cuda_ipc_mapping_) {
                    ptr = static_cast<char *>(md->cuda_ipc_mapping_.get()) + md->cuda_ipc_.offset +
                        (desc.addr - md->remote_buf_addr_);
                }
            }
            direct_ptrs.push_back(ptr);
        }
        return NIXL_SUCCESS;
    }

private:
    struct op {
        uint64_t operand = 0;
        uint64_t remote_addr = 0;
        const nixlLibfabricPublicMetadata *rmd = nullptr;
        size_t rail_id = 0;
        size_t remote_ep = 0;
        size_t epoch = 0;
        bool is_put = false;
        std::atomic<nixl_status_t> status = NIXL_IN_PROG;
    };

    struct alignas(64) ringState {
        std::vector<op> ops;
        size_t next_op = 0;
        size_t next_rail = 0;
        size_t next_endpoint = 0;
        size_t atomic_rail = SIZE_MAX;
        std::deque<uint32_t> puts_in_flight{0};
        size_t first_epoch = 0;
        std::deque<op *> signals;
        uint32_t thread = 0;
        bool anchor = false;
    };

    [[nodiscard]] size_t
    ringIndex(uint32_t channel, uint32_t peer) const noexcept {
        return static_cast<size_t>(channel) * max_peers_ + peer;
    }

    void
    pollRails(uint32_t thread) noexcept {
        for (size_t rail_id = thread; rail_id < rails_.getNumRails(); rail_id += thread_count_) {
            rails_.getRail(rail_id).progressCompletionQueue();
        }
    }

    [[nodiscard]] int
    selectRail(ringState &ring, const std::vector<size_t> &selected_rails) const noexcept {
        for (size_t i = 0; i < selected_rails.size(); ++i) {
            const size_t rail_id = selected_rails[ring.next_rail++ % selected_rails.size()];
            if (rail_id % thread_count_ == ring.thread) {
                return static_cast<int>(rail_id);
            }
        }
        return -1;
    }

    nixl_status_t
    postPut(const nixl::proxyBackendSubmission &submission, ringState &ring, op &record) {
        const nixlMetaDesc &local = submission.local;
        const nixlMetaDesc &remote = submission.remote;
        const auto *lmd = static_cast<const nixlLibfabricPrivateMetadata *>(local.metadataP);
        const auto *rmd = static_cast<const nixlLibfabricPublicMetadata *>(remote.metadataP);
        const int rail_id = selectRail(ring, lmd->selected_rails_);
        if (rail_id < 0) {
            return NIXL_ERR_INVALID_PARAM;
        }
        ring.atomic_rail = rail_id;
        const auto &endpoints = rmd->remote_selected_endpoints_;
        const size_t remote_ep = endpoints[ring.next_endpoint++ % endpoints.size()];
        record.is_put = true;
        return post(
            ring, rail_id, *rmd, remote_ep, remote.addr, record, [&](auto &rail, auto... args) {
                return rail.postWrite(reinterpret_cast<void *>(local.addr),
                                      submission.size,
                                      fi_mr_desc(lmd->rail_mr_list_[rail_id]),
                                      0,
                                      args...,
                                      FI_DELIVERY_COMPLETE);
            });
    }

    nixl_status_t
    postAtomicAdd(const nixl::proxyBackendSubmission &submission, ringState &ring, op &record) {
        const nixlMetaDesc &remote = submission.remote;
        const auto *rmd = static_cast<const nixlLibfabricPublicMetadata *>(remote.metadataP);
        const auto &endpoints = rmd->remote_selected_endpoints_;
        record.operand = submission.value;
        record.remote_addr = remote.addr;
        record.rmd = rmd;
        record.remote_ep = endpoints[(remote.addr / sizeof(uint64_t)) % endpoints.size()];
        record.rail_id = ring.atomic_rail == SIZE_MAX ? ring.thread : ring.atomic_rail;
        record.is_put = false;
        if (ring.signals.empty() && ring.puts_in_flight.front() == 0) {
            return postSignal(ring, record);
        }
        record.status = NIXL_IN_PROG;
        ring.signals.push_back(&record);
        ring.puts_in_flight.push_back(0);
        return NIXL_IN_PROG;
    }

    nixl_status_t
    postSignal(ringState &ring, op &record) {
        return post(ring,
                    record.rail_id,
                    *record.rmd,
                    record.remote_ep,
                    record.remote_addr,
                    record,
                    [&](auto &rail, auto... args) {
                        return rail.postAtomicAdd(&record.operand, args..., FI_DELIVERY_COMPLETE);
                    });
    }

    template<typename PostFn>
    nixl_status_t
    post(ringState &ring,
         size_t rail_id,
         const nixlLibfabricPublicMetadata &rmd,
         size_t remote_ep,
         uint64_t remote_addr,
         op &record,
         PostFn &&post_fn) {
        nixlLibfabricRail &rail = rails_.getRail(rail_id);
        const fi_addr_t dest = rmd.conn_->rail_remote_addr_list_.at(rail_id)[remote_ep];
        nixlLibfabricReq *req = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 0);
        if (req == nullptr) {
            return NIXL_ERR_BACKEND;
        }
        record.status = NIXL_IN_PROG;
        record.epoch = ring.first_epoch + ring.puts_in_flight.size() - 1;
        req->completion_callback = [&record, &ring](nixl_status_t status) {
            if (record.is_put) {
                --ring.puts_in_flight[record.epoch - ring.first_epoch];
            }
            record.status = status;
        };

        const nixl_status_t status =
            post_fn(rail, dest, remote_addr, rmd.rail_remote_key_list_[remote_ep], req);
        if (status != NIXL_SUCCESS) {
            rail.releaseRequest(req);
            return status;
        }
        if (record.is_put) {
            ++ring.puts_in_flight.back();
        }
        return NIXL_IN_PROG;
    }

    nixlLibfabricRailManager &rails_;
    const uint32_t max_peers_;
    const uint32_t thread_count_;
    std::vector<ringState> rings_;
};
} // namespace nixl::libfabric

namespace {
[[nodiscard]] nixl::deviceOps &
requireDeviceOps() {
    nixl::deviceOps *ops = nixl::getDeviceOps();
    if (ops == nullptr) {
        nixl::throwRuntimeError(
            "device proxy needs a device operations implementation; none is loaded");
    }
    return *ops;
}
} // namespace

namespace nixl::libfabric {
proxyEngine::proxyEngine(const nixlBackendInitParams &init_params, const nixl::proxyConfig &config)
    : nixlLibfabricEngine(&init_params),
      ops_(requireDeviceOps()) {
    if (config.thread_count > railManager().getNumRails()) {
        nixl::throwRuntimeError("proxy_thread_count exceeds the libfabric rail count");
    }

    runtime_ = nixl::proxyRuntime::create(
        std::make_unique<transport>(railManager(), config), config, ops_);
}

proxyEngine::~proxyEngine() = default;

template<typename DlistT>
nixl_status_t
proxyEngine::prepProxyMemView(const DlistT &dlist, nixlMemViewH &mvh) const {
    nixlMemViewH proxy_mvh = nullptr;
    nixl_status_t status = runtime_->prepMemView(dlist, &proxy_mvh);
    if (status != NIXL_SUCCESS) {
        return status;
    }

    status = nixl::deviceMemViewAllocate(ops_, nixl_device_exec_mode_t::PROXY, proxy_mvh, mvh);
    if (status != NIXL_SUCCESS) {
        runtime_->releaseMemView(proxy_mvh);
    }
    return status;
}

nixl_status_t
proxyEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                         nixlMemViewH &mvh,
                         const nixl_opt_b_args_t *) const {
    return prepProxyMemView(dlist, mvh);
}

nixl_status_t
proxyEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                         nixlMemViewH &mvh,
                         const nixl_opt_b_args_t *) const {
    return prepProxyMemView(dlist, mvh);
}

void
proxyEngine::releaseMemView(nixlMemViewH mem_view) const {
    if (mem_view == nullptr) {
        return;
    }

    nixlMemViewH proxy_mvh = nullptr;
    if (nixl::deviceMemViewGetBackend(ops_, mem_view, proxy_mvh) != NIXL_SUCCESS) {
        NIXL_FATAL << "Failed to read proxy memory view before retirement";
    }

    runtime_->releaseMemView(proxy_mvh);
    nixl::deviceMemViewFree(ops_, mem_view);
}
} // namespace nixl::libfabric
