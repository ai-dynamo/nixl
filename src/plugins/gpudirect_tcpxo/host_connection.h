/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#ifndef GPUDIRECT_TCPXO_HOST_CONNECTION_H_
#define GPUDIRECT_TCPXO_HOST_CONNECTION_H_

#include <cinttypes>
#include <cstddef>
#include <cstdint>

#include <atomic>
#include <ios>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "backend/backend_aux.h"
#include "common/nixl_log.h"
#include "nixl_types.h"

#include "control_channel.h"
#include "control_channel.pb.h"
#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/dxs-client-interface.h"
#include "dxs/client/dxs-client-types.h"
#else
#include "rxdm_dxs_stub.h"
#endif
#include "dxs_endpoint.h"
#include "mpmc_queue.h"
#include "params.h"

namespace tcpxo {

class HostConnection;

enum RequestId : uint64_t {};

struct EndpointPair {
    uint64_t local_fastrak_idx;
    uint64_t remote_fastrak_idx;

    bool
    operator==(const EndpointPair &other) const {
        return local_fastrak_idx == other.local_fastrak_idx &&
            remote_fastrak_idx == other.remote_fastrak_idx;
    }

    template<typename H>
    friend H
    AbslHashValue(H h, const EndpointPair &c) {
        return H::combine(std::move(h), c.local_fastrak_idx, c.remote_fastrak_idx);
    }

    template<typename Sink>
    friend void
    AbslStringify(Sink &sink, const EndpointPair &ep) {
        absl::Format(&sink, "%d->%d", ep.local_fastrak_idx, ep.remote_fastrak_idx);
    }

    friend std::ostream &
    operator<<(std::ostream &os, const EndpointPair &ep) {
        return os << ep.local_fastrak_idx << "->" << ep.remote_fastrak_idx;
    }
};

struct MemoryRegion {
    uintptr_t addr;
    size_t len;

    friend std::ostream &
    operator<<(std::ostream &os, const MemoryRegion &mr) {
        auto flags = os.flags();
        os << "[" << std::hex << mr.addr << ", " << std::dec << mr.len << "]";
        os.flags(flags);
        return os;
    }
};

struct DxsOpParams {
    EndpointPair endpoint_pair;

    MemoryRegion local_mr;
    MemoryRegion remote_mr;

    dxs::Reg local_reg_handle;
    dxs::Reg remote_reg_handle;
    // The type that results from `uintptr_t - uintptr_t`
    ptrdiff_t local_page_offset;

    friend std::ostream &
    operator<<(std::ostream &os, const DxsOpParams &params) {
        return os << "{ep: " << params.endpoint_pair << ", local_mr: " << params.local_mr
                  << ", remote_mr: " << params.remote_mr
                  << ", local_reg: " << params.local_reg_handle
                  << ", remote_reg: " << params.remote_reg_handle
                  << ", offset: " << params.local_page_offset << "}";
    }
};

// This struct holds the send/recv DXS connections for the host<->host connection abstraction
struct EndpointConnection {
    DxsEndpoint &endpoint_;
    DxsConnection conn_;
};

// DXS Op Wrapper.
class nixlTcpxoDxsOp {
public:
    nixlTcpxoDxsOp(const nixl_xfer_op_t op_type,
                   std::unique_ptr<dxs::OpInterface> dxs_op,
                   const DxsOpParams params,
                   uint32_t flow_idx = 0)
        : op_type_(op_type),
          dxs_op_(std::move(dxs_op)),
          start_time_(absl::InfinitePast()),
          end_time_(absl::InfinitePast()),
          flow_idx_(flow_idx),
          params_(params) {}
    ~nixlTcpxoDxsOp() = default;
    nixlTcpxoDxsOp(nixlTcpxoDxsOp &&) = default;
    nixlTcpxoDxsOp &
    operator=(nixlTcpxoDxsOp &&) = default;

    // CheckXfer Per Op
    nixl_status_t
    checkXfer(absl::Duration slowness_threshold, absl::Duration timeout_threshold);

    inline void
    MarkStarted() {
        start_time_ = absl::Now();
        end_time_ = absl::InfiniteFuture();
        completion_status_ = std::nullopt;
    }

    inline void
    MarkCompleted(nixl_status_t completion_status) {
        completion_status_ = completion_status;
        end_time_ = absl::Now();
        NIXL_DEBUG << "OP completed: " << params_ << ". Start time: " << start_time_
                   << ". End time: " << end_time_ << ". Status: " << completion_status;
        current_slowness_threshold_ = absl::ZeroDuration();
    }

    inline const DxsOpParams &
    params() const {
        return params_;
    }

    inline void
    set_flow_idx(uint32_t flow_idx) {
        flow_idx_ = flow_idx;
    }

    inline uint32_t
    flow_idx() const {
        return flow_idx_;
    }

    inline const std::unique_ptr<dxs::OpInterface> &
    dxs_op() const {
        return dxs_op_;
    }

    inline void
    set_dxs_op(std::unique_ptr<dxs::OpInterface> &&dxs_op) {
        dxs_op_ = std::move(dxs_op);
    }

    inline absl::Time
    start_time() const {
        return start_time_;
    }

    inline absl::Time
    end_time() const {
        return end_time_;
    }

    inline nixl_xfer_op_t
    op_type() const {
        return op_type_;
    }

private:
    nixl_xfer_op_t op_type_;
    std::unique_ptr<dxs::OpInterface> dxs_op_{nullptr};
    std::optional<nixl_status_t> completion_status_{std::nullopt};
    absl::Time start_time_{absl::InfiniteFuture()};
    absl::Time end_time_{absl::InfiniteFuture()};
    uint32_t flow_idx_;
    absl::Duration current_slowness_threshold_{absl::ZeroDuration()};

    DxsOpParams params_;
};

/* Request handle for transfer operations. Has a reference to the HostConnection for easily checking
 * the DXS transfer */
class nixlTcpxoBackendReqH : public nixlBackendReqH {
public:
    nixlTcpxoBackendReqH(const nixl_xfer_op_t,
                         HostConnection &,
                         std::vector<DxsOpParams>,
                         std::optional<std::string> notification = std::nullopt);

    ~nixlTcpxoBackendReqH() = default;

    // Xfer APIs
    absl::Status
    postXfer();
    nixl_status_t
    checkXfer();

    inline void
    RecordSubmission() {
        submitted_requests_++;
    }

    inline void
    RecordCompletion() {
        completed_requests_++;
    }

    inline void
    MarkXferActive() {
        post_active_ = true;
    }

    inline void
    MarkXferInactive(nixl_status_t last_post_result) {
        last_post_result_ = last_post_result;
        post_active_ = false;
        xfer_msg_sent_ = false;
    }

    inline void
    MarkXferMsgSent() {
        xfer_msg_sent_ = true;
    }

    inline void
    set_notification(const std::string &notif) {
        notification_ = notif;
    }

    // Getters
    inline const std::optional<std::string> &
    notification() const {
        return notification_;
    }

    inline bool
    xfer_msg_sent() const {
        return xfer_msg_sent_;
    }

    inline bool
    post_active() const {
        return post_active_;
    }

    inline HostConnection &
    connection() const {
        return connection_;
    }

    inline RequestId
    id() const {
        return id_;
    }

    inline const std::vector<nixlTcpxoDxsOp> &
    ops() const {
        return ops_;
    }

    inline std::vector<nixlTcpxoDxsOp> &
    ops() {
        return ops_;
    }

    inline nixl_xfer_op_t
    op_type() const {
        return op_type_;
    }

private:
    static RequestId
    GenerateNextRequestId();

    // Request ID for this handle
    const RequestId id_;

    // NIXL Transfer OP Type
    const nixl_xfer_op_t op_type_;

    // The number of times each individual op was submitted
    std::atomic<size_t> completed_requests_;
    // The number of times each individual op was completed
    std::atomic<size_t> submitted_requests_;

    bool post_active_{false};
    bool xfer_msg_sent_{false};
    nixl_status_t last_post_result_{NIXL_ERR_NOT_POSTED};

    // The Host Connection associated with this request handle.
    HostConnection &connection_;

    // Container for all the DXS ops and their parsed params.
    std::vector<nixlTcpxoDxsOp> ops_;

    std::optional<std::string> notification_;
};

using HostConnectionTaskCallback = absl::AnyInvocable<void()>;

// A host<->host connection. Represents the control channel and DXS connections. Synchronization to
// this class is done through locking the `remote_agent_` map in the backend.
class HostConnection {
public:
    HostConnection(std::string remote_name,
                   PeerHandle remote_handle,
                   ControlChannel &channel,
                   absl::flat_hash_map<EndpointPair, EndpointConnection> &&connection_map,
                   const Params &params,
                   MPMCQueue<std::pair<std::string, std::string>> &pending_notifs,
                   HostConnectionTaskCallback host_connection_task_cb)
        : remote_name_(remote_name),
          remote_handle_(remote_handle),
          slowness_threshold_(absl::Milliseconds(params.fastrak_data_transfer_slowness_ms.value)),
          timeout_threshold_(absl::Milliseconds(params.fastrak_data_transfer_timeout_ms.value)),
          channel_(channel),
          connection_map_(std::move(connection_map)),
          pending_notifs_(pending_notifs),
          host_connection_task_cb_(std::move(host_connection_task_cb)) {}

    virtual ~HostConnection();

    inline absl::StatusOr<bool>
    IsConnectionReady() {
        return channel_.IsConnectionReady(remote_handle_);
    }

    absl::StatusOr<Message>
    ConstructXferMessageAndAssignFlows(nixlTcpxoBackendReqH &handle);

    absl::Status
    SendNotification(const std::string &);

    absl::Status
    PrepXfer(const nixl_xfer_op_t &operation,
             std::vector<DxsOpParams> xfer_ops,
             nixlTcpxoBackendReqH *&handle,
             std::optional<std::string> notification = std::nullopt);
    absl::Status
    PostXfer(nixlTcpxoBackendReqH &handle);
    nixl_status_t
    CheckXfer(nixlTcpxoBackendReqH &handle);
    nixl_status_t
    ReleaseReqH(nixlTcpxoBackendReqH &handle);

    absl::Status
    HandleRemoteXferOps(std::vector<nixlTcpxoDxsOp> ops);

    absl::Status
    IssueDxsOp(nixlTcpxoDxsOp &);

    void
    TestProgress();

    inline PeerHandle
    remote_handle() const {
        return remote_handle_;
    }

    inline std::string
    remote_name() const {
        return remote_name_;
    }

    inline absl::Duration
    slowness_threshold() const {
        return slowness_threshold_;
    }

    inline absl::Duration
    timeout_threshold() const {
        return timeout_threshold_;
    }

    /**
     * @brief Called by the backend's postXfer, this function checks if any endpoint pair is not
     * established
     * @details
     * If all endpoint pairs are established, then the backend can perform the send inline.
     */
    virtual absl::Status AreDxsEndpointsConnected(EndpointPair);

    /**
     * @brief Checks if the remote listen map contains addresses for all given operations.
     * @returns
     * - absl::FailedPreconditionError if the remote listen map is empty.
     * @returns
     * - absl::NotFoundError if the remote listen map is missing the listen information for an
     *   endpoint pair. This is a bad application state, because we only ever get the one
     *   DxsAddressExchange message. There are no more listen handles coming down the pipe after we
     *   populate the map once. We should give up on this operation entirely. Furthermore, the
     *   workload is probably not setup correctly in some bizzare way.
     */
    absl::Status
    HasRemoteListenHandles(const std::vector<nixlTcpxoDxsOp> &);

    /**
     * @brief Helper function for EstablishEndpointConnectionsAndDrainPendingOps
     * @details
     * Finds endpoints in the ops list that aren't connected and adds them to the set
     */
    absl::Status
    FindUnconnectedEndpoints(const std::vector<nixlTcpxoDxsOp> &,
                             absl::flat_hash_set<EndpointPair> &);

    /**
     * @brief Enqueues an event to the Workers to perform the DXS connect/accept dance
     * @details
     * Returns true if the span is non-empty. Returns false otherwise. This function doesn't do the
     * connect/accept dance on the same thread as its caller, as that caller may be a NIXL thread.
     */
    virtual bool
    EstablishEndpointConnections(const absl::flat_hash_set<EndpointPair> &);

    /**
     * @brief Drains any pending ops, performing the connect/accept dance
     * @details
     * Expected to be called on the worker thread, from the backend's DXS Address exchange callback.
     * At this point, both peers have each other's endpoints.
     */
    absl::Status
    EstablishEndpointConnectionsAndDrainPendingOps();

    /**
     * @brief This is the connection state machine
     * @details
     * The high-level flow is this:
     * 1. Issue all connects, and track each connect with a Connecting enum and the number of times
     *    a connection has been attempted
     * 2. Call accept sequentially for each flow. Keep a timeout for each individual flow.
     * 3. Add the created sockets to DxsFlow objects as they successfully connect/accept.
     * 4. If not all flows are fully connected, this function requeues
     *    ProgressDxsConnectionState.
     */
    void
    ProgressDxsConnectionState(absl::Duration dxs_connect_timeout_ms,
                               absl::Duration dxs_accept_timeout_ms);

    absl::Status
    ParseRemoteListenMap(const DxsAddressExchangeMessage &);

private:
    enum class DxsConnectAcceptStates { kInitial, kInProgress, kDone };

    struct DxsConnectionStatePerEndpointPair {
        DxsConnectAcceptStates connect_state = DxsConnectAcceptStates::kInitial;
        DxsConnectAcceptStates accept_state = DxsConnectAcceptStates::kInitial;
        absl::Time connect_start_time = absl::InfinitePast();
        absl::Time accept_start_time = absl::InfinitePast();
        EndpointPair endpoint_pair;
    };

    struct DxsConnectionState {
        std::vector<DxsConnectionStatePerEndpointPair> endpoints_to_connect;
    };

    absl::StatusOr<EndpointConnection *>
    GetEndpointConnectionFromConnectionMap(EndpointPair endpoint_pair);

    absl::Status
    ProgressDxsConnect(DxsConnectionStatePerEndpointPair &endpoint_pair_conn_state,
                       absl::Duration dxs_connect_timeout_ms);

    absl::Status
    ProgressDxsAccept(DxsConnectionStatePerEndpointPair &endpoint_pair_conn_state,
                      absl::Duration dxs_accept_timeout_ms);

    const std::string remote_name_;
    const PeerHandle remote_handle_;
    const absl::Duration slowness_threshold_;
    const absl::Duration timeout_threshold_;

    ControlChannel &channel_;
    absl::flat_hash_map<EndpointPair, EndpointConnection> connection_map_;

    absl::flat_hash_map<EndpointPair, std::vector<DxsAddress>> remote_listen_map_;

    MPMCQueue<std::pair<std::string, std::string>> &pending_notifs_;
    // The HostConnection will call this function from the TCPXO Backend whenever it needs to have
    // the backend do some work on its behalf. Right now, we only have one thing to tell the backend
    // to do (DxsConnectionEstablishCallback), so this function type takes no arguments, and we keep
    // the state the function needs to do the work on this object (i.e. DxsConnectionState)
    HostConnectionTaskCallback host_connection_task_cb_;

    /**
     * @brief Keeps track of the endpoints we're trying to connect, and their progress
     * @details
     * If no endpoints are present, then there's no in-progress connection attempt.
     */
    DxsConnectionState dxs_state_;

    absl::Mutex request_map_mutex_;
    absl::flat_hash_map<RequestId, std::unique_ptr<nixlTcpxoBackendReqH>>
        request_map_ ABSL_GUARDED_BY(request_map_mutex_);
    std::vector<nixlTcpxoBackendReqH * absl_nonnull> pending_xfers_;
    std::vector<nixlTcpxoDxsOp> pending_remote_xfers_;

    MPMCQueue<nixlTcpxoDxsOp> remote_initiated_ops_;
};

} // namespace tcpxo

#endif // GPUDIRECT_TCPXO_HOST_CONNECTION_H_
