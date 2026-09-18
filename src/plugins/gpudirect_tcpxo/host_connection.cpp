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

#include "host_connection.h"

#include <algorithm>
#include <google/protobuf/repeated_ptr_field.h>
#include <optional>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "backend_aux.h"
#include "common/nixl_log.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/oss/status_macros.h"
#else
#include "rxdm_dxs_stub.h"
#endif

namespace tcpxo {

nixl_status_t
nixlTcpxoDxsOp::checkXfer(absl::Duration slowness_threshold, absl::Duration timeout_threshold) {
    VLOG_EVERY_N_SEC(1, 1) << "nixlTcpxoDxsOp::checkXfer called for op: flow: " << flow_idx_
                           << " params: " << params_ << " (" << COUNTER << " total)";
    if (completion_status_.has_value()) {
        return *completion_status_;
    }
    // Likely waiting on the endpoints to be connected
    if (dxs_op_ == nullptr) {
        return NIXL_ERR_NOT_POSTED;
    }

    // Ensure slowness threshold is initialized
    if (current_slowness_threshold_ == absl::ZeroDuration()) {
        current_slowness_threshold_ = slowness_threshold;
    }

    std::optional<absl::Status> status{std::nullopt};
    if (op_type_ == NIXL_WRITE) {
        auto *dxs_send_op = reinterpret_cast<dxs::SendOpInterface *>(dxs_op_.get());
        status = dxs_send_op->Test();
    } else {
        auto *dxs_recv_op = reinterpret_cast<dxs::LinearizedRecvOpInterface *>(dxs_op_.get());
        // For RecvLinearized, Test returns std::optional<absl::StatusOr<uint64_t>>. We just map it
        // to std::optional<absl::Status> and drop the bytes received (if present).
        if (const auto recv_status = dxs_recv_op->Test(); recv_status.has_value()) {
            if (recv_status->ok()) {
                NIXL_DEBUG << "Op: flow: " << flow_idx_ << " params: " << params_ << " received "
                           << **recv_status << " bytes";
            }
            status = recv_status.value().status();
        }
    }

    if (!status.has_value()) {
        absl::Duration elapsed = absl::Now() - start_time_;
        if (timeout_threshold != absl::ZeroDuration() && elapsed > timeout_threshold) {
            NIXL_ERROR << "Op Timeout: flow: " << flow_idx_ << " params: " << params_
                       << " exceeded timeout threshold of " << timeout_threshold;
            MarkCompleted(NIXL_ERR_CANCELED);
            return *completion_status_;
        }

        if (current_slowness_threshold_ != absl::ZeroDuration() &&
            elapsed > current_slowness_threshold_) {
            NIXL_WARN << "Op Slowness: flow: " << flow_idx_ << " params: " << params_
                      << " is pending for more than " << current_slowness_threshold_;
            current_slowness_threshold_ *= 2;
        }
        VLOG_EVERY_N_SEC(1, 1) << "Op: flow: " << flow_idx_ << " params: " << params_
                               << " is still in progress";
        return NIXL_IN_PROG;
    } else if (!status.value().ok()) {
        NIXL_ERROR << "Op Failed: flow: " << flow_idx_ << " params: " << params_
                   << " with status: " << status.value();
        MarkCompleted(NIXL_ERR_CANCELED);
        return *completion_status_;
    }
    // Status has value and it is OK, successful Op
    NIXL_DEBUG << "Op " << params_ << " completed successfully";
    MarkCompleted(NIXL_SUCCESS);
    return *completion_status_;
}

nixlTcpxoBackendReqH::nixlTcpxoBackendReqH(const nixl_xfer_op_t op_type,
                                           HostConnection &connection,
                                           std::vector<DxsOpParams> dxs_op_params_list,
                                           std::optional<std::string> notification)
    : nixlBackendReqH(),
      id_(GenerateNextRequestId()),
      op_type_(op_type),
      completed_requests_(0),
      submitted_requests_(0),
      connection_(connection),
      notification_(std::move(notification)) {
    ops_.reserve(dxs_op_params_list.size());
    for (auto &op : dxs_op_params_list) {
        ops_.emplace_back(op_type, nullptr, std::move(op), 0);
    }
}

absl::Status
nixlTcpxoBackendReqH::postXfer() {
    NIXL_DEBUG << "nixlTcpxoBackendReqH::postXfer called for handle " << id_;
    std::vector<nixlTcpxoDxsOp * absl_nonnull> issued_ops;
    std::vector<nixlTcpxoDxsOp * absl_nonnull> unissued_ops;
    absl::Status error_status{absl::OkStatus()};
    nixlTcpxoDxsOp *failed_op{nullptr};

    if (notification_.has_value()) {
        if (const auto status = connection_.SendNotification(*notification_); !status.ok()) {
            NIXL_ERROR << "Failed to send notification: " << status;
            for (auto &op : ops_) {
                op.MarkCompleted(NIXL_ERR_BACKEND);
            }
            MarkXferInactive(NIXL_ERR_BACKEND);
            return status;
        }
    }

    for (auto &op : ops_) {
        if (!error_status.ok()) {
            unissued_ops.push_back(&op);
            continue;
        }

        error_status = connection_.IssueDxsOp(op);
        if (error_status.ok()) {
            issued_ops.push_back(&op);
            RecordSubmission();
        } else {
            failed_op = &op;
        }
    }

    if (!error_status.ok()) {
        NIXL_ERROR << "Failed to issue DXS op: " << failed_op->params()
                   << " with status: " << error_status;
        failed_op->MarkCompleted(NIXL_ERR_BACKEND);

        for (auto *op : issued_ops) {
            NIXL_DEBUG << "Cancelling issued DXS op: " << op->params();
            op->MarkCompleted(NIXL_ERR_CANCELED);
        }

        for (auto *op : unissued_ops) {
            NIXL_DEBUG << "Unissued DXS op cancelled: " << op->params();
            op->MarkCompleted(NIXL_ERR_NOT_POSTED);
        }
        MarkXferInactive(NIXL_ERR_BACKEND);
        return error_status;
    }

    NIXL_DEBUG << "Successfully issued all DXS ops for handle " << id_;
    return absl::OkStatus();
}

nixl_status_t
nixlTcpxoBackendReqH::checkXfer() {
    if (!post_active()) {
        return last_post_result_;
    }

    std::vector<nixl_status_t> op_statuses;
    op_statuses.reserve(ops_.size());

    // Make sure to call test on all ops
    for (auto &op : ops_) {
        op_statuses.push_back(
            op.checkXfer(connection_.slowness_threshold(), connection_.timeout_threshold()));
    }
    std::optional<nixl_status_t> any_failed{std::nullopt};
    bool any_in_progress = false;
    for (const auto &op_status : op_statuses) {
        if (op_status == NIXL_IN_PROG || op_status == NIXL_ERR_NOT_POSTED) {
            any_in_progress = true;
        } else if (op_status != NIXL_SUCCESS) {
            any_failed = op_status;
        } else {
            RecordCompletion();
        }
    }

    if (any_failed.has_value()) {
        NIXL_ERROR << "Some DXS ops failed for handle " << id_;
        MarkXferInactive(*any_failed);
        return last_post_result_;
    } else if (any_in_progress) {
        VLOG_EVERY_N_SEC(1, 1)
            << "nixlTcpxoBackendReqH::checkXfer: transfer in progress for handle " << id_ << " ("
            << COUNTER << " total)";
        return NIXL_IN_PROG;
    } else {
        NIXL_DEBUG << "All DXS ops completed for handle " << id_;
        MarkXferInactive(NIXL_SUCCESS);
        return last_post_result_;
    }
}

HostConnection::~HostConnection() {

    absl::MutexLock lock(&request_map_mutex_);
    request_map_.clear();
}

absl::Status
HostConnection::SendNotification(const std::string &msg) {
    Message msg_pb;
    msg_pb.mutable_notif()->set_message(msg);
    auto send_id_or_status = channel_.ScheduleSend(remote_handle_, std::move(msg_pb));
    return send_id_or_status.status();
}

absl::Status
HostConnection::PrepXfer(const nixl_xfer_op_t &operation,
                         std::vector<DxsOpParams> dxs_op_params_list,
                         nixlTcpxoBackendReqH *&handle,
                         std::optional<std::string> notification) {
    auto request_handle = std::make_unique<nixlTcpxoBackendReqH>(
        operation, *this, std::move(dxs_op_params_list), std::move(notification));
    handle = request_handle.get();
    absl::MutexLock lock(&request_map_mutex_);
    request_map_.emplace(handle->id(), std::move(request_handle));
    return absl::OkStatus();
}

absl::StatusOr<Message>
HostConnection::ConstructXferMessageAndAssignFlows(nixlTcpxoBackendReqH &handle) {
    Message msg_pb;
    auto *xfer_msg = msg_pb.mutable_xfer();
    xfer_msg->set_type(handle.op_type() == NIXL_WRITE ? XFER_MESSAGE_TYPE_RECV :
                                                        XFER_MESSAGE_TYPE_SEND);

    for (auto &op : handle.ops()) {
        const auto &endpoint_pair = op.params().endpoint_pair;

        auto it = connection_map_.find(endpoint_pair);
        if (it == connection_map_.end()) {
            return absl::NotFoundError(
                absl::StrFormat("Connection entry for endpoint pair %v missing. This should not be "
                                "a possible state",
                                endpoint_pair));
        }

        auto &dxs_conn = it->second.conn_;
        uint8_t flow_idx = dxs_conn.GetNextFlow();
        op.set_flow_idx(flow_idx);

        auto *xfer_op = xfer_msg->add_ops();
        xfer_op->set_target_fastrak_idx(endpoint_pair.remote_fastrak_idx);
        xfer_op->set_initiator_fastrak_idx(endpoint_pair.local_fastrak_idx);
        xfer_op->set_memory_address(op.params().remote_mr.addr);
        xfer_op->set_len(op.params().remote_mr.len);
        xfer_op->set_flow_idx(flow_idx);
    }

    return msg_pb;
}

absl::Status
HostConnection::HasRemoteListenHandles(const std::vector<nixlTcpxoDxsOp> &ops) {
    if (remote_listen_map_.empty()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Remote listen map not yet initialized for agent ", remote_name_));
    }
    for (const auto &op : ops) {
        if (!remote_listen_map_.contains(op.params().endpoint_pair)) {
            return absl::NotFoundError(absl::StrCat(
                "Listen handle not found for endpoint pair ",
                op.params().endpoint_pair,
                ". This operation is malformed and not within the parameter the workload setup."));
        }
    }
    return absl::OkStatus();
}

absl::Status
HostConnection::PostXfer(nixlTcpxoBackendReqH &handle) {
    NIXL_DEBUG << "HostConnection::PostXfer called for handle " << handle.id() << " to remote "
               << remote_name_;
    absl::MutexLock lock(&request_map_mutex_);
    if (request_map_.find(handle.id()) == request_map_.end()) {
        return absl::NotFoundError("postXfer requested with unknown handle");
    }
    if (handle.post_active()) {
        return absl::AlreadyExistsError("request is currently being processed");
    }

    handle.MarkXferActive();
    absl::Cleanup reset_handle_active = [&handle] { handle.MarkXferInactive(NIXL_ERR_BACKEND); };

    const auto status = HasRemoteListenHandles(handle.ops());
    if (status.code() == absl::StatusCode::kFailedPrecondition) {
        NIXL_DEBUG << "Deferring postXfer for handle " << handle.id()
                   << " until remote listen handles are received from " << remote_name_;
        pending_xfers_.push_back(&handle);
        std::move(reset_handle_active).Cancel();
        return absl::OkStatus();
    } else if (!status.ok()) {
        return status;
    }

    // The XferMessage this constructs also tells the target which endpoints we need to
    // connect/accept on. Since connection are symmetric, both sides know which endpoints are and
    // aren't connected. The target can correctly infer which endpoints it needs to establish based
    // on the XferOps in this message. The connect/accept dance serves as barrier for both sides to
    // synchronize this transfer on
    ASSIGN_OR_RETURN(tcpxo::Message msg, ConstructXferMessageAndAssignFlows(handle));
    auto send_id_or_status = channel_.ScheduleSend(remote_handle_, std::move(msg));
    if (!send_id_or_status.ok()) {
        return send_id_or_status.status();
    }
    handle.MarkXferMsgSent();

    absl::flat_hash_set<EndpointPair> endpoints_to_establish;
    RETURN_IF_ERROR(FindUnconnectedEndpoints(handle.ops(), endpoints_to_establish));
    const auto establish_connections_enqueued =
        EstablishEndpointConnections(endpoints_to_establish);
    if (establish_connections_enqueued) {
        NIXL_DEBUG << "Deferring postXfer for handle " << handle.id()
                   << " until endpoint connections are established";
        pending_xfers_.push_back(&handle);
        std::move(reset_handle_active).Cancel();
        return absl::OkStatus();
    }

    NIXL_DEBUG << "Proceeding with immediate postXfer for handle " << handle.id();
    std::move(reset_handle_active).Cancel();
    return handle.postXfer();
}

nixl_status_t
HostConnection::CheckXfer(nixlTcpxoBackendReqH &handle) {
    VLOG_EVERY_N_SEC(1, 1) << "HostConnection::CheckXfer called for handle " << handle.id();
    absl::MutexLock lock(&request_map_mutex_);
    if (request_map_.find(handle.id()) == request_map_.end()) {
        return NIXL_ERR_BACKEND;
    }
    return handle.checkXfer();
}

nixl_status_t
HostConnection::ReleaseReqH(nixlTcpxoBackendReqH &handle) {
    absl::MutexLock lock(&request_map_mutex_);
    request_map_.erase(handle.id());
    return NIXL_SUCCESS;
}

absl::Status
HostConnection::HandleRemoteXferOps(std::vector<nixlTcpxoDxsOp> ops) {
    NIXL_DEBUG << "HostConnection::HandleRemoteXferOps called with " << ops.size()
               << " ops from remote " << remote_name_;
    for (auto &op : ops) {
        pending_remote_xfers_.push_back(std::move(op));
    }
    return EstablishEndpointConnectionsAndDrainPendingOps();
}

absl::StatusOr<EndpointConnection *>
HostConnection::GetEndpointConnectionFromConnectionMap(EndpointPair endpoint_pair) {
    auto it_connection_map = connection_map_.find(endpoint_pair);
    if (it_connection_map == connection_map_.end()) {
        return absl::NotFoundError(absl::StrCat("Endpoint Pair Not Found ",
                                                endpoint_pair.local_fastrak_idx,
                                                "-",
                                                endpoint_pair.remote_fastrak_idx));
    }
    return &it_connection_map->second;
}

absl::Status
HostConnection::AreDxsEndpointsConnected(EndpointPair endpoint_pair) {
    const auto it = connection_map_.find(endpoint_pair);
    if (it == connection_map_.end()) {
        return absl::NotFoundError(
            absl::StrCat(endpoint_pair,
                         " is not present in connection map. This pairing likely doesn't exist, or "
                         "the connection_map_ is uninitialized"));
    }

    const auto &dxs_conn = it->second;

    auto is_ready = [&endpoint_pair](const auto &socket) -> absl::Status {
        if (!socket) {
            return absl::FailedPreconditionError(
                absl::StrCat("Flows for ", endpoint_pair, " are not connected."));
        }
        auto ready = socket->SocketReady();
        if (ready.has_value() && ready.value().ok()) {
            return absl::OkStatus();
        } else {
            return absl::FailedPreconditionError(
                absl::StrCat("Flows for ", endpoint_pair, " are not connected."));
        }
    };

    for (const auto &flow : dxs_conn.conn_.flows()) {
        if (!is_ready(flow.send_socket).ok() || !is_ready(flow.recv_socket).ok()) {
            return absl::FailedPreconditionError(
                absl::StrCat("Flows for ", endpoint_pair, " are not connected."));
        }
    }

    return absl::OkStatus();
}

absl::Status
HostConnection::FindUnconnectedEndpoints(const std::vector<nixlTcpxoDxsOp> &ops,
                                         absl::flat_hash_set<EndpointPair> &unconnected_endpoints) {
    for (const auto &op : ops) {
        const auto status = AreDxsEndpointsConnected(op.params().endpoint_pair);
        if (status.ok()) {
            continue;
        }
        if (status.code() != absl::StatusCode::kFailedPrecondition) {
            return status;
        }

        unconnected_endpoints.insert(op.params().endpoint_pair);
    }
    return absl::OkStatus();
}

absl::Status
HostConnection::ProgressDxsConnect(DxsConnectionStatePerEndpointPair &endpoint_pair_conn_state,
                                   absl::Duration dxs_connect_timeout_ms) {
    EndpointPair endpoint_pair = endpoint_pair_conn_state.endpoint_pair;

    // Fetch Local DXS Connection and Endpoint Client for Endpoint pair
    ASSIGN_OR_RETURN(EndpointConnection * ec,
                     GetEndpointConnectionFromConnectionMap(endpoint_pair));
    DxsConnection &dxs_conn = ec->conn_;
    DxsEndpoint &dxs_ep_client = ec->endpoint_;

    // Fetch Remote DXS Listen Handles for Endpoint pair
    auto it_remote_listen_map = remote_listen_map_.find(endpoint_pair);
    if (it_remote_listen_map == remote_listen_map_.end()) {
        return absl::NotFoundError(absl::StrCat("Remote Endpoint Listen Handles Not Found for ",
                                                endpoint_pair.local_fastrak_idx,
                                                "-",
                                                endpoint_pair.remote_fastrak_idx));
    }
    std::vector<DxsAddress> &remote_listen_handles = it_remote_listen_map->second;

    // Size for number of flows and listen handles should be same.
    if (remote_listen_handles.size() != dxs_conn.flows().size()) {
        return absl::InternalError(
            "Remote Listen Handles and Local Connect Handles Count Mismatch");
    }
    auto conn_count = remote_listen_handles.size();
    NIXL_DEBUG << "Number of Remote Listen Size: " << conn_count;

    size_t connecting_count = 0;
    size_t connected_count = 0;

    // Progress DXS Connect State
    switch (endpoint_pair_conn_state.connect_state) {
    case DxsConnectAcceptStates::kInitial:
        connecting_count = 0;
        for (size_t i = 0; i < conn_count; i++) {

            // Time stamp start time.
            if (endpoint_pair_conn_state.connect_start_time == absl::InfinitePast()) {
                endpoint_pair_conn_state.connect_start_time = absl::Now();
            }

            // Connect if needed.
            if (dxs_conn.flows()[i].send_socket.get() == nullptr) {
                NIXL_DEBUG << "Connecting Endpoint Pair " << endpoint_pair << " Flow " << i
                           << " Remote DXS Address " << remote_listen_handles[i].addr().addr()
                           << " Remote DXS Port " << remote_listen_handles[i].port();
                ASSIGN_OR_RETURN(dxs_conn.flows()[i].send_socket,
                                 dxs_ep_client.dxs_client()->Connect(
                                     remote_listen_handles[i].addr().addr().c_str(),
                                     remote_listen_handles[i].port()));
            }

            // Account for connected flow.
            if (dxs_conn.flows()[i].send_socket.get() != nullptr) {
                connecting_count++;
            }
        }

        // If all flows are in connecting state progress state machine.
        if (connecting_count == conn_count) {
            NIXL_DEBUG << "Marking Connect State IN PROGRESS for endpoint pair " << endpoint_pair;
            endpoint_pair_conn_state.connect_state = DxsConnectAcceptStates::kInProgress;
        }
        break;
    case DxsConnectAcceptStates::kInProgress:
        connected_count = 0;
        for (size_t i = 0; i < conn_count; i++) {
            NIXL_DEBUG << "Checking Connect State for Endpoint Pair " << endpoint_pair << " Flow "
                       << i;
            std::optional<absl::Status> ready =
                (dxs_conn.flows()[i].send_socket.get())->SocketReady();
            if (!ready.has_value()) {
                // Check Timeout
                if (absl::Now() - endpoint_pair_conn_state.connect_start_time >
                    dxs_connect_timeout_ms) {
                    NIXL_ERROR << "Connect timed out for Endpoint Pair " << endpoint_pair
                               << " Flow " << i;
                    return absl::DeadlineExceededError(
                        absl::StrCat("Connect timed out for endpoint pair ",
                                     endpoint_pair.local_fastrak_idx,
                                     "-",
                                     endpoint_pair.remote_fastrak_idx,
                                     " flow ",
                                     i));
                }
                continue;
            }
            if (!ready.value().ok()) {
                // Check Error
                NIXL_ERROR << "Connect failed for Endpoint Pair " << endpoint_pair << " Flow " << i;
                return ready.value();
            } else {
                connected_count++;
                NIXL_DEBUG << "Connected Endpoint Pair " << endpoint_pair << " Flow " << i;
            }
        }

        // If all flows are in connected state progress state machine.
        if (connected_count == conn_count) {
            NIXL_DEBUG << "Marking Connect State DONE for endpoint pair " << endpoint_pair;
            endpoint_pair_conn_state.connect_state = DxsConnectAcceptStates::kDone;
        }
        break;
    case DxsConnectAcceptStates::kDone:
        // Nothing to do.
        break;
    default:
        return absl::InternalError("Invalid Connect State");
    }

    return absl::OkStatus();
}

absl::Status
HostConnection::ProgressDxsAccept(DxsConnectionStatePerEndpointPair &endpoint_pair_conn_state,
                                  absl::Duration dxs_accept_timeout_ms) {
    EndpointPair endpoint_pair = endpoint_pair_conn_state.endpoint_pair;

    // Fetch Local DXS Connection for Endpoint pair
    ASSIGN_OR_RETURN(EndpointConnection * ec,
                     GetEndpointConnectionFromConnectionMap(endpoint_pair));
    DxsConnection &dxs_conn = ec->conn_;

    auto conn_count = dxs_conn.flows().size();
    NIXL_DEBUG << "Local Accept Count: " << conn_count;

    size_t accepting_count = 0;
    size_t accepted_count = 0;

    // Progress DXS Accept State
    switch (endpoint_pair_conn_state.accept_state) {
    case DxsConnectAcceptStates::kInitial:
        accepting_count = 0;
        for (size_t i = 0; i < conn_count; i++) {
            // Time stamp start time.
            if (endpoint_pair_conn_state.accept_start_time == absl::InfinitePast()) {
                endpoint_pair_conn_state.accept_start_time = absl::Now();
            }

            // Check for timeout in case Accept() returned nullptr in previous call.
            if (absl::Now() - endpoint_pair_conn_state.accept_start_time > dxs_accept_timeout_ms) {
                NIXL_ERROR << "Accept timed out for Endpoint Pair " << endpoint_pair << " Flow "
                           << i;
                return absl::DeadlineExceededError(
                    absl::StrCat("Accept timed out for endpoint pair ",
                                 endpoint_pair.local_fastrak_idx,
                                 "-",
                                 endpoint_pair.remote_fastrak_idx,
                                 " flow ",
                                 i));
            }

            // Accept if needed.
            if (dxs_conn.flows()[i].recv_socket.get() == nullptr) {
                NIXL_DEBUG << "Accepting Endpoint Pair " << endpoint_pair << " Flow " << i;
                ASSIGN_OR_RETURN(dxs_conn.flows()[i].recv_socket,
                                 dxs_conn.flows()[i].listen_socket->Accept());
                if (dxs_conn.flows()[i].recv_socket.get() != nullptr) {
                    // Newly accepted in this iteration.
                    accepting_count++;
                }
            } else {
                // Previously accepted.
                accepting_count++;
            }
        }

        // If all flows are in accepting state progress state machine.
        if (accepting_count == conn_count) {
            NIXL_DEBUG << "Marking Accept State IN PROGRESS for endpoint pair " << endpoint_pair;
            endpoint_pair_conn_state.accept_state = DxsConnectAcceptStates::kInProgress;
        }
        break;
    case DxsConnectAcceptStates::kInProgress:
        accepted_count = 0;
        for (size_t i = 0; i < conn_count; i++) {
            NIXL_DEBUG << "Checking Accept State for Endpoint Pair " << endpoint_pair << " Flow "
                       << i;
            std::optional<absl::Status> ready =
                dxs_conn.flows()[i].recv_socket.get()->SocketReady();
            if (!ready.has_value()) {
                // Check Timeout
                if (absl::Now() - endpoint_pair_conn_state.accept_start_time >
                    dxs_accept_timeout_ms) {
                    NIXL_ERROR << "Accept timed out for Endpoint Pair " << endpoint_pair << " Flow "
                               << i;
                    return absl::DeadlineExceededError(
                        absl::StrCat("Accept timed out for endpoint pair ",
                                     endpoint_pair.local_fastrak_idx,
                                     "-",
                                     endpoint_pair.remote_fastrak_idx,
                                     " flow ",
                                     i));
                }
                continue;
            }
            if (!ready.value().ok()) {
                // Check Error
                NIXL_ERROR << "Accept failed for Endpoint Pair " << endpoint_pair << " Flow " << i;
                return ready.value();
            } else {
                accepted_count++;
                NIXL_DEBUG << "Accepted Endpoint Pair " << endpoint_pair << " Flow " << i;
            }
        }

        // If all flows are in accepted state progress state machine.
        if (accepted_count == conn_count) {
            NIXL_DEBUG << "Marking Accept State DONE for endpoint pair" << endpoint_pair;
            endpoint_pair_conn_state.accept_state = DxsConnectAcceptStates::kDone;
        }
        break;
    case DxsConnectAcceptStates::kDone:
        // Nothing to do.
        break;
    default:
        return absl::InternalError("Invalid Accept State");
    }

    return absl::OkStatus();
}

void
HostConnection::ProgressDxsConnectionState(absl::Duration dxs_connect_timeout_ms,
                                           absl::Duration dxs_accept_timeout_ms) {

    NIXL_DEBUG << "Progressing DXS Connection State...";

    while (dxs_state_.endpoints_to_connect.size() > 0) {

        NIXL_DEBUG << "Dxs Enpoints to Connect/Accept: " << dxs_state_.endpoints_to_connect.size();

        // Progress Connect and then Accept if needed.
        for (auto &endpoint_pair_conn_state : dxs_state_.endpoints_to_connect) {
            auto connect_status =
                (endpoint_pair_conn_state.connect_state == DxsConnectAcceptStates::kDone) ?
                absl::OkStatus() :
                ProgressDxsConnect(endpoint_pair_conn_state, dxs_connect_timeout_ms);

            auto accept_status =
                (endpoint_pair_conn_state.accept_state == DxsConnectAcceptStates::kDone) ?
                absl::OkStatus() :
                ProgressDxsAccept(endpoint_pair_conn_state, dxs_accept_timeout_ms);
            if (!connect_status.ok() || !accept_status.ok()) {
                NIXL_ERROR << "ProgressDxsConnect failed. Connect Status: " << connect_status
                           << " Accept Status: " << accept_status;
            }
            absl::SleepFor(absl::Milliseconds(100));
        }

        // Loop over all and delete ones which are completed.
        dxs_state_.endpoints_to_connect.erase(
            std::remove_if(dxs_state_.endpoints_to_connect.begin(),
                           dxs_state_.endpoints_to_connect.end(),
                           [](const DxsConnectionStatePerEndpointPair &ep) {
                               if (ep.connect_state == DxsConnectAcceptStates::kDone &&
                                   ep.accept_state == DxsConnectAcceptStates::kDone) {
                                   NIXL_INFO << "Connect Accept Dance Done for Endpoint Pair "
                                             << ep.endpoint_pair;
                                   return true;
                               }
                               return false;
                           }),
            dxs_state_.endpoints_to_connect.end());
    }

    if (const auto status = EstablishEndpointConnectionsAndDrainPendingOps(); !status.ok()) {
        NIXL_ERROR << "Failed to drain pending ops after establishing connections: " << status;
    }
}

bool
HostConnection::EstablishEndpointConnections(const absl::flat_hash_set<EndpointPair> &endpoints) {
    if (endpoints.empty()) {
        return false;
    }

    // Deduplicate endpoints and add them to our state tracking
    for (const auto &ep : endpoints) {
        if (std::find_if(dxs_state_.endpoints_to_connect.begin(),
                         dxs_state_.endpoints_to_connect.end(),
                         [&ep](const DxsConnectionStatePerEndpointPair &state) {
                             return state.endpoint_pair == ep;
                         }) == dxs_state_.endpoints_to_connect.end()) {
            dxs_state_.endpoints_to_connect.push_back({
                .connect_state = DxsConnectAcceptStates::kInitial,
                .accept_state = DxsConnectAcceptStates::kInitial,
                .connect_start_time = absl::InfinitePast(),
                .accept_start_time = absl::InfinitePast(),
                .endpoint_pair = ep,
            });
        }
    }

    // Start the state machine loop
    host_connection_task_cb_();

    return true;
}

absl::Status
HostConnection::IssueDxsOp(nixlTcpxoDxsOp &op) {
    NIXL_DEBUG << "HostConnection::IssueDxsOp called for op " << op.params();
    const auto &endpoint_pair = op.params().endpoint_pair;
    auto it = connection_map_.find(endpoint_pair);
    if (it == connection_map_.end()) {
        return absl::NotFoundError(absl::StrCat(
            "Connection for endpoint pair ",
            endpoint_pair,
            " missing. Did something go wrong when establishing the DXS connections?"));
    }

    auto &dxs_conn = it->second.conn_;
    if (op.flow_idx() >= dxs_conn.flows().size()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Invalid flow index (", op.flow_idx(), ") for DXS connection"));
    }

    auto &flow = dxs_conn.flows()[op.flow_idx()];
    std::unique_ptr<dxs::OpInterface> dxs_op{nullptr};
    if (op.op_type() == NIXL_WRITE) {
        auto &send_socket = flow.send_socket;
        if (!send_socket) {
            return absl::FailedPreconditionError(
                absl::StrCat("Send socket for flow index (", op.flow_idx(), ") is null."));
        }
        ASSIGN_OR_RETURN(dxs_op,
                         send_socket->Send(op.params().local_page_offset,
                                           op.params().local_mr.len,
                                           op.params().local_reg_handle));
    } else {
        auto &recv_socket = flow.recv_socket;
        if (!recv_socket) {
            return absl::FailedPreconditionError(
                absl::StrCat("Recv socket for flow index (", op.flow_idx(), ") is null."));
        }
        ASSIGN_OR_RETURN(dxs_op,
                         recv_socket->RecvLinearized(op.params().local_page_offset,
                                                     op.params().local_mr.len,
                                                     op.params().local_reg_handle));
    }
    op.set_dxs_op(std::move(dxs_op));
    op.MarkStarted();
    return absl::OkStatus();
}

absl::Status
HostConnection::EstablishEndpointConnectionsAndDrainPendingOps() {
    absl::flat_hash_set<EndpointPair> endpoints_to_establish;

    auto it = pending_xfers_.begin();
    while (it != pending_xfers_.end()) {
        auto *handle = *it;

        const auto status = HasRemoteListenHandles(handle->ops());
        if (status.code() == absl::StatusCode::kFailedPrecondition) {
            NIXL_DEBUG << "Still waiting for remote listen map from remote agent: " << remote_name_;
            return absl::OkStatus();
        } else if (!status.ok()) {
            return status;
        }

        if (!handle->xfer_msg_sent()) {
            auto msg_or_status = ConstructXferMessageAndAssignFlows(*handle);
            if (!msg_or_status.ok()) {
                NIXL_ERROR << "Failed to construct Xfer message for handle " << handle->id()
                           << " with status: " << msg_or_status.status();
                handle->MarkXferInactive(NIXL_ERR_BACKEND);
                it = pending_xfers_.erase(it);
                continue;
            }

            auto send_id_or_status =
                channel_.ScheduleSend(remote_handle_, *std::move(msg_or_status));
            if (!send_id_or_status.ok()) {
                NIXL_ERROR << "Failed to send Xfer message for handle " << handle->id()
                           << ". We are probably mid-connection teardown. Status: "
                           << send_id_or_status.status();
                handle->MarkXferInactive(NIXL_ERR_BACKEND);
                it = pending_xfers_.erase(it);
                continue;
            }
            handle->MarkXferMsgSent();
        }

        RETURN_IF_ERROR(FindUnconnectedEndpoints(handle->ops(), endpoints_to_establish));
        ++it;
    }

    RETURN_IF_ERROR(FindUnconnectedEndpoints(pending_remote_xfers_, endpoints_to_establish));

    const auto establish_connections_enqueued =
        EstablishEndpointConnections(endpoints_to_establish);
    if (establish_connections_enqueued) {
        NIXL_DEBUG << "Establishing connections for:";
        for (const auto &endpoints : endpoints_to_establish) {
            NIXL_DEBUG << "\t" << endpoints;
        }
        return absl::OkStatus();
    }

    // All connections are present, post the OPs
    if (!pending_remote_xfers_.empty()) {
        NIXL_DEBUG << "Issuing " << pending_remote_xfers_.size()
                   << " pending remote DXS ops for connection to " << remote_name_;
    }
    for (auto &op : pending_remote_xfers_) {
        auto it_conn = connection_map_.find(op.params().endpoint_pair);
        if (it_conn == connection_map_.end()) {
            NIXL_ERROR << "Xfer enqueued without a valid connection. This shouldn't be possible! "
                          "Missing connection: "
                       << op.params().endpoint_pair;
            continue;
        }

        auto &dxs_conn = it_conn->second.conn_;
        dxs_conn.set_last_flow_used(op.flow_idx());

        if (const auto status = IssueDxsOp(op); !status.ok()) {
            NIXL_ERROR << "Failed to issue remote DXS ops: " << status;
            continue;
        }
        remote_initiated_ops_.Enqueue(std::move(op));
    }
    pending_remote_xfers_.clear();

    it = pending_xfers_.begin();
    while (it != pending_xfers_.end()) {
        auto *handle = *it;
        const auto status = HasRemoteListenHandles(handle->ops());
        if (status.code() == absl::StatusCode::kFailedPrecondition) {
            const auto msg =
                absl::StrCat("Remote listen map is empty, but we discovered this after issuing the "
                             "XferMessage? This should not be possible! HostConnection for ",
                             remote_name_);
            NIXL_ERROR << msg;
            return absl::InternalError(msg);
        } else if (!status.ok()) {
            return status;
        }
        NIXL_DEBUG << "Resuming deferred postXfer for handle " << handle->id();
        if (const auto status = handle->postXfer(); !status.ok()) {
            NIXL_ERROR << "Failed to issue local DXS ops for handle " << handle->id()
                       << " with status: " << status;
        }
        it = pending_xfers_.erase(it);
    }
    return absl::OkStatus();
}

absl::Status
HostConnection::ParseRemoteListenMap(const DxsAddressExchangeMessage &msg) {
    for (const DxsEndpointListenInfo &listen_info : msg.endpoint_pairs()) {
        std::vector<DxsAddress> remote_listen_handles;
        remote_listen_handles.reserve(listen_info.listen_handles_size());
        for (const DxsAddress &listen_handle : listen_info.listen_handles()) {
            remote_listen_handles.push_back(listen_handle);
        }
        auto [it, inserted] = remote_listen_map_.insert(
            {EndpointPair{
                 // local_fastrak_idx on the target side should be remote_fastrak_idx on the
                 // initiator side, and vice versa.
                 .local_fastrak_idx = static_cast<uint64_t>(listen_info.remote_fastrak_idx()),
                 .remote_fastrak_idx = static_cast<uint64_t>(listen_info.local_fastrak_idx()),
             },
             std::move(remote_listen_handles)});
        if (!inserted) {
            return absl::InternalError(
                absl::StrFormat("Store listen handles error within %s", __func__));
        }
    }
    return absl::OkStatus();
}

// Request ID Generation
RequestId
nixlTcpxoBackendReqH::GenerateNextRequestId() {
    static std::atomic<uint64_t> next_request_id{0};
    uint64_t next_id = next_request_id.fetch_add(1, std::memory_order_relaxed);

    while (next_id == 0) {
        next_id = next_request_id.fetch_add(1, std::memory_order_relaxed);
    }

    return static_cast<RequestId>(next_id);
}

void
HostConnection::TestProgress() {
    // Process Locally Initiated Ops
    {
        absl::MutexLock lock(&request_map_mutex_);
        for (auto &request_map_entry : request_map_) {
            request_map_entry.second->checkXfer();
        }
    }

    // Process Remote Initiated Ops.
    //
    // We dequeue and process each op once. If the op is complete,
    // we are done. If the op is pending we enqueue it back but dont
    // process it again in this iteration. This is guaranteed by the
    // `loop_iter` logic.
    {
        size_t loop_iter = remote_initiated_ops_.size();
        for (size_t i = 0; i < loop_iter; ++i) {
            auto nixlop_or_status = remote_initiated_ops_.TryDequeue();
            if (!nixlop_or_status.ok()) {
                break;
            }
            auto nixlop = std::move(*nixlop_or_status);
            auto status = nixlop.checkXfer(slowness_threshold(), timeout_threshold());
            if (status == NIXL_IN_PROG) {
                // Op Pending
                remote_initiated_ops_.Enqueue(std::move(nixlop));
            } else if (status == NIXL_ERR_BACKEND) {
                // Op Error
                NIXL_ERROR << "Remote Initated Op Failed: " << status;
            } else {
                // Op Complete
            }
        }
    }
}

} // namespace tcpxo
