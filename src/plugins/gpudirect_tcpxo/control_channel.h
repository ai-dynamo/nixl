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

#ifndef GPUDIRECT_TCPXO_CONTROL_CHANNEL_H_
#define GPUDIRECT_TCPXO_CONTROL_CHANNEL_H_

#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <netinet/in.h>
#include <unistd.h>

#include <atomic>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"

#include "control_channel.pb.h"
#include "mpmc_queue.h"
#include "sliding_buffer.h"

namespace tcpxo {

inline constexpr uint8_t kMaxPendingConnections = 96;
inline constexpr size_t kSerializedExpectedLenBytes = sizeof(uint32_t);
// Protobuf has a 2GB serialized message size limit.
inline constexpr size_t kMaxSerializedProtobufSize = std::numeric_limits<int32_t>::max();
// This is a guess for the overhead for serializing a WorkloadNotification
inline constexpr size_t kNotificationSerializationOverheadBytes = 32;
inline constexpr size_t kMaxNotificiationSize =
    kMaxSerializedProtobufSize - kNotificationSerializationOverheadBytes;

enum class SendStatus : uint8_t {
    kInvalid = 0,
    kPending = 1,
    kCompleted = 2,
    kFailed = 3,
    kNotFound = 4,
};

enum class SendId : uint64_t {
    kInvalid = 0,
};

enum class PeerHandle : int {
    kInvalid = -1,
};

enum class IoResult : uint8_t {
    kSuccess = 0,
    kEagain = 1,
    kDisconnected = 2,
};

using AgentAddressKey = std::pair<std::string, uint16_t>;

class AgentAddress {
public:
    AgentAddress(IPAddress addr, uint16_t port) : addr_(addr), port_(port) {}

    AgentAddress() = default;
    // Copyable
    AgentAddress(const AgentAddress &other) = default;
    AgentAddress &
    operator=(const AgentAddress &other) = default;
    // Moveable
    AgentAddress(AgentAddress &&other) = default;
    AgentAddress &
    operator=(AgentAddress &&other) = default;

    template<typename Sink>
    friend void
    AbslStringify(Sink &sink, const AgentAddress &addr) {
        absl::Format(&sink, "%s:%" PRIu16, addr.ip(), addr.port_);
    }

    bool
    operator==(const AgentAddress &other) const {
        return port_ == other.port_ && ip() == other.ip();
    }

    bool
    operator!=(const AgentAddress &other) const {
        return !(*this == other);
    }

    inline AgentAddressKey
    MakeKey() const {
        return std::make_pair(ip(), port_);
    }

    inline const std::string &
    ip() const {
        return addr_.addr();
    }

    inline uint16_t
    port() const {
        return port_;
    }

private:
    IPAddress addr_;
    uint16_t port_;
};

class ManagedFileDescriptor {
public:
    ManagedFileDescriptor() = default;

    /**
     * @brief Construct a ManagedFileDescriptor and capture errno
     * @details
     * This class is intended for use with C-functions that return a file descriptor, and set errno.
     * For example:
     *
     * ```
     * ManagedFileDescriptor fd(::open(...));
     * ```
     *
     * This way, the class both gets the fd returned, as well as captures the errno of its
     * construction.
     */
    explicit ManagedFileDescriptor(int fd) : fd_{fd} {
        if (fd < 0) {
            fd_errno_ = errno;
        }
    }

    ~ManagedFileDescriptor() {
        Close();
    }

    // Not copyable
    ManagedFileDescriptor(const ManagedFileDescriptor &other) = delete;
    ManagedFileDescriptor &
    operator=(const ManagedFileDescriptor &other) = delete;

    // Moveable
    ManagedFileDescriptor(ManagedFileDescriptor &&other) noexcept
        : fd_{other.fd_},
          fd_errno_{other.fd_errno_} {
        other.fd_ = std::nullopt;
        other.fd_errno_ = std::nullopt;
    }

    ManagedFileDescriptor &
    operator=(ManagedFileDescriptor &&other) noexcept;

    inline const int &
    operator*() const {
        return *fd_;
    }

    inline std::optional<int>
    fd() const {
        return fd_;
    }

    inline std::optional<int>
    fd_errno() const {
        return fd_errno_;
    }

    inline bool
    HasValidFd() const {
        return fd_.has_value() && *fd_ >= 0;
    }

    inline void
    Close() {
        if (HasValidFd()) {
            ::close(*fd_);
            fd_ = std::nullopt;
        }
    }

private:
    std::optional<int> fd_{std::nullopt};
    std::optional<int> fd_errno_{std::nullopt};
};

class ManagedEpollRegistration {
public:
    ManagedEpollRegistration() = default;

    ManagedEpollRegistration(const ManagedFileDescriptor &epoll_fd,
                             const ManagedFileDescriptor &socket_fd,
                             uint32_t events);

    ~ManagedEpollRegistration() {
        Unregister();
    }

    // Not copyable
    ManagedEpollRegistration(const ManagedEpollRegistration &other) = delete;
    ManagedEpollRegistration &
    operator=(const ManagedEpollRegistration &other) = delete;

    // Moveable
    ManagedEpollRegistration &
    operator=(ManagedEpollRegistration &&other) noexcept;
    // Can be defined later if needed
    ManagedEpollRegistration(ManagedEpollRegistration &&other) = delete;

    inline std::optional<int>
    epoll_add_errno() const {
        return epoll_add_errno_;
    }

    inline std::optional<int>
    epoll_del_errno() const {
        return epoll_del_errno_;
    }

    void
    Unregister();

private:
    std::optional<int> epoll_fd_{std::nullopt};
    std::optional<int> socket_fd_{std::nullopt};

    std::optional<int> epoll_add_errno_{std::nullopt};
    std::optional<int> epoll_del_errno_{std::nullopt};
};

/**
 * @brief Represents a connected ControlChannel peer.
 * @details
 * Each peer consists of the sockets used to receive/send data, as well as any pending data to send
 * or receive
 */
class Peer {
public:
    struct PendingSend {
        const SendId id;
        std::vector<uint8_t> data;
        size_t bytes_written{0};
    };

    Peer(ManagedFileDescriptor &&fd,
         AgentAddress socket_addr,
         std::optional<AgentAddress> service_addr = std::nullopt)
        : fd_(std::move(fd)),
          socket_addr_(socket_addr),
          service_addr_(service_addr) {}

    inline const ManagedFileDescriptor &
    fd() const {
        return fd_;
    }

    inline const AgentAddress &
    socket_addr() const {
        return socket_addr_;
    }

    inline const std::optional<AgentAddress> &
    service_addr() const {
        return service_addr_;
    }

    inline void
    set_service_addr(const AgentAddress &addr) {
        service_addr_ = addr;
    }

    inline void
    set_epoll_reg(ManagedEpollRegistration &&epoll_reg) {
        epoll_reg_ = std::move(epoll_reg);
    }

    inline IoResult
    DoSocketRead();

    inline std::vector<uint8_t>
    TryPopFullMessage();

    inline const std::deque<PendingSend>
    send_queue() const {
        return send_queue_;
    }

    inline bool
    IsSendQueueEmpty() {
        return send_queue_.empty();
    }

    SendId
    EnqueueSend(std::vector<uint8_t> &&data);

    inline void
    PopLatestSend() {
        send_queue_.pop_back();
    }

    inline bool
    HasMessageToSend() const {
        return !send_queue_.empty();
    }

    inline IoResult
    DoSocketWrite();

    inline std::optional<SendId>
    TryPopSentMessage();

private:
    // Socket file descriptor
    const ManagedFileDescriptor fd_;
    const AgentAddress socket_addr_;
    std::optional<AgentAddress> service_addr_;

    /**
     * @brief Assigned after construction to avoid polling on unregistered Peers
     */
    ManagedEpollRegistration epoll_reg_;

    SlidingBuffer read_buf_;
    /**
     * @brief The length of the oldest message on read_buf
     * @details
     * This is used for us know how much data we want to read from read_buf, in case we haven't yet
     * received a full message.
     */
    std::optional<size_t> expected_len_{std::nullopt};
    std::deque<PendingSend> send_queue_;
};

struct BasePeerEvent {
    PeerHandle handle;
    AgentAddress socket_addr;
    AgentAddress service_addr;
};

struct ConnectedEvent : public BasePeerEvent {
    std::string agent_name;
    DxsAddressExchangeMessage dxs_msg;
};

struct DisconnectedEvent : public BasePeerEvent {};

struct MessageReceivedEvent : public BasePeerEvent {
    std::vector<uint8_t> msg_data;
};

using PeerEvent = std::variant<ConnectedEvent, DisconnectedEvent, MessageReceivedEvent>;
using EventCallback = absl::AnyInvocable<void(PeerEvent &&)>;

using ConnectionCallback = absl::AnyInvocable<void(PeerHandle,
                                                   const AgentAddress &,
                                                   const AgentAddress &,
                                                   const std::string &,
                                                   const DxsAddressExchangeMessage &)>;
using DisconnectionCallback =
    absl::AnyInvocable<void(PeerHandle, const AgentAddress &, const AgentAddress &)>;
// The message callbacks shouldn't ever be interested in the socket/service address difference
using NotificationCallback =
    absl::AnyInvocable<void(PeerHandle, const AgentAddress &, WorkloadNotificationMessage &&)>;
using DxsAddressExchangeCallback =
    absl::AnyInvocable<void(PeerHandle, const AgentAddress &, const DxsAddressExchangeMessage &)>;
using XferCallback =
    absl::AnyInvocable<void(PeerHandle, const AgentAddress &, const XferMessage &)>;

struct DxsConnectionEstablishTask {
    std::string remote_name;
};

using HostConnectionTask = std::variant<DxsConnectionEstablishTask>;
using DxsConnectionEstablishCallback = absl::AnyInvocable<void(const std::string &)>;

using WorkerTask = std::variant<PeerEvent, HostConnectionTask>;

/**
 * @brief This worker class receives messages from the ControlChannel and processes them.
 * @details
 * See `control_channel.proto` for the type of messages that can be received.
 */
class Worker {
public:
    static std::pair<Message, bool>
    ParseMessage(const std::vector<uint8_t> &msg_data);

    Worker(ConnectionCallback absl_nonnull conn_cb,
           DisconnectionCallback absl_nonnull disc_cb,
           NotificationCallback absl_nonnull notif_cb,
           DxsAddressExchangeCallback absl_nonnull dxs_cb,
           XferCallback absl_nonnull xfer_cb,
           DxsConnectionEstablishCallback absl_nonnull dxs_conn_est_cb)
        : connection_callback_(std::move(conn_cb)),
          disconnection_callback_(std::move(disc_cb)),
          notification_callback_(std::move(notif_cb)),
          dxs_address_exchange_callback_(std::move(dxs_cb)),
          xfer_callback_(std::move(xfer_cb)),
          dxs_connection_establish_callback_(std::move(dxs_conn_est_cb)) {}

    void
    Start();

    void
    EnqueueEvent(PeerEvent &&);

    void
    EnqueueHostConnectionTask(HostConnectionTask &&);

    void
    Stop();

    // The intended design is that the control channel finds the worker with the least work, and
    // enqueues the message onto that worker
    inline size_t
    pending_task_count() const {
        return pending_task_count_.load();
    }

private:
    void
    ProcessEventLoop();

    absl::Mutex tasks_mutex_;
    std::deque<WorkerTask> pending_tasks_ ABSL_GUARDED_BY(tasks_mutex_);
    std::atomic<size_t> pending_task_count_{0};

    ConnectionCallback absl_nonnull connection_callback_;
    DisconnectionCallback absl_nonnull disconnection_callback_;
    NotificationCallback absl_nonnull notification_callback_;
    DxsAddressExchangeCallback absl_nonnull dxs_address_exchange_callback_;
    XferCallback absl_nonnull xfer_callback_;
    DxsConnectionEstablishCallback absl_nonnull dxs_connection_establish_callback_;

    absl::Notification cancel_notif_;
    std::thread message_thread_;
};

/**
 * @brief This class serves as the backend communication between two nodes.
 * @details
 * It reads messages from connected clients, and sends them to workers for processing. It sends
 * messages passed to it by its caller.
 */
class ControlChannel {
public:
    using CommandFunction = absl::AnyInvocable<void()>;
    ControlChannel(std::string nic_name,
                   absl::Duration heartbeat_send_period,
                   absl::Duration heartbeat_timeout,
                   EventCallback absl_nonnull event_callback);

    ~ControlChannel() {
        Stop();
    }

    // Thread safe
    void
    Stop();
    /**
     * @brief Start the control channel
     * @details
     * Initializes the listening socket and starts the epoll thread
     *
     * Note: This function is not thread safe, is presumed to be called only once, and presumed to
     * be called before any function below. Functions below expect values set by this function to
     * already be set.
     */
    absl::StatusOr<AgentAddress>
    Listen();

    // Thread safe
    absl::StatusOr<PeerHandle>
    Connect(const AgentAddress &, const std::string &, std::vector<DxsEndpointListenInfo>);

    // Thread safe
    absl::Status Disconnect(PeerHandle);

    // Thread safe
    absl::StatusOr<SendId>
    ScheduleSend(PeerHandle, const Message &);

    // Thread safe
    absl::StatusOr<SendId>
    ExchangeDxsAddress(PeerHandle handle, std::vector<DxsEndpointListenInfo> endpoint_listen_infos);

    // Thread safe
    SendStatus
    CheckSendStatus(SendId send_id);

    // Thread safe
    absl::StatusOr<bool>
    IsConnectionReady(PeerHandle handle);

    // Won't be valid until Listen() is called
    inline const AgentAddress &
    GetServiceAddress() const {
        return service_addr_;
    }

private:
    absl::Status
    RunOnEpollThread(CommandFunction cmd);

    absl::StatusOr<SendId>
    ScheduleSendInternal(Peer &, std::vector<uint8_t> &&) ABSL_SHARED_LOCKS_REQUIRED(epoll_mutex_);

    absl::Status DisconnectInternal(PeerHandle) ABSL_SHARED_LOCKS_REQUIRED(epoll_mutex_);

    void HandlePeerRead(PeerHandle) ABSL_SHARED_LOCKS_REQUIRED(epoll_mutex_);

    void HandlePeerWrite(PeerHandle) ABSL_SHARED_LOCKS_REQUIRED(epoll_mutex_);

    void
    AcceptPeers() ABSL_SHARED_LOCKS_REQUIRED(epoll_mutex_);

    /**
     * @brief Execute all queued commands
     * @details
     * We prioritize local command latency over responding over processing network events
     */
    void
    DrainCommands() ABSL_SHARED_LOCKS_REQUIRED(epoll_mutex_);

    void
    LoopEpoll();

    absl::StatusOr<Peer * absl_nonnull> FindPeer(PeerHandle)
        ABSL_SHARED_LOCKS_REQUIRED(epoll_mutex_);

    void
    FailPendingSends(Peer &) ABSL_SHARED_LOCKS_REQUIRED(epoll_mutex_);

    const std::string control_nic_dev_name_;
    // How often to send heartbeats
    const absl::Duration heartbeat_send_period_;
    const absl::Duration heartbeat_timeout_;

    /**
     * Compile time guarantee that _only_ the epoll thread has access to certain functions or data
     * structures
     */
    absl::Mutex epoll_mutex_;

    /**
     * @brief Safely destruct the control channel.
     * @details
     * Used to synchronize between hypothetical destructor invocations coinciding with calls to
     * Stop()
     */
    absl::Mutex stopping_mutex_;
    bool stopped_ ABSL_GUARDED_BY(stopping_mutex_){false};

    AgentAddress service_addr_;

    struct ::in_addr binary_service_ip_ {
        0
    };

    absl::flat_hash_map<PeerHandle, std::unique_ptr<Peer>> peers_;
    absl::flat_hash_map<PeerHandle, std::unique_ptr<Peer>> prospective_peers_;

    absl::flat_hash_set<PeerHandle> ready_to_read_;
    absl::flat_hash_set<PeerHandle> ready_to_write_;
    absl::flat_hash_set<PeerHandle> ready_to_write_without_data_;

    ManagedFileDescriptor listen_fd_;
    ManagedFileDescriptor epoll_fd_;
    ManagedEpollRegistration listen_epoll_registration_;

    MPMCQueue<CommandFunction> command_queue_;
    ManagedFileDescriptor command_fd_;
    ManagedEpollRegistration command_epoll_registration_;

    absl::Notification stop_notif_;

    EventCallback absl_nonnull event_callback_;

    /**
     * @brief Used to synchronize between callers of CheckSendStatus and the epoll thread
     */
    absl::Mutex send_statuses_mutex_;
    absl::flat_hash_map<SendId, SendStatus> send_statuses_ ABSL_GUARDED_BY(send_statuses_mutex_);

    std::thread epoll_thread_;
};

SendId
GenerateSendId();

inline std::ostream &
operator<<(std::ostream &os, const AgentAddress &addr) {
    os << addr.ip() << ":" << addr.port();
    return os;
}

inline std::ostream &
operator<<(std::ostream &os, const PeerHandle &handle) {
    os << "PeerHandle:" << static_cast<int>(handle);
    return os;
}

} // namespace tcpxo

#endif // GPUDIRECT_TCPXO_CONTROL_CHANNEL_H_
