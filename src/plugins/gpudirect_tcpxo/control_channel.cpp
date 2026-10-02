/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC. All rights reserved.
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

#include "control_channel.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "common/nixl_log.h"

#include "control_channel.pb.h"

namespace tcpxo {

static constexpr size_t kMaxEventsPerPoll = 64;
static constexpr absl::Duration kEventPollTimeout = absl::Milliseconds(100);
static constexpr size_t kMaxSocketReadBytes = 1 << 13;
static constexpr absl::Duration kCommandRewriteDelay = absl::Milliseconds(1);

namespace {
    using WriteStatus = std::pair<ssize_t, absl::Status>;

    absl::Status
    SetTcpKeepalive(int fd, absl::Duration send_period, absl::Duration timeout) {
        int enable = 1;
        if (::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable)) < 0) {
            return absl::ErrnoToStatus(errno, "failed to enable SO_KEEPALIVE");
        }

        int idle_begin = absl::ToInt64Seconds(send_period);
        if (::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle_begin, sizeof(idle_begin)) < 0) {
            return absl::ErrnoToStatus(errno, "failed to set TCP_KEEPIDLE");
        }

        int period = absl::ToInt64Seconds(send_period);
        if (::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &period, sizeof(period)) < 0) {
            return absl::ErrnoToStatus(errno, "failed to set TCP_KEEPINTVL");
        }

        int num_max_failed_probes = std::ceil(static_cast<double>(absl::ToInt64Seconds(timeout)) /
                                              static_cast<double>(period));
        if (::setsockopt(fd,
                         IPPROTO_TCP,
                         TCP_KEEPCNT,
                         &num_max_failed_probes,
                         sizeof(num_max_failed_probes)) < 0) {
            return absl::ErrnoToStatus(errno, "failed to set TCP_KEEPCNT");
        }

        return absl::OkStatus();
    }

    absl::Status
    SetNonBlocking(int fd) {
        const auto flags = ::fcntl(fd, F_GETFL, 0);
        if (flags < 0) {
            return absl::ErrnoToStatus(errno, "failed to retrieve socket flags");
        }
        if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
            return absl::ErrnoToStatus(errno, "failed to add `O_NONBLOCK` to socket flags");
        }
        return absl::OkStatus();
    }

    absl::StatusOr<std::pair<IPAddress, struct ::in_addr>>
    GetNicIpAddress(absl::string_view nic_name) {
        IPAddress ip_addr;
        struct ::in_addr binary_ip_addr;

        struct ifaddrs *ifaddr_head, *ifaddr_itr;
        if (::getifaddrs(&ifaddr_head) < 0) {
            return absl::ErrnoToStatus(errno, "getifaddrs failed");
        }
        absl::Cleanup ifaddr_closer = [ifaddr_head] { ::freeifaddrs(ifaddr_head); };

        bool found_ip = false;
        for (ifaddr_itr = ifaddr_head; ifaddr_itr != nullptr; ifaddr_itr = ifaddr_itr->ifa_next) {
            if (ifaddr_itr->ifa_addr == nullptr) {
                continue;
            }
            // IPv4 only for internal-only A3* VPCs
            if (ifaddr_itr->ifa_addr->sa_family != AF_INET) {
                continue;
            }
            if (ifaddr_itr->ifa_name != nic_name) {
                continue;
            }

            struct sockaddr_in *sa = (struct sockaddr_in *)ifaddr_itr->ifa_addr;
            std::array<char, INET_ADDRSTRLEN> ip_str;
            if (::inet_ntop(AF_INET, &sa->sin_addr, ip_str.data(), ip_str.size()) == nullptr) {
                return absl::ErrnoToStatus(errno, "failed to convert binary IP to string format");
            }

            ip_addr.set_addr(ip_str.data());
            binary_ip_addr = sa->sin_addr;
            found_ip = true;
            break;
        }

        if (!found_ip) {
            return absl::NotFoundError(absl::StrCat("Could not find IP for NIC: ", nic_name));
        }
        return std::make_pair(ip_addr, binary_ip_addr);
    }

    std::vector<uint8_t>
    SerializeMsg(const Message &msg) {
        const auto serialized_msg_size = static_cast<uint32_t>(msg.ByteSizeLong());
        const uint32_t net_len = ::htonl(serialized_msg_size);
        std::vector<uint8_t> data(serialized_msg_size + kSerializedExpectedLenBytes);

        *reinterpret_cast<uint32_t *>(data.data()) = net_len;
        msg.SerializeToArray(data.data() + kSerializedExpectedLenBytes, serialized_msg_size);

        return data;
    }

    std::vector<uint8_t>
    CreateIdentityMessagePayload(const AgentAddress &service_addr,
                                 const std::string &agent_name,
                                 std::vector<DxsEndpointListenInfo> endpoint_listen_infos) {
        Message msg;
        IdentityMessage *identity = msg.mutable_identity();
        identity->mutable_addr()->set_addr(service_addr.ip());
        identity->set_port(service_addr.port());
        identity->set_agent_name(agent_name);
        DxsAddressExchangeMessage *dxs = identity->mutable_dxs();
        for (auto &endpoint_listen_info : endpoint_listen_infos) {
            *dxs->add_endpoint_pairs() = std::move(endpoint_listen_info);
        }
        return SerializeMsg(msg);
    }
} // namespace

// Managed Resources Implementations

ManagedFileDescriptor &
ManagedFileDescriptor::operator=(ManagedFileDescriptor &&other) noexcept {
    if (this != &other) {
        Close();
        fd_ = other.fd_;
        fd_errno_ = other.fd_errno_;
        other.fd_ = std::nullopt;
        other.fd_errno_ = std::nullopt;
    }
    return *this;
}

// Takes just the underlying FDs, because the ManagedFileDescriptors usually move after this is
// created
ManagedEpollRegistration::ManagedEpollRegistration(const ManagedFileDescriptor &epoll_fd,
                                                   const ManagedFileDescriptor &socket_fd,
                                                   uint32_t events)
    : epoll_fd_{*epoll_fd},
      socket_fd_{*socket_fd} {
    struct epoll_event ev;
    ev.events = events;
    ev.data.fd = *socket_fd_;
    if (::epoll_ctl(*epoll_fd_, EPOLL_CTL_ADD, *socket_fd_, &ev) < 0) {
        epoll_add_errno_ = errno;
    }
}

ManagedEpollRegistration &
ManagedEpollRegistration::operator=(ManagedEpollRegistration &&other) noexcept {
    if (this != &other) {
        Unregister();

        epoll_fd_ = other.epoll_fd_;
        socket_fd_ = other.socket_fd_;
        epoll_add_errno_ = other.epoll_add_errno_;
        epoll_del_errno_ = other.epoll_del_errno_;

        other.epoll_fd_ = std::nullopt;
        other.socket_fd_ = std::nullopt;
        other.epoll_add_errno_ = std::nullopt;
        other.epoll_del_errno_ = std::nullopt;
    }
    return *this;
}

void
ManagedEpollRegistration::Unregister() {
    if (epoll_fd_.has_value() && socket_fd_.has_value()) {
        if (::epoll_ctl(*epoll_fd_, EPOLL_CTL_DEL, *socket_fd_, nullptr) < 0) {
            epoll_del_errno_ = errno;
        }
    }
    epoll_fd_ = std::nullopt;
    socket_fd_ = std::nullopt;
}

// Worker implementation

std::pair<Message, bool>
Worker::ParseMessage(const std::vector<uint8_t> &msg_data) {
    uint32_t net_len;
    std::memcpy(&net_len, msg_data.data(), kSerializedExpectedLenBytes);
    size_t expected_len = ::ntohl(net_len);

    Message msg;
    bool is_valid_message =
        msg.ParseFromArray(msg_data.data() + kSerializedExpectedLenBytes, expected_len);

    return {std::move(msg), is_valid_message};
}

void
Worker::Start() {
    message_thread_ = std::thread(&Worker::ProcessEventLoop, this);
}

void
Worker::EnqueueEvent(PeerEvent &&event) {
    absl::MutexLock lock(tasks_mutex_);

    pending_tasks_.push_back(std::move(event));
    pending_task_count_++;
}

void
Worker::EnqueueHostConnectionTask(HostConnectionTask &&task) {
    absl::MutexLock lock(tasks_mutex_);

    pending_tasks_.push_front(std::move(task));
    pending_task_count_++;
}

void
Worker::Stop() {
    cancel_notif_.Notify();
    // Lock and release the mutex to wake up our condition waiters
    { absl::MutexLock lock(tasks_mutex_); }
    if (message_thread_.joinable()) {
        message_thread_.join();
    }
}

void
Worker::ProcessEventLoop() {
    while (!cancel_notif_.HasBeenNotified()) {
        WorkerTask task;
        {
            absl::MutexLock lock(tasks_mutex_);
            auto event_present_or_cancelled_cond =
                [this]() ABSL_SHARED_LOCKS_REQUIRED(tasks_mutex_) {
                    return !pending_tasks_.empty() || cancel_notif_.HasBeenNotified();
                };
            tasks_mutex_.Await(absl::Condition(&event_present_or_cancelled_cond));

            if (cancel_notif_.HasBeenNotified()) {
                break;
            }
            if (pending_tasks_.empty()) {
                continue;
            }

            task = std::move(pending_tasks_.front());
            pending_tasks_.pop_front();
            pending_task_count_--;
        }

        // https://medium.com/@weidagang/modern-c-std-variant-and-std-visit-3c16084db7dc
        std::visit(
            [this](auto &&task_variant) {
                using VariantT = std::decay_t<decltype(task_variant)>;
                if constexpr (std::is_same_v<VariantT, HostConnectionTask>) {
                    std::visit(
                        [this](auto &&event) {
                            using T = std::decay_t<decltype(event)>;
                            if constexpr (std::is_same_v<T, DxsConnectionEstablishTask>) {
                                dxs_connection_establish_callback_(event.remote_name);
                            }
                        },
                        task_variant);
                } else if constexpr (std::is_same_v<VariantT, PeerEvent>) {
                    std::visit(
                        [this](auto &&event) {
                            using T = std::decay_t<decltype(event)>;
                            if constexpr (std::is_same_v<T, ConnectedEvent>) {
                                connection_callback_(event.handle,
                                                     event.socket_addr,
                                                     event.service_addr,
                                                     event.agent_name,
                                                     event.dxs_msg);
                            } else if constexpr (std::is_same_v<T, DisconnectedEvent>) {
                                disconnection_callback_(
                                    event.handle, event.socket_addr, event.service_addr);
                            } else if constexpr (std::is_same_v<T, MessageReceivedEvent>) {
                                auto [msg, is_valid_message] = Worker::ParseMessage(event.msg_data);

                                if (!is_valid_message) {
                                    NIXL_ERROR << "Failed to parse message";
                                    return;
                                }

                                if (msg.has_notif()) {
                                    notification_callback_(event.handle,
                                                           event.service_addr,
                                                           std::move(*msg.mutable_notif()));
                                } else if (msg.has_dxs()) {
                                    dxs_address_exchange_callback_(
                                        event.handle, event.service_addr, msg.dxs());
                                } else if (msg.has_xfer()) {
                                    xfer_callback_(event.handle, event.service_addr, msg.xfer());
                                }
                            }
                        },
                        task_variant);
                }
            },
            task);
    }
}

// Peer implementation

inline SendId
Peer::EnqueueSend(std::vector<uint8_t> &&data) {
    const SendId send_id = GenerateSendId();
    send_queue_.push_back({send_id, std::move(data), 0});
    return send_id;
}

inline IoResult
Peer::DoSocketRead() {
    read_buf_.reserve(kMaxSocketReadBytes);
    ssize_t bytes_read = ::recv(*fd_, read_buf_.write_head(), read_buf_.GetFreeSpace(), 0);
    if (bytes_read > 0) {
        read_buf_.CommitWrite(bytes_read);
        return IoResult::kSuccess;
    } else if (bytes_read == 0) {
        return IoResult::kDisconnected;
    } else {
        if (errno == EINTR) {
            return IoResult::kSuccess;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return IoResult::kEagain;
        } else {
            NIXL_PWARN << "failed to recv from socket";
            return IoResult::kDisconnected;
        }
    }
}

inline absl::StatusOr<std::vector<uint8_t>>
Peer::TryPopFullMessage() {
    // We're expecting a brand new message
    if (!expected_len_.has_value()) {
        // We don't even have enough bytes to read off the expected length
        if (read_buf_.GetReadableSpace() < kSerializedExpectedLenBytes) {
            return std::vector<uint8_t>{};
        }

        uint32_t net_len;
        std::memcpy(&net_len, read_buf_.read_head(), kSerializedExpectedLenBytes);
        uint32_t msg_len = ::ntohl(net_len);
        if (msg_len > kMaxSerializedProtobufSize) {
            return absl::InvalidArgumentError(
                absl::StrCat("Invalid control channel message length: ", msg_len));
        }
        expected_len_ = msg_len;
        read_buf_.reserve(kSerializedExpectedLenBytes + kMaxSocketReadBytes + *expected_len_);
    }

    // Have we read enough data for the complete message?
    if (read_buf_.GetReadableSpace() < kSerializedExpectedLenBytes + *expected_len_) {
        return std::vector<uint8_t>{};
    }

    size_t total_msg_size = kSerializedExpectedLenBytes + *expected_len_;
    std::vector<uint8_t> payload_to_parse(read_buf_.read_head(),
                                          read_buf_.read_head() + total_msg_size);
    read_buf_.Consume(total_msg_size);
    expected_len_ = std::nullopt;

    return payload_to_parse;
}

inline IoResult
Peer::DoSocketWrite() {
    auto &pending = send_queue_.front();
    ssize_t written = ::send(*fd_,
                             pending.data.data() + pending.bytes_written,
                             pending.data.size() - pending.bytes_written,
                             0);
    if (written > 0) {
        pending.bytes_written += written;
        return IoResult::kSuccess;
    } else if (written == -1) {
        if (errno == EINTR) {
            return IoResult::kSuccess;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return IoResult::kEagain;
        } else {
            NIXL_PWARN << "failed to send to socket";
            return IoResult::kDisconnected;
        }
    }
    return IoResult::kSuccess;
}

inline std::optional<SendId>
Peer::TryPopSentMessage() {
    auto &pending = send_queue_.front();
    if (pending.bytes_written < pending.data.size()) {
        return std::nullopt;
    }

    const auto send_id = pending.id;
    send_queue_.pop_front();
    return send_id;
}

// Control channel implementation

ControlChannel::ControlChannel(std::string nic_name,
                               absl::Duration heartbeat_send_period,
                               absl::Duration heartbeat_timeout,
                               EventCallback absl_nonnull event_callback)
    : control_nic_dev_name_(nic_name),
      heartbeat_send_period_(heartbeat_send_period),
      heartbeat_timeout_(heartbeat_timeout),
      event_callback_(std::move(event_callback)) {}

absl::Status
ControlChannel::RunOnEpollThread(CommandFunction cmd) {
    if (!command_fd_.HasValidFd()) {
        return absl::UnavailableError("Call Listen() before running functions on the epoll thread");
    }
    absl::MutexLock stopping_lock(stopping_mutex_);
    if (stop_notif_.HasBeenNotified()) {
        return absl::AbortedError(
            "The ControlChannel is stopped. Will not enqueue dead-end message.");
    }

    command_queue_.Enqueue(std::move(cmd));
    uint64_t val = 1;
    while (true) {
        if (::write(*command_fd_, &val, sizeof(val)) < 0) {
            // Unlikely for an eventfd with such small writes, but let's be comprehensive
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                absl::SleepFor(kCommandRewriteDelay);
                continue;
            }

            return absl::InternalError("Failed to notify epoll thread of pending command");
        }
        break;
    }

    return absl::OkStatus();
}

void
ControlChannel::Stop() {
    absl::MutexLock stopping_lock(stopping_mutex_);

    // Ensure this is only called once
    if (stopped_) {
        return;
    }

    stop_notif_.Notify();
    if (command_fd_.HasValidFd()) {
        uint64_t val = 1;
        while (true) {
            if (::write(*command_fd_, &val, sizeof(val)) < 0) {
                if (errno == EINTR) {
                    continue;
                }
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    absl::SleepFor(kCommandRewriteDelay);
                    continue;
                }
                NIXL_ERROR << "Failed to send command notification to epoll thread";
            }
            break;
        }
    }

    if (epoll_thread_.joinable()) {
        epoll_thread_.join();
    }

    stopped_ = true;
}

absl::StatusOr<AgentAddress>
ControlChannel::Listen() {
    ManagedFileDescriptor listen_fd{::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0)};
    if (listen_fd.fd_errno().has_value()) {
        return absl::ErrnoToStatus(*listen_fd.fd_errno(), "failed to create socket");
    }

    int opt = 1;
    if (::setsockopt(*listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        return absl::ErrnoToStatus(
            errno, absl::StrFormat("failed to set SO_REUSEADDR on socket %d", *listen_fd));
    }

    auto control_nic_addr_or_status = GetNicIpAddress(control_nic_dev_name_);
    // This should not fail, but we'll fail without crashing for now
    if (!control_nic_addr_or_status.ok()) {
        return control_nic_addr_or_status.status();
    }
    auto control_nic_addr = std::move(*control_nic_addr_or_status);

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr = control_nic_addr.second;
    addr.sin_port = 0;
    if (::bind(*listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return absl::ErrnoToStatus(errno, "failed to bind socket");
    }
    if (::listen(*listen_fd, kMaxPendingConnections) < 0) {
        return absl::ErrnoToStatus(errno, "failed to listen on socket");
    }

    socklen_t addr_len = sizeof(addr);
    if (::getsockname(*listen_fd, (struct sockaddr *)&addr, &addr_len) < 0) {
        return absl::ErrnoToStatus(errno, "failed to retrieve socket IP address and port");
    }

    ManagedFileDescriptor epoll_fd{::epoll_create1(0)};
    if (epoll_fd.fd_errno().has_value()) {
        return absl::ErrnoToStatus(*epoll_fd.fd_errno(), "failed to create epoll");
    }
    ManagedEpollRegistration listen_epoll_registration{epoll_fd, listen_fd, EPOLLIN | EPOLLET};
    if (listen_epoll_registration.epoll_add_errno().has_value()) {
        return absl::ErrnoToStatus(*listen_epoll_registration.epoll_add_errno(),
                                   "failed to add listen socket to epoll tracking");
    }

    ManagedFileDescriptor command_fd{::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)};
    if (command_fd.fd_errno().has_value()) {
        return absl::ErrnoToStatus(*command_fd.fd_errno(),
                                   "failed to create eventfd for command queueing");
    }
    ManagedEpollRegistration command_epoll_registration{epoll_fd, command_fd, EPOLLIN | EPOLLET};
    if (command_epoll_registration.epoll_add_errno().has_value()) {
        return absl::ErrnoToStatus(*command_epoll_registration.epoll_add_errno(),
                                   "failed to add our command queueing file descriptor to epoll");
    }

    service_addr_ = AgentAddress(control_nic_addr.first, ::ntohs(addr.sin_port));
    binary_service_ip_ = control_nic_addr.second;
    listen_fd_ = std::move(listen_fd);
    epoll_fd_ = std::move(epoll_fd);
    listen_epoll_registration_ = std::move(listen_epoll_registration);
    command_fd_ = std::move(command_fd);
    command_epoll_registration_ = std::move(command_epoll_registration);

    epoll_thread_ = std::thread(&ControlChannel::LoopEpoll, this);

    return service_addr_;
}

absl::StatusOr<PeerHandle>
ControlChannel::Connect(const AgentAddress &addr,
                        const std::string &agent_name,
                        std::vector<DxsEndpointListenInfo> endpoint_listen_infos) {
    ManagedFileDescriptor peer_fd{::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0)};
    if (peer_fd.fd_errno().has_value()) {
        return absl::ErrnoToStatus(*peer_fd.fd_errno(), "socket failed");
    }

    struct sockaddr_in sin;
    std::memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr = binary_service_ip_;
    if (::bind(*peer_fd, (struct sockaddr *)&sin, sizeof(sin)) < 0) {
        return absl::ErrnoToStatus(errno, "failed to bind peer socket to control nic");
    }

    std::memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = ::htons(addr.port());
    if (::inet_pton(AF_INET, addr.ip().c_str(), &sin.sin_addr) <= 0) {
        return absl::ErrnoToStatus(errno, absl::StrCat("invalid IP address", addr.ip()));
    }

    // Since it's non-blocking, connect might return EINPROGRESS
    if (::connect(*peer_fd, (struct sockaddr *)&sin, sizeof(sin)) < 0) {
        if (errno != EINPROGRESS) {
            return absl::ErrnoToStatus(errno, "connect failed");
        }
    }

    if (const auto status = SetTcpKeepalive(*peer_fd, heartbeat_send_period_, heartbeat_timeout_);
        !status.ok()) {
        return status;
    }

    auto peer = std::make_unique<Peer>(std::move(peer_fd), addr, addr);
    const PeerHandle handle = static_cast<PeerHandle>(*peer->fd());
    auto id_payload =
        CreateIdentityMessagePayload(service_addr_, agent_name, std::move(endpoint_listen_infos));

    absl::Notification done;
    absl::Status result = absl::OkStatus();

    const auto status = RunOnEpollThread([this,
                                          &done,
                                          &result,
                                          peer = std::move(peer),
                                          handle,
                                          id_payload = std::move(id_payload)]() mutable {
        auto [it, unused_inserted] = peers_.try_emplace(handle, std::move(peer));
        auto &peer_ptr = it->second;
        ManagedEpollRegistration peer_epoll_reg{
            epoll_fd_, peer_ptr->fd(), EPOLLIN | EPOLLOUT | EPOLLET};
        if (peer_epoll_reg.epoll_add_errno().has_value()) {
            result = absl::ErrnoToStatus(*peer_epoll_reg.epoll_add_errno(),
                                         "failed to add peer socket to epoll tracking");
            peers_.erase(it);
            done.Notify();
            return;
        }
        peer_ptr->set_epoll_reg(std::move(peer_epoll_reg));

        auto send_id_or_status = ScheduleSendInternal(*peer_ptr, std::move(id_payload));
        if (!send_id_or_status.ok()) {
            NIXL_ERROR << "Failed to send self-id to newly created peer: "
                       << send_id_or_status.status();
            result = send_id_or_status.status();
            peers_.erase(it);
        } else {
            // Immediately reap the send, as no one will be looking for it
            absl::MutexLock lock(send_statuses_mutex_);
            send_statuses_.erase(*send_id_or_status);
        }

        done.Notify();
    });
    if (!status.ok()) {
        return status;
    }

    done.WaitForNotification();
    if (!result.ok()) {
        return result;
    }
    return handle;
}

absl::Status
ControlChannel::Disconnect(PeerHandle handle) {
    absl::Notification done;
    absl::Status result;

    const auto status = RunOnEpollThread([this, handle, &done, &result]() {
        result = DisconnectInternal(handle);
        done.Notify();
    });
    if (!status.ok()) {
        return status;
    }

    done.WaitForNotification();
    return result;
}

absl::StatusOr<SendId>
ControlChannel::ScheduleSend(PeerHandle handle, const Message &msg) {
    // Serialize on the caller's thread
    auto data = SerializeMsg(msg);

    absl::Notification done;
    absl::StatusOr<SendId> result;

    const auto status =
        RunOnEpollThread([this, handle, data = std::move(data), &done, &result]() mutable {
            auto peer_ptr_or_status = FindPeer(handle);
            if (!peer_ptr_or_status.ok()) {
                result = peer_ptr_or_status.status();
                done.Notify();
                return;
            }

            result = ScheduleSendInternal(**peer_ptr_or_status, std::move(data));
            done.Notify();
        });
    if (!status.ok()) {
        return status;
    }

    done.WaitForNotification();
    return result;
}

absl::StatusOr<SendId>
ControlChannel::ExchangeDxsAddress(PeerHandle handle,
                                   std::vector<DxsEndpointListenInfo> endpoint_listen_infos) {
    Message msg;
    DxsAddressExchangeMessage *dxs_payload = msg.mutable_dxs();
    for (auto &listen_info : endpoint_listen_infos) {
        *dxs_payload->add_endpoint_pairs() = std::move(listen_info);
    }

    return ScheduleSend(handle, msg);
}

absl::StatusOr<SendId>
ControlChannel::ScheduleSendInternal(Peer &peer, std::vector<uint8_t> &&data) {
    auto send_id = peer.EnqueueSend(std::move(data));

    PeerHandle handle = static_cast<PeerHandle>(*peer.fd());
    if (ready_to_write_without_data_.contains(handle)) {
        ready_to_write_without_data_.erase(handle);
        ready_to_write_.insert(handle);
    }
    absl::MutexLock lock(send_statuses_mutex_);
    send_statuses_[send_id] = SendStatus::kPending;

    return send_id;
}

SendStatus
ControlChannel::CheckSendStatus(SendId send_id) {
    absl::MutexLock lock(send_statuses_mutex_);

    auto it = send_statuses_.find(send_id);
    if (it == send_statuses_.end()) {
        return SendStatus::kNotFound;
    }

    const auto status = it->second;
    if (status == SendStatus::kFailed || status == SendStatus::kCompleted) {
        send_statuses_.erase(it);
    }

    return status;
}

absl::StatusOr<bool>
ControlChannel::IsConnectionReady(PeerHandle handle) {
    absl::Notification done;
    absl::StatusOr<bool> result;

    const auto status = RunOnEpollThread([this, handle, &done, &result]() {
        auto peer_ptr_or_status = FindPeer(handle);
        if (!peer_ptr_or_status.ok()) {
            result = peer_ptr_or_status.status();
            done.Notify();
            return;
        }

        auto *peer_ptr = *peer_ptr_or_status;
        if (!peer_ptr->service_addr().has_value()) {
            result = false;
            done.Notify();
            return;
        }

        struct sockaddr_in sin;
        socklen_t len = sizeof(sin);
        if (::getpeername(*peer_ptr->fd(), (struct sockaddr *)&sin, &len) == 0) {
            result = true;
        } else {
            result = false;
        }

        done.Notify();
    });
    if (!status.ok()) {
        return status;
    }

    done.WaitForNotification();
    return result;
}

absl::Status
ControlChannel::DisconnectInternal(PeerHandle handle) {
    std::unique_ptr<Peer> peer_ptr;
    bool was_connected = false;

    if (auto it = peers_.find(handle); it != peers_.end()) {
        peer_ptr = std::move(it->second);
        peers_.erase(it);
        was_connected = true;
    } else if (auto prospect_it = prospective_peers_.find(handle);
               prospect_it != prospective_peers_.end()) {
        peer_ptr = std::move(prospect_it->second);
        prospective_peers_.erase(prospect_it);
    } else {
        return absl::NotFoundError(absl::StrCat("Peer with handle ", handle, " not found"));
    }

    ready_to_read_.erase(handle);
    ready_to_write_.erase(handle);
    ready_to_write_without_data_.erase(handle);
    FailPendingSends(*peer_ptr);

    if (was_connected) {
        event_callback_(
            DisconnectedEvent{handle, peer_ptr->socket_addr(), *peer_ptr->service_addr()});
    }

    return absl::OkStatus();
}

void
ControlChannel::HandlePeerRead(PeerHandle handle) {
    auto peer_ptr_or_status = FindPeer(handle);
    if (!peer_ptr_or_status.ok()) {
        NIXL_WARN << "Peer socket is ready to read but peer not present!";
        ready_to_read_.erase(handle);
        return;
    }
    auto *peer_ptr = *peer_ptr_or_status;

    IoResult res = peer_ptr->DoSocketRead();
    if (res == IoResult::kDisconnected) {
        if (const auto status = DisconnectInternal(handle); !status.ok()) {
            NIXL_WARN << "Failed to disconnect from a peer: " << status;
        }
        ready_to_read_.erase(handle);
        return;
    } else if (res == IoResult::kEagain) {
        ready_to_read_.erase(handle);
    }

    while (true) {
        auto msg_data_or_status = peer_ptr->TryPopFullMessage();
        if (!msg_data_or_status.ok()) {
            NIXL_ERROR << "Invalid message frame from peer: " << msg_data_or_status.status()
                       << ". Disconnecting.";
            if (const auto status = DisconnectInternal(handle); !status.ok()) {
                NIXL_WARN << "Failed to disconnect invalid peer: " << status;
            }
            return;
        }
        std::vector<uint8_t> msg_data = std::move(*msg_data_or_status);
        if (msg_data.size() == 0) {
            break;
        }

        auto ppeer_it = prospective_peers_.find(handle);
        if (ppeer_it == prospective_peers_.end()) {
            // This is an already identified peer, trigger the normal message received callback
            event_callback_(MessageReceivedEvent{
                handle, peer_ptr->socket_addr(), *peer_ptr->service_addr(), std::move(msg_data)});
            continue;
        }

        // If it's a prospective peer, this MUST be an IdentityMessage
        Message msg;
        // Skip the message size at the start of the array.
        bool is_valid = msg.ParseFromArray(msg_data.data() + kSerializedExpectedLenBytes,
                                           msg_data.size() - kSerializedExpectedLenBytes);
        if (!is_valid || !msg.has_identity()) {
            NIXL_ERROR << "First message from prospective peer is not a valid IdentityMessage. "
                          "Disconnecting.";
            if (const auto status = DisconnectInternal(handle); !status.ok()) {
                NIXL_WARN << "Failed to disconnect invalid peer: " << status;
            }
            return;
        }

        IPAddress ip_addr;
        ip_addr.set_addr(msg.identity().addr().addr());
        AgentAddress peer_service_addr(ip_addr, static_cast<uint16_t>(msg.identity().port()));
        peer_ptr->set_service_addr(peer_service_addr);

        peers_.try_emplace(handle, std::move(ppeer_it->second));
        prospective_peers_.erase(ppeer_it);

        NIXL_DEBUG << "Accepted peer: " << peer_service_addr;
        event_callback_(ConnectedEvent{handle,
                                       peer_ptr->socket_addr(),
                                       peer_service_addr,
                                       msg.identity().agent_name(),
                                       std::move(*msg.mutable_identity()->mutable_dxs())});
    }
}

void
ControlChannel::HandlePeerWrite(PeerHandle handle) {
    auto peer_ptr_or_status = FindPeer(handle);
    if (!peer_ptr_or_status.ok()) {
        NIXL_WARN << "Peer socket is ready to write but peer not present!";
        ready_to_write_.erase(handle);
        return;
    }
    auto *peer_ptr = *peer_ptr_or_status;

    // We may not actually have data to write on the first edge trigger
    if (!peer_ptr->HasMessageToSend()) {
        ready_to_write_without_data_.insert(handle);
        ready_to_write_.erase(handle);
        return;
    }

    IoResult res = peer_ptr->DoSocketWrite();
    if (res == IoResult::kDisconnected || res == IoResult::kEagain) {
        ready_to_write_.erase(handle);
        if (res == IoResult::kDisconnected) {
            if (const auto status = DisconnectInternal(handle); !status.ok()) {
                NIXL_WARN << "Failed to disconnect from a peer: " << status;
            }
        }
        return;
    }

    auto maybe_sent = peer_ptr->TryPopSentMessage();
    if (maybe_sent.has_value()) {
        auto completed_id = *maybe_sent;
        {
            absl::MutexLock status_lock(send_statuses_mutex_);
            send_statuses_[completed_id] = SendStatus::kCompleted;
        }
        // Check to see if that pop cleared our queue
        if (!peer_ptr->HasMessageToSend()) {
            // Move to waiting for data
            ready_to_write_without_data_.insert(handle);
            ready_to_write_.erase(handle);
        }
    }
}

void
ControlChannel::AcceptPeers() {
    // Loop until EAGAIN for accept
    while (true) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        auto client_fd = ManagedFileDescriptor(
            ::accept(*listen_fd_, (struct sockaddr *)&client_addr, &client_len));
        if (!client_fd.HasValidFd()) {
            if (*client_fd.fd_errno() == EAGAIN || *client_fd.fd_errno() == EWOULDBLOCK) {
                break;
            }
            if (*client_fd.fd_errno() == EINTR) {
                continue;
            }
            NIXL_ERROR << absl::ErrnoToStatus(client_fd.fd_errno().value_or(errno),
                                              "failed to accept connection");
            break;
        }

        if (const auto status =
                SetTcpKeepalive(*client_fd, heartbeat_send_period_, heartbeat_timeout_);
            !status.ok()) {
            NIXL_ERROR << "Failed to set TCP Keepalive parameters on accepted socket: " << status;
            continue;
        }

        const auto status = SetNonBlocking(*client_fd);
        if (!status.ok()) {
            NIXL_ERROR << "Failed to set socket to non-blocking mode. Dropping connection."
                       << status;
            continue;
        }

        const auto service_port = ::ntohs(client_addr.sin_port);
        std::array<char, INET_ADDRSTRLEN> ip_str;
        if (::inet_ntop(AF_INET, &client_addr.sin_addr, ip_str.data(), ip_str.size()) == nullptr) {
            NIXL_PERROR << "inet_ntop failed:";
            continue;
        }
        IPAddress ip_addr;
        ip_addr.set_addr(ip_str.data());
        AgentAddress new_peer_addr{ip_addr, service_port};

        const int new_peer_fd = *client_fd;
        const PeerHandle new_peer_handle = static_cast<PeerHandle>(new_peer_fd);

        auto [it, unused_inserted] = prospective_peers_.try_emplace(
            new_peer_handle, std::make_unique<Peer>(std::move(client_fd), new_peer_addr));
        auto &peer_ptr = it->second;

        ManagedEpollRegistration peer_epoll_reg{
            epoll_fd_, peer_ptr->fd(), EPOLLIN | EPOLLOUT | EPOLLET};
        if (peer_epoll_reg.epoll_add_errno().has_value()) {
            NIXL_ERROR << absl::ErrnoToStatus(
                *peer_epoll_reg.epoll_add_errno(),
                "failed to add accepted peer to epoll tracking structure");
            prospective_peers_.erase(it);
            continue;
        }
        peer_ptr->set_epoll_reg(std::move(peer_epoll_reg));

        NIXL_DEBUG << "Accepted prospective peer: " << new_peer_addr;
    }
}

void
ControlChannel::DrainCommands() {
    uint64_t val;
    int read_result = 0;
    while ((read_result = ::read(*command_fd_, &val, sizeof(val))) > 0) {}
    if (read_result < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
        NIXL_PWARN << "read on command fd failed! TODO: Tell upper layer that the "
                      "control channel is borked";
    }

    while (true) {
        auto cmd_or_status = command_queue_.TryDequeue();
        if (cmd_or_status.ok()) {
            (std::move(*cmd_or_status))();
        } else {
            break;
        }
    }
}

void
ControlChannel::LoopEpoll() {
    absl::MutexLock epoll_lock(epoll_mutex_);

    std::array<struct epoll_event, kMaxEventsPerPoll> events;

    while (!stop_notif_.HasBeenNotified()) {
        int timeout_ms = absl::ToInt64Milliseconds(kEventPollTimeout);
        // Don't wait on epoll if we have Peers to process
        if (!ready_to_read_.empty() || !ready_to_write_.empty()) {
            timeout_ms = 0;
        }

        const int nfds = ::epoll_wait(*epoll_fd_, events.data(), kMaxEventsPerPoll, timeout_ms);
        if (nfds == -1) {
            if (errno == EINTR) {
                continue;
            }
            NIXL_PWARN
                << "epoll_wait failed! TODO: Tell upper layer that the control channel is borked";
            continue;
        }

        for (int i = 0; i < nfds; ++i) {
            const auto raw_fd = events[i].data.fd;
            if (raw_fd == *command_fd_) {
                DrainCommands();
                continue;
            }
            if (raw_fd == *listen_fd_) {
                AcceptPeers();
                continue;
            }
            if (events[i].events & (EPOLLERR | EPOLLHUP)) {
                if (const auto status = DisconnectInternal(static_cast<PeerHandle>(raw_fd));
                    !status.ok()) {
                    NIXL_WARN << "Failed to disconnect stale peer: " << status;
                }
                continue;
            }

            PeerHandle handle = static_cast<PeerHandle>(raw_fd);

            if (events[i].events & EPOLLIN) {
                ready_to_read_.insert(handle);
            }
            if (events[i].events & EPOLLOUT) {
                ready_to_write_.insert(handle);
            }
        }

        // Process reads
        std::vector<PeerHandle> current_reads(ready_to_read_.begin(), ready_to_read_.end());
        for (auto handle : current_reads) {
            HandlePeerRead(handle);
        }

        // Process writes
        std::vector<PeerHandle> current_writes(ready_to_write_.begin(), ready_to_write_.end());
        for (auto handle : current_writes) {
            HandlePeerWrite(handle);
        }
    }

    // Drain all commands before exiting
    DrainCommands();
}

absl::StatusOr<Peer * absl_nonnull>
ControlChannel::FindPeer(PeerHandle handle) {
    auto it = peers_.find(handle);
    if (it != peers_.end()) {
        return it->second.get();
    }

    auto ppeer_it = prospective_peers_.find(handle);
    if (ppeer_it != prospective_peers_.end()) {
        return ppeer_it->second.get();
    }

    return absl::NotFoundError(absl::StrCat("Peer with handle ", handle, " not found"));
}

void
ControlChannel::FailPendingSends(Peer &peer) {
    absl::MutexLock status_lock(send_statuses_mutex_);

    for (const auto &pending : peer.send_queue()) {
        // Normal messages are failed and will eventually be reaped by our callers
        send_statuses_[pending.id] = SendStatus::kFailed;
    }
}

SendId
GenerateSendId() {
    static std::atomic<uint64_t> next_id{1};

    auto id = next_id.fetch_add(1, std::memory_order_relaxed);
    while (id == static_cast<uint64_t>(SendId::kInvalid)) {
        id = next_id.fetch_add(1, std::memory_order_relaxed);
    }

    return static_cast<SendId>(id);
}

} // namespace tcpxo
