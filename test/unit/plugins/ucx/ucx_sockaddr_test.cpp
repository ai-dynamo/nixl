/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Two-process test of the UCX backend running with connection_mode=sockaddr,
 * i.e. connections established through the UCP client/server API (and hence
 * through a UCX connection manager such as RDMA CM).
 *
 * A plain TCP side channel is used to exchange NIXL connection info and memory
 * metadata; it plays the role the NIXL agent metadata exchange plays in a real
 * deployment and is unrelated to the UCX data path.
 *
 * Usage:
 *   ucx_sockaddr_test target <local_ip> <local_ucx_port> <ctrl_port>
 *   ucx_sockaddr_test initiator <local_ip> <local_ucx_port> <peer_ip> <ctrl_port>
 *
 * The target must be started first. Both processes exercise:
 *   listener creation, sockaddr connection establishment, memory registration,
 *   rkey exchange, WRITE + notification, READ, data validation, disconnect and
 *   reconnect.
 */

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <unistd.h>

#include "ucx_backend.h"
#include "serdes/serdes.h"
#include "test_utils.h"

namespace {

constexpr size_t buffer_len = 1 * 1024 * 1024;
constexpr int dev_id = 0;
constexpr const char *target_agent = "TargetAgent";
constexpr const char *initiator_agent = "InitiatorAgent";
constexpr const char *notif_msg = "sockaddr-test-notif";

/* ============================ TCP side channel ============================ */

class ctrlChannel {
public:
    /* Target side: bind and accept one connection. */
    static ctrlChannel
    accept(uint16_t port) {
        const int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        nixl_exit_on_failure(listen_fd >= 0, "ctrl: socket() failed");

        const int reuse = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);

        nixl_exit_on_failure(bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0,
                             "ctrl: bind() failed");
        nixl_exit_on_failure(listen(listen_fd, 1) == 0, "ctrl: listen() failed");

        std::cout << "ctrl: waiting for peer on port " << port << std::endl;
        const int fd = ::accept(listen_fd, nullptr, nullptr);
        nixl_exit_on_failure(fd >= 0, "ctrl: accept() failed");
        close(listen_fd);

        return ctrlChannel(fd);
    }

    /* Initiator side: connect, retrying while the target comes up. */
    static ctrlChannel
    connect(const std::string &ip, uint16_t port) {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        nixl_exit_on_failure(inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) == 1,
                             "ctrl: invalid peer IP");

        for (int attempt = 0; attempt < 100; ++attempt) {
            const int fd = socket(AF_INET, SOCK_STREAM, 0);
            nixl_exit_on_failure(fd >= 0, "ctrl: socket() failed");

            if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
                return ctrlChannel(fd);
            }
            close(fd);
            usleep(100000);
        }

        nixl_exit_on_failure(false, "ctrl: failed to connect to peer");
        return ctrlChannel(-1);
    }

    ~ctrlChannel() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    ctrlChannel(ctrlChannel &&other) noexcept : fd_(other.fd_) {
        other.fd_ = -1;
    }
    ctrlChannel(const ctrlChannel &) = delete;

    void
    send(const std::string &msg) const {
        const uint64_t len = msg.size();
        sendAll(&len, sizeof(len));
        sendAll(msg.data(), msg.size());
    }

    /* Receives while progressing the UCX worker: the peer may be waiting for
     * us to accept its connection request, which only happens from within
     * ucp_worker_progress(). */
    [[nodiscard]] std::string
    recv(nixlUcxEngine &engine) const {
        waitReadable(engine);
        uint64_t len = 0;
        recvAll(&len, sizeof(len));

        std::string msg(len, '\0');
        waitReadable(engine);
        recvAll(msg.data(), len);
        return msg;
    }

    /* Rendezvous with the peer while keeping the UCX worker progressing, so
     * that incoming connection requests are accepted while we wait. */
    void
    barrier(nixlUcxEngine &engine) const {
        const char token = 'B';
        sendAll(&token, 1);

        waitReadable(engine);
        char peer_token = 0;
        recvAll(&peer_token, 1);
        nixl_exit_on_failure(peer_token == 'B', "ctrl: unexpected barrier token");
    }

    /* Spins on worker progress until the control socket has data to read. */
    void
    waitReadable(nixlUcxEngine &engine) const {
        while (true) {
            engine.progress();

            pollfd pfd{fd_, POLLIN, 0};
            if (poll(&pfd, 1, 1) > 0) {
                return;
            }
        }
    }

private:
    explicit ctrlChannel(int fd) : fd_(fd) {
        const int nodelay = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    }

    void
    sendAll(const void *buf, size_t len) const {
        size_t sent = 0;
        while (sent < len) {
            const ssize_t rc = ::send(fd_, static_cast<const char *>(buf) + sent, len - sent, 0);
            nixl_exit_on_failure(rc > 0, "ctrl: send() failed");
            sent += size_t(rc);
        }
    }

    void
    recvAll(void *buf, size_t len) const {
        size_t received = 0;
        while (received < len) {
            const ssize_t rc = ::recv(fd_, static_cast<char *>(buf) + received, len - received, 0);
            nixl_exit_on_failure(rc > 0, "ctrl: recv() failed");
            received += size_t(rc);
        }
    }

    int fd_;
};

/* ============================== NIXL helpers ============================== */

std::unique_ptr<nixlUcxEngine>
createEngine(const std::string &name, const std::string &listen_ip, uint16_t listen_port) {
    nixl_b_params_t custom_params;
    custom_params[std::string(nixl_ucx_conn_mode_param_name)] = "sockaddr";
    custom_params[std::string(nixl_ucx_listen_address_param_name)] = listen_ip;
    custom_params[std::string(nixl_ucx_listen_port_param_name)] = std::to_string(listen_port);

    nixlBackendInitParams init_params;
    init_params.localAgent = name;
    init_params.enableProgTh = false;
    init_params.pthrDelay = 100;
    init_params.customParams = &custom_params;
    init_params.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE;
    init_params.type = "UCX";

    try {
        auto engine = nixlUcxEngine::create(init_params);
        nixl_exit_on_failure(engine && !engine->getInitErr(), "Failed to initialize UCX engine",
                             name);
        return engine;
    }
    catch (const std::exception &e) {
        nixl_exit_on_failure(false, std::string("Failed to initialize UCX engine: ") + e.what(),
                             name);
        throw;
    }
}

/* Serialized description of a registered buffer: addr, len, devId, rkey. */
std::string
packMemInfo(void *addr, size_t len, const std::string &rkey) {
    nixlSerDes ser_des;
    ser_des.addBuf("addr", &addr, sizeof(addr));
    ser_des.addBuf("len", &len, sizeof(len));
    ser_des.addStr("rkey", rkey);
    return ser_des.exportStr();
}

nixlBlobDesc
unpackMemInfo(const std::string &blob) {
    nixlSerDes ser_des;
    nixl_exit_on_failure(ser_des.importStr(blob) == NIXL_SUCCESS, "Failed to import mem info");

    void *addr = nullptr;
    size_t len = 0;
    nixl_exit_on_failure(ser_des.getBuf("addr", &addr, sizeof(addr)) == NIXL_SUCCESS,
                         "Failed to get remote addr");
    nixl_exit_on_failure(ser_des.getBuf("len", &len, sizeof(len)) == NIXL_SUCCESS,
                         "Failed to get remote len");

    nixlBlobDesc desc;
    desc.addr = uintptr_t(addr);
    desc.len = len;
    desc.devId = dev_id;
    desc.metaInfo = ser_des.getStr("rkey");
    nixl_exit_on_failure(!desc.metaInfo.empty(), "Failed to get remote rkey");
    return desc;
}

void
fillPattern(char *buf, size_t len, char seed) {
    for (size_t i = 0; i < len; ++i) {
        buf[i] = char(seed + char(i % 251));
    }
}

bool
checkPattern(const char *buf, size_t len, char seed) {
    for (size_t i = 0; i < len; ++i) {
        if (buf[i] != char(seed + char(i % 251))) {
            std::cerr << "Data mismatch at offset " << i << ": got " << int(buf[i]) << " expected "
                      << int(char(seed + char(i % 251))) << std::endl;
            return false;
        }
    }
    return true;
}

/* Runs one transfer and waits for its completion, progressing the worker. */
void
runTransfer(nixlUcxEngine &engine,
            nixl_xfer_op_t op,
            nixl_meta_dlist_t &local,
            nixl_meta_dlist_t &remote,
            const std::string &remote_agent,
            bool with_notif) {
    nixl_opt_b_args_t opt_args;
    opt_args.hasNotif = with_notif;
    opt_args.notifMsg = notif_msg;

    nixlBackendReqH *handle = nullptr;
    nixl_status_t status = engine.prepXfer(op, local, remote, remote_agent, handle, &opt_args);
    nixl_exit_on_failure(status == NIXL_SUCCESS, "prepXfer failed");

    status = engine.postXfer(op, local, remote, remote_agent, handle, &opt_args);
    nixl_exit_on_failure(status == NIXL_SUCCESS || status == NIXL_IN_PROG, "postXfer failed");

    while (status == NIXL_IN_PROG) {
        engine.progress();
        status = engine.checkXfer(handle);
    }
    nixl_exit_on_failure(status == NIXL_SUCCESS, "Transfer failed");

    engine.releaseReqH(handle);
}

void
waitForNotif(nixlUcxEngine &engine, const std::string &expected_agent) {
    notif_list_t notifs;

    while (notifs.empty()) {
        engine.progress();
        const nixl_status_t status = engine.getNotifs(notifs);
        nixl_exit_on_failure(status == NIXL_SUCCESS, "getNotifs failed");
    }

    nixl_exit_on_failure(notifs.size() == 1, "Unexpected number of notifications");
    nixl_exit_on_failure(notifs[0].first == expected_agent, "Notification from unexpected agent");
    nixl_exit_on_failure(notifs[0].second == notif_msg, "Unexpected notification message");
    std::cout << "PASS: notification received from " << notifs[0].first << std::endl;
}

/* ================================== Roles ================================= */

struct peerState {
    nixlBackendMD *local_md = nullptr;
    nixlBackendMD *remote_md = nullptr;
    void *local_buf = nullptr;
    nixlBlobDesc remote_desc;
};

/* Connects to the peer: exchange connection info and register memory. */
peerState
setupPeer(nixlUcxEngine &engine,
          const ctrlChannel &ctrl,
          const std::string &peer_name,
          char pattern_seed) {
    peerState state;

    std::string local_conn_info;
    nixl_exit_on_failure(engine.getConnInfo(local_conn_info) == NIXL_SUCCESS,
                         "getConnInfo failed");
    std::cout << "Local connection info: " << local_conn_info << std::endl;

    ctrl.send(local_conn_info);
    const std::string remote_conn_info = ctrl.recv(engine);
    std::cout << "Remote connection info: " << remote_conn_info << std::endl;

    nixl_exit_on_failure(engine.loadRemoteConnInfo(peer_name, remote_conn_info) == NIXL_SUCCESS,
                         "loadRemoteConnInfo failed");
    std::cout << "PASS: sockaddr connection established towards " << peer_name << std::endl;

    /* Register a local buffer and publish its rkey. */
    state.local_buf = calloc(1, buffer_len);
    nixl_exit_on_failure(state.local_buf != nullptr, "Failed to allocate buffer");
    fillPattern(static_cast<char *>(state.local_buf), buffer_len, pattern_seed);

    nixlBlobDesc local_desc;
    local_desc.addr = uintptr_t(state.local_buf);
    local_desc.len = buffer_len;
    local_desc.devId = dev_id;
    nixl_exit_on_failure(engine.registerMem(local_desc, DRAM_SEG, state.local_md) == NIXL_SUCCESS,
                         "registerMem failed");

    std::string rkey;
    nixl_exit_on_failure(engine.getPublicData(state.local_md, rkey) == NIXL_SUCCESS,
                         "getPublicData failed");
    nixl_exit_on_failure(!rkey.empty(), "Empty rkey");

    ctrl.send(packMemInfo(state.local_buf, buffer_len, rkey));
    state.remote_desc = unpackMemInfo(ctrl.recv(engine));

    /* The peer's endpoint towards us is only accepted while we progress. */
    ctrl.barrier(engine);

    nixl_exit_on_failure(engine.loadRemoteMD(state.remote_desc, DRAM_SEG, peer_name,
                                             state.remote_md) == NIXL_SUCCESS,
                         "loadRemoteMD failed");
    std::cout << "PASS: rkey exchanged with " << peer_name << std::endl;

    return state;
}

void
teardownPeer(nixlUcxEngine &engine, peerState &state, const std::string &peer_name) {
    if (state.remote_md != nullptr) {
        engine.unloadMD(state.remote_md);
        state.remote_md = nullptr;
    }
    if (state.local_md != nullptr) {
        engine.deregisterMem(state.local_md);
        state.local_md = nullptr;
    }
    free(state.local_buf);
    state.local_buf = nullptr;
}

void
buildDescs(nixl_meta_dlist_t &local,
           nixl_meta_dlist_t &remote,
           void *local_addr,
           const nixlBlobDesc &remote_desc,
           nixlBackendMD *local_md,
           nixlBackendMD *remote_md) {
    nixlMetaDesc l;
    l.addr = uintptr_t(local_addr);
    l.len = buffer_len;
    l.devId = dev_id;
    l.metadataP = local_md;
    local.addDesc(l);

    nixlMetaDesc r;
    r.addr = remote_desc.addr;
    r.len = buffer_len;
    r.devId = dev_id;
    r.metadataP = remote_md;
    remote.addDesc(r);
}

int
runInitiator(const std::string &local_ip,
             uint16_t local_port,
             const std::string &peer_ip,
             uint16_t ctrl_port) {
    auto engine = createEngine(initiator_agent, local_ip, local_port);
    const ctrlChannel ctrl = ctrlChannel::connect(peer_ip, ctrl_port);

    for (int round = 0; round < 2; ++round) {
        std::cout << "=== Initiator round " << round << " ===" << std::endl;

        peerState state = setupPeer(*engine, ctrl, target_agent, char('A' + round));

        nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
        buildDescs(local, remote, state.local_buf, state.remote_desc, state.local_md,
                   state.remote_md);

        /* WRITE the local pattern into the target buffer, with notification. */
        runTransfer(*engine, NIXL_WRITE, local, remote, target_agent, true);
        std::cout << "PASS: WRITE completed" << std::endl;

        ctrl.barrier(*engine);

        /* READ the target buffer back; the target has overwritten it with its
         * own pattern, which we validate. */
        runTransfer(*engine, NIXL_READ, local, remote, target_agent, false);
        nixl_exit_on_failure(
            checkPattern(static_cast<char *>(state.local_buf), buffer_len, char('T' + round)),
            "READ data validation failed");
        std::cout << "PASS: READ data matches" << std::endl;

        ctrl.barrier(*engine);

        teardownPeer(*engine, state, target_agent);
        nixl_exit_on_failure(engine->disconnect(target_agent) == NIXL_SUCCESS, "disconnect failed");
        std::cout << "PASS: disconnected" << std::endl;

        ctrl.barrier(*engine);
    }

    std::cout << "ALL TESTS PASSED (initiator)" << std::endl;
    return 0;
}

int
runTarget(const std::string &local_ip, uint16_t local_port, uint16_t ctrl_port) {
    auto engine = createEngine(target_agent, local_ip, local_port);
    const ctrlChannel ctrl = ctrlChannel::accept(ctrl_port);

    for (int round = 0; round < 2; ++round) {
        std::cout << "=== Target round " << round << " ===" << std::endl;

        peerState state = setupPeer(*engine, ctrl, initiator_agent, char('X' + round));

        /* The initiator writes its pattern into our buffer and notifies. */
        waitForNotif(*engine, initiator_agent);
        nixl_exit_on_failure(
            checkPattern(static_cast<char *>(state.local_buf), buffer_len, char('A' + round)),
            "WRITE data validation failed");
        std::cout << "PASS: WRITE data matches" << std::endl;

        /* Refill with a pattern the initiator will READ back. */
        fillPattern(static_cast<char *>(state.local_buf), buffer_len, char('T' + round));

        ctrl.barrier(*engine);
        ctrl.barrier(*engine);

        teardownPeer(*engine, state, initiator_agent);
        nixl_exit_on_failure(engine->disconnect(initiator_agent) == NIXL_SUCCESS,
                             "disconnect failed");
        std::cout << "PASS: disconnected" << std::endl;

        ctrl.barrier(*engine);
    }

    std::cout << "ALL TESTS PASSED (target)" << std::endl;
    return 0;
}

void
usage(const char *prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " target <local_ip> <local_ucx_port> <ctrl_port>\n"
              << "  " << prog << " initiator <local_ip> <local_ucx_port> <peer_ip> <ctrl_port>\n";
}

} // namespace

int
main(int argc, char *argv[]) {
    if (argc < 5) {
        usage(argv[0]);
        return 1;
    }

    const std::string role = argv[1];
    const std::string local_ip = argv[2];
    const auto local_port = uint16_t(std::stoi(argv[3]));

    if (role == "target") {
        return runTarget(local_ip, local_port, uint16_t(std::stoi(argv[4])));
    }

    if (role == "initiator") {
        if (argc < 6) {
            usage(argv[0]);
            return 1;
        }
        return runInitiator(local_ip, local_port, argv[4], uint16_t(std::stoi(argv[5])));
    }

    usage(argv[0]);
    return 1;
}
