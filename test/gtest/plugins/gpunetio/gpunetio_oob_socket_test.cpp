/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_backend_aux.h"

#include <iostream>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

int
main() {
    int sockets[2] = {-1, -1};
    auto fail = [&](const char *message) {
        std::cerr << message << std::endl;
        if (sockets[0] >= 0) {
            close(sockets[0]);
        }
        if (sockets[1] >= 0) {
            close(sockets[1]);
        }
        return 1;
    };
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
        return fail("socketpair failed");
    }
    if (setOobSocketTimeouts(sockets[0]) != 0) {
        return fail("setting socket timeouts failed");
    }

    timeval recv_timeout{};
    timeval send_timeout{};
    socklen_t recv_size = sizeof(recv_timeout);
    socklen_t send_size = sizeof(send_timeout);
    if (getsockopt(sockets[0], SOL_SOCKET, SO_RCVTIMEO, &recv_timeout, &recv_size) != 0) {
        return fail("reading receive timeout failed");
    }
    if (getsockopt(sockets[0], SOL_SOCKET, SO_SNDTIMEO, &send_timeout, &send_size) != 0) {
        return fail("reading send timeout failed");
    }
    if (recv_timeout.tv_sec != DOCA_OOB_SOCKET_TIMEOUT_SEC || recv_timeout.tv_usec != 0) {
        return fail("receive timeout mismatch");
    }
    if (send_timeout.tv_sec != DOCA_OOB_SOCKET_TIMEOUT_SEC || send_timeout.tv_usec != 0) {
        return fail("send timeout mismatch");
    }

    close(sockets[0]);
    close(sockets[1]);
    return 0;
}
