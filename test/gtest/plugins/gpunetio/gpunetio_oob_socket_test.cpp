/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_backend_aux.h"

#include <cassert>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

int
main() {
    int sockets[2] = {-1, -1};
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    assert(setOobSocketTimeouts(sockets[0]) == 0);

    timeval recv_timeout{};
    timeval send_timeout{};
    socklen_t recv_size = sizeof(recv_timeout);
    socklen_t send_size = sizeof(send_timeout);
    assert(getsockopt(sockets[0], SOL_SOCKET, SO_RCVTIMEO, &recv_timeout, &recv_size) == 0);
    assert(getsockopt(sockets[0], SOL_SOCKET, SO_SNDTIMEO, &send_timeout, &send_size) == 0);
    assert(recv_timeout.tv_sec == DOCA_OOB_SOCKET_TIMEOUT_SEC);
    assert(recv_timeout.tv_usec == 0);
    assert(send_timeout.tv_sec == DOCA_OOB_SOCKET_TIMEOUT_SEC);
    assert(send_timeout.tv_usec == 0);

    close(sockets[0]);
    close(sockets[1]);
    return 0;
}
