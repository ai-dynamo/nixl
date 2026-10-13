/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_backend_aux.h"

#include <gtest/gtest.h>

#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

TEST(GpunetioOobSocket, InstallsReceiveAndSendTimeouts) {
    int sockets[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
    const int result = setOobSocketTimeouts(sockets[0]);
    timeval receive{};
    timeval send{};
    socklen_t receive_size = sizeof(receive);
    socklen_t send_size = sizeof(send);
    const int receive_result =
        getsockopt(sockets[0], SOL_SOCKET, SO_RCVTIMEO, &receive, &receive_size);
    const int send_result = getsockopt(sockets[0], SOL_SOCKET, SO_SNDTIMEO, &send, &send_size);
    close(sockets[0]);
    close(sockets[1]);
    ASSERT_EQ(result, 0);
    ASSERT_EQ(receive_result, 0);
    ASSERT_EQ(send_result, 0);
    EXPECT_EQ(receive.tv_sec, DOCA_OOB_SOCKET_TIMEOUT_SEC);
    EXPECT_EQ(receive.tv_usec, 0);
    EXPECT_EQ(send.tv_sec, DOCA_OOB_SOCKET_TIMEOUT_SEC);
    EXPECT_EQ(send.tv_usec, 0);
}

TEST(GpunetioOobSocket, RejectsInvalidSocket) {
    EXPECT_NE(setOobSocketTimeouts(-1), 0);
}
