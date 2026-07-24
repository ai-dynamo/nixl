/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>

#include <gtest/gtest.h>

#include "ucx_utils.h"

namespace {

TEST(UcxEndpoint, ErrorCallbackMarksEndpointFailedWithoutClosingIt) {
    std::vector<std::string> devices;
    nixlUcxContext contexts[] = {
        {devices, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 1},
        {devices, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT, 1},
    };
    nixlUcxWorker consumer(contexts[0], UCP_ERR_HANDLING_MODE_PEER);
    nixlUcxWorker producer(contexts[1], UCP_ERR_HANDLING_MODE_PEER);
    std::string producerAddress = producer.epAddr();
    auto endpoint = consumer.connect(producerAddress.data(), producerAddress.size());
    ASSERT_NE(endpoint, nullptr);

    const ucp_ep_h nativeEndpoint = endpoint->getEp();
    endpoint->err_cb(nativeEndpoint, UCS_ERR_CONNECTION_RESET);

    EXPECT_EQ(endpoint->checkTxState(), NIXL_ERR_REMOTE_DISCONNECT);
    EXPECT_EQ(endpoint->getEp(), nativeEndpoint);
}

} // namespace
