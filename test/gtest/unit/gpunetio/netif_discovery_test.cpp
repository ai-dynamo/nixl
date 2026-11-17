/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpunetio_backend_aux.h"

#include <gtest/gtest.h>

#include <net/if.h>

TEST(GpunetioNetifDiscovery, MissingInterfaceIsNotSuccess) {
    const char *missing = "nixl-interface-that-cannot-exist";
    ASSERT_EQ(if_nametoindex(missing), 0u);
    sockaddr address{};
    sockaddr netmask{};
    EXPECT_NE(netif_get_addr(missing, AF_INET, &address, &netmask), 0);
}

TEST(GpunetioNetifDiscovery, FindsLoopbackIpv4) {
    sockaddr address{};
    sockaddr netmask{};
    ASSERT_EQ(netif_get_addr("lo", AF_INET, &address, &netmask), 0);
    EXPECT_EQ(address.sa_family, AF_INET);
}
