/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "obj_test_base.h"
#include "mock_client_registry.h"

namespace gtest::obj {
namespace {
    mockClientRegistrar reg_dell("dell", []() { return std::make_shared<mockS3Client>(); });
} // namespace
} // namespace gtest::obj
