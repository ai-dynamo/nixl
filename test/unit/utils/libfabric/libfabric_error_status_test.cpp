/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "libfabric/libfabric_common.h"

#include <cassert>

int
main() {
    assert(LibfabricUtils::cqErrorToNixlStatus(FI_ENOTCONN) == NIXL_ERR_REMOTE_DISCONNECT);
    assert(LibfabricUtils::cqErrorToNixlStatus(FI_EIO) == NIXL_ERR_BACKEND);
    return 0;
}
