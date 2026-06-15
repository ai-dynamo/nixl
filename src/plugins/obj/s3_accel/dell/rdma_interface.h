/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compatibility shim: interface generalized to iS3RdmaClient in s3_accel/rdma_interface.h
#ifndef NIXL_OBJ_PLUGIN_S3_DELL_RDMA_INTERFACE_H
#define NIXL_OBJ_PLUGIN_S3_DELL_RDMA_INTERFACE_H

#include "s3_accel/rdma_interface.h"

using iDellS3RdmaClient = iS3RdmaClient;

#endif // NIXL_OBJ_PLUGIN_S3_DELL_RDMA_INTERFACE_H
