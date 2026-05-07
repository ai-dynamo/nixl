/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_RUNTIME_NULL_RT_H
#define NIXLBENCH_RUNTIME_NULL_RT_H

#include "runtime/runtime.h"

class xferBenchNullRT : public xferBenchRT {
public:
    xferBenchNullRT();

    int
    sendInt(int *buffer, int dest_rank) override;

    int
    recvInt(int *buffer, int src_rank) override;

    int
    broadcastInt(int *buffer, size_t count, int root_rank) override;

    int
    sendChar(char *buffer, size_t count, int dest_rank) override;

    int
    recvChar(char *buffer, size_t count, int src_rank) override;

    int
    reduceSumDouble(double *local_value, double *global_value, int dest_rank) override;

    int
    barrier(const std::string &barrier_id) override;
};

#endif // NIXLBENCH_RUNTIME_NULL_RT_H
