/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/null_rt.h"

xferBenchNullRT::xferBenchNullRT() {
    setSize(1);
    setRank(0);
}

int
xferBenchNullRT::sendInt(int *buffer, int dest_rank) {
    (void)buffer;
    (void)dest_rank;
    return 0;
}

int
xferBenchNullRT::recvInt(int *buffer, int src_rank) {
    (void)buffer;
    (void)src_rank;
    return 0;
}

int
xferBenchNullRT::broadcastInt(int *buffer, size_t count, int root_rank) {
    (void)buffer;
    (void)count;
    (void)root_rank;
    return 0;
}

int
xferBenchNullRT::sendChar(char *buffer, size_t count, int dest_rank) {
    (void)buffer;
    (void)count;
    (void)dest_rank;
    return 0;
}

int
xferBenchNullRT::recvChar(char *buffer, size_t count, int src_rank) {
    (void)buffer;
    (void)count;
    (void)src_rank;
    return 0;
}

int
xferBenchNullRT::reduceSumDouble(double *local_value, double *global_value, int dest_rank) {
    (void)dest_rank;
    *global_value = *local_value;
    return 0;
}

int
xferBenchNullRT::barrier(const std::string &barrier_id) {
    (void)barrier_id;
    return 0;
}
