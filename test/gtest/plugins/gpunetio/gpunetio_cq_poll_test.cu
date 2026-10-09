/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <infiniband/mlx5dv.h>
#include "gpunetio_completion.cuh"

#include <cerrno>
#include <cstdio>
#include <cstring>

__global__ void
pollOnce(doca_gpu_dev_verbs_cq *cq, bool receive, int *status) {
    *status = receive ?
        nixl_gpunetio_dev_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                         DOCA_GPUNETIO_VERBS_QP_RQ>(cq, 0) :
        nixl_gpunetio_dev_poll_one_cq_at(cq, 0);
}

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        const auto cuda_error = (call);                                              \
        if (cuda_error != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(cuda_error)); \
            return 1;                                                                \
        }                                                                            \
    } while (0)

int
main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        return 77;
    }
    doca_gpu_dev_verbs_cq *cq;
    doca_gpu_dev_verbs_cqe64 *ring;
    int *status;
    CUDA_CHECK(cudaMallocManaged(&cq, sizeof(*cq)));
    CUDA_CHECK(cudaMallocManaged(&ring, 8 * sizeof(*ring)));
    CUDA_CHECK(cudaMallocManaged(&status, sizeof(*status)));

    struct Case {
        const char *name;
        uint64_t reserved;
        uint64_t consumed;
        int opcode;
        bool receive;
        int expected;
    };

    const Case cases[] = {
        {"SQ ready", 0, 0, MLX5_CQE_REQ, false, 0},
        {"pending", 0, 0, MLX5_CQE_INVALID, false, EBUSY},
        {"SQ reserved offset", 4, 0, MLX5_CQE_REQ, false, 0},
        {"SQ owner wrap", 8, 0, MLX5_CQE_REQ, false, 0},
        {"SQ already consumed", 0, 1, MLX5_CQE_INVALID, false, 0},
        {"SQ error", 0, 0, MLX5_CQE_REQ_ERR, false, -EIO},
        {"RQ ready", 0, 0, MLX5_CQE_RESP_SEND, true, 0},
        {"RQ reserved offset", 4, 0, MLX5_CQE_RESP_SEND, true, 0},
        {"RQ owner wrap", 8, 0, MLX5_CQE_RESP_SEND, true, 0},
        {"RQ error", 0, 0, MLX5_CQE_RESP_ERR, true, -EIO},
    };
    int failures = 0;
    for (const auto &test : cases) {
        std::memset(cq, 0, sizeof(*cq));
        std::memset(ring, 0, 8 * sizeof(*ring));
        for (int i = 0; i < 8; ++i) {
            ring[i].op_own = (MLX5_CQE_INVALID << 4) | 1;
        }
        cq->cqe_daddr = reinterpret_cast<uint8_t *>(ring);
        cq->cqe_num = 8;
        cq->cqe_rsvd = test.reserved;
        cq->cqe_ci = test.consumed;
        if (test.opcode != MLX5_CQE_INVALID) {
            ring[test.reserved & 7].op_own = (test.opcode << 4) | !!(test.reserved & 8);
        }
        pollOnce<<<1, 1>>>(cq, test.receive, status);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("%s: got=%d expected=%d\n", test.name, *status, test.expected);
        failures += *status != test.expected;
    }
    CUDA_CHECK(cudaFree(status));
    CUDA_CHECK(cudaFree(ring));
    CUDA_CHECK(cudaFree(cq));
    return failures == 0 ? 0 : 1;
}
