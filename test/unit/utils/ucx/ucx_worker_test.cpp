/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <vector>
#include <string>
#include <cassert>
#include <cstring>
#include <iostream>

#include "ucx/ucx_utils.h"
#include "ucx/rkey.h"
//TODO: meson conditional build for CUDA
//#define USE_VRAM

#ifdef USE_VRAM

#include <cuda.h>
#include <cuda_runtime.h>

int gpu_id = 0;

static void checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
    std::cerr << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif


using namespace std;


void completeRequest(nixlUcxWorker w[2], std::string op, bool is_flush, nixl_status_t ret,  nixlUcxReq &req)
{
    assert( ret == NIXL_SUCCESS || ret == NIXL_IN_PROG);
    if (ret == NIXL_SUCCESS) {
        if (!is_flush) {
            cout << "WARNING: " << op << " request completed immediately - no testing non-inline path" << endl;
        }
    } else {
        if (!is_flush) {
            cout << "NOTE: Testing non-inline " << op << " path!" << endl;
        }

        ret = NIXL_IN_PROG;
        do {
            ret = w[0].test(req);
            w[1].progress();
        } while( ret == NIXL_IN_PROG);
        assert(ret == NIXL_SUCCESS);
        w[0].reqRelease(req);
    }
}

int main()
{
    vector<string> devs;
    // TODO: pass dev name for testing
    // in CI it would be goot to test both SHM and IB
    //devs.push_back("mlx5_0");
    nixlUcxContext c[2] = {{devs, 0, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE},
                           {devs, 0, false, 1, nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE}};

    nixlUcxWorker w[2] = {
        nixlUcxWorker(c[0]),
        nixlUcxWorker(c[1])
    };
    std::unique_ptr<nixlUcxEp> ep[2];
    nixlUcxMem mem[2];
    std::unique_ptr<nixl::ucx::rkey> rkey[2];
    nixlUcxReq req;
    uint8_t *buffer[2];
    uint8_t *chk_buffer;
    nixl_status_t ret;
    size_t buf_size = 128 * 1024 * 1024; /* Use large buffer to ensure non-inline transfer */
    size_t i;
    nixl_mem_t nixl_mem_type;

#ifdef USE_VRAM
    checkCudaError(cudaSetDevice(gpu_id), "Failed to set device");
    checkCudaError(cudaMalloc(&buffer[0], buf_size), "Failed to allocate CUDA buffer 0");
    checkCudaError(cudaMalloc(&buffer[1], buf_size), "Failed to allocate CUDA buffer 1");
    nixl_mem_type = VRAM_SEG;
#else
    buffer[0] = (uint8_t*) calloc(1, buf_size);
    buffer[1] = (uint8_t*) calloc(1, buf_size);
    nixl_mem_type = DRAM_SEG;
#endif
    chk_buffer = (uint8_t*) calloc(1, buf_size);

    assert(buffer[0]);
    assert(buffer[1]);
    /* Test control path */
    for(i = 0; i < 2; i++) {
        const std::string addr = w[i].epAddr();
        assert(!addr.empty());
        auto result = w[!i].connect((void*)addr.data(), addr.size());
        assert(result.ok());
        ep[!i] = std::move(*result);
        assert(0 == c[i].memReg(buffer[i], buf_size, mem[i], nixl_mem_type));
        std::string rkey_tmp = c[i].packRkey(mem[i]);
        assert(!rkey_tmp.empty());
        rkey[!i] = std::make_unique<nixl::ucx::rkey>(*ep[!i], rkey_tmp.data());
    }

    /* =========================================
     *   Test Write operation
     * ========================================= */

#ifdef USE_VRAM
    checkCudaError(cudaMemset(buffer[1], 0xbb, buf_size), "Failed to memset");
    checkCudaError(cudaMemset(buffer[0], 0xda, buf_size), "Failed to memset");
#else
    memset(buffer[1], 0xbb, buf_size);
    memset(buffer[0], 0xda, buf_size);
#endif

    // Write request
    ret = ep[0]->write(buffer[0], mem[0], (uint64_t)buffer[1], *rkey[0], buf_size / 2, req);
    completeRequest(w, std::string("WRITE"), false, ret, req);

    // Flush to ensure that all data is in-place
    ret = ep[0]->flushEp(req);
    completeRequest(w, std::string("WRITE"), true, ret, req);

#ifdef USE_VRAM
    checkCudaError(cudaMemcpy(chk_buffer, buffer[1], 128, cudaMemcpyDeviceToHost), "Failed to memcpy");
#else
    memcpy(chk_buffer, buffer[1], buf_size);
#endif

    for(i = 0; i < buf_size/2; i++) {
        assert(chk_buffer[i] == 0xda);
    }
    for(; i < buf_size; i++) {
        assert(chk_buffer[i] == 0xbb);
    }

    /* =========================================
     *   Test Read operation
     * ========================================= */

#ifdef USE_VRAM
    checkCudaError(cudaMemset(buffer[0], 0xbb, buf_size), "Failed to memset");
    checkCudaError(cudaMemset(buffer[1], 0xbb, buf_size/3), "Failed to memset");
    checkCudaError(cudaMemset(buffer[1] + 32, 0xda, buf_size - buf_size / 3), "Failed to memset");
#else
    memset(buffer[0], 0xbb, buf_size);
    memset(buffer[1], 0xbb, buf_size/3);
    memset(buffer[1] + buf_size/3, 0xda, buf_size - buf_size / 3);
#endif

    // Read request
    ret = ep[0]->read((uint64_t)buffer[1], *rkey[0], buffer[0], mem[0], buf_size, req);
    completeRequest(w, std::string("READ"), false, ret, req);

    // Flush to ensure that all data is in-place
    ret = ep[0]->flushEp(req);
    completeRequest(w, std::string("READ"), true, ret, req);

#ifdef USE_VRAM
    checkCudaError(cudaMemcpy(chk_buffer, buffer[0], buf_size, cudaMemcpyDeviceToHost), "Failed to memcpy");
#else
    memcpy(chk_buffer, buffer[0], buf_size);
#endif

    for(i = 0; i < buf_size/3; i++) {
        assert(chk_buffer[i] == 0xbb);
    }
    for(; i < buf_size; i++) {
        assert(chk_buffer[i] == 0xda);
    }

    /* Test shutdown */
    for(i = 0; i < 2; i++) {
        c[i].memDereg(mem[i]);
        assert(ep[i].release());
    }


#ifdef USE_VRAM
    checkCudaError(cudaFree(buffer[0]), "Failed to allocate CUDA buffer 0");
    checkCudaError(cudaFree(buffer[1]), "Failed to allocate CUDA buffer 0");
#else
    free(buffer[0]);
    free(buffer[1]);
#endif
    free(chk_buffer);
}
