#include <vector>
#include <string>
#include <cassert>
#include <cstring>
#include <iostream>

#include "ucx_utils.h"

//TODO: meson conditional build for CUDA
//#define USE_VRAM

#ifdef USE_VRAM

#include <cuda_runtime.h>
#include <cufile.h>

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


typedef struct requestData_s {
    int initialized;
} requestData;

static void nixlUcxRequestInit(void *request)
{
    requestData *req = (requestData *)request;

    req->initialized = 1;
}

void completeRequest(nixlUcxWorker w[2], std::string op, bool is_flush, nixl_xfer_state_t ret,  nixlUcxReq &req)
{
    assert( ret == NIXL_XFER_DONE || ret == NIXL_XFER_PROC);
    if (ret == NIXL_XFER_DONE) {
        if (!is_flush) {
            cout << "WARNING: " << op << " request completed immmediately - no testing non-inline path" << endl;
        }
    } else {
        if (!is_flush) {
            cout << "NOTE: Testing non-inline " << op << " path!" << endl;
        }
        assert( ((requestData *)req)->initialized == 1);

        ret = NIXL_XFER_PROC;
        do {
            ret = w[0].test(req);
            w[1].progress();
        } while( ret == NIXL_XFER_PROC);
        assert(ret == NIXL_XFER_DONE);
        w[0].reqRelease(req);
    }
}

int main()
{
    vector<string> devs;
    // TODO: pass dev name for testing
    // in CI it would be goot to test both SHM and IB
    //devs.push_back("mlx5_0");
    nixlUcxContext c[2] = {
        nixlUcxContext(devs, sizeof(requestData), nixlUcxRequestInit, NULL, NIXL_UCX_MT_SINGLE),
        nixlUcxContext(devs, sizeof(requestData), nixlUcxRequestInit, NULL, NIXL_UCX_MT_SINGLE)
    };

    nixlUcxWorker w[2] = {
        nixlUcxWorker(&c[0]),
        nixlUcxWorker(&c[1])
    };
    nixlUcxEp ep[2];
    nixlUcxMem mem[2];
    nixlUcxRkey rkey[2];
    nixlUcxReq req;
    uint8_t *buffer[2];
    uint8_t *chk_buffer;
    nixl_xfer_state_t ret;
    size_t buf_size = 128 * 1024 * 1024; /* Use large buffer to ensure non-inline transfer */
    size_t i;

#ifdef USE_VRAM
    checkCudaError(cudaSetDevice(gpu_id), "Failed to set device");
    checkCudaError(cudaMalloc(&buffer[0], buf_size), "Failed to allocate CUDA buffer 0");
    checkCudaError(cudaMalloc(&buffer[1], buf_size), "Failed to allocate CUDA buffer 1");
#else
    buffer[0] = (uint8_t*) calloc(1, buf_size);
    buffer[1] = (uint8_t*) calloc(1, buf_size);
#endif
    chk_buffer = (uint8_t*) calloc(1, buf_size);

    assert(buffer[0]);
    assert(buffer[1]);
    /* Test control path */
    for(i = 0; i < 2; i++) {
        uint64_t addr;
        size_t size;
        assert(0 == w[i].epAddr(addr, size));
        assert(0 == w[!i].connect((void*) addr, size, ep[!i]));
        free((void*) addr);
        assert(0 == w[i].memReg(buffer[i], buf_size, mem[i]));
        assert(0 == w[i].packRkey(mem[i], addr, size));
        assert(0 == w[!i].rkeyImport(ep[!i], (void*) addr, size, rkey[!i]));
        free((void*) addr);
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
    ret = w[0].write(ep[0], buffer[0], mem[0], (uint64_t) buffer[1], rkey[0], buf_size/2, req);
    completeRequest(w, std::string("WRITE"), false, ret, req);

    // Flush to ensure that all data is in-place
    ret = w[0].flushEp(ep[0], req);
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
    ret = w[0].read(ep[0], (uint64_t) buffer[1], rkey[0], buffer[0], mem[0], buf_size, req);
    completeRequest(w, std::string("READ"), false, ret, req);

    // Flush to ensure that all data is in-place
    ret = w[0].flushEp(ep[0], req);
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
        w[i].rkeyDestroy(rkey[i]);
        w[i].memDereg(mem[i]);
        assert(0 == w[i].disconnect(ep[i]));
    }


#ifdef USE_VRAM
    checkCudaError(cudaFree(buffer[0]), "Failed to allocate CUDA buffer 0");
    checkCudaError(cudaFree(buffer[1]), "Failed to allocate CUDA buffer 0");
#else
    free(buffer[0]);
    free(buffer[1]);
#endif

}
