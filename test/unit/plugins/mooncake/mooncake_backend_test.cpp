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
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>

#include "mooncake_backend.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda.h>

int gpu_id = 0;

static void
checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << cudaGetErrorString(result)
                  << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

using namespace std;

class testHndlIterator {
private:
    bool reuse;
    bool set;
    bool prepare;
    bool release;
    nixlBackendReqH *handle;

public:
    testHndlIterator(bool _reuse) {
        reuse = _reuse;
        if (reuse) {
            prepare = true;
            release = false;
        } else {
            prepare = true;
            release = true;
        }
        handle = nullptr;
        set = false;
    }

    ~testHndlIterator() {
        /* Make sure that handler was released */
        assert(!set);
    }

    bool
    needPrep() {
        if (reuse) {
            if (!prepare) {
                return false;
            }
        }
        return true;
    }

    bool
    needRelease() {
        return release;
    }

    void
    isLast() {
        if (reuse) {
            release = true;
        }
    }

    void
    setHandle(nixlBackendReqH *_handle) {
        assert(!set);
        handle = _handle;
        set = true;
        if (reuse) {
            prepare = false;
        }
    }

    void
    unsetHandle() {
        assert(set);
        set = false;
    }

    nixlBackendReqH *&
    getHandle() {
        assert(set);
        return handle;
    }
};

nixlBackendEngine *
createEngine(std::string name, bool p_thread) {
    nixlBackendEngine *mooncake;
    nixlBackendInitParams init;
    nixl_b_params_t custom_params;

    init.enableProgTh = p_thread;
    init.pthrDelay = 100;
    init.localAgent = name;
    init.customParams = &custom_params;
    init.type = "Mooncake";

    mooncake = (nixlBackendEngine *)new nixlMooncakeEngine(&init);
    assert(!mooncake->getInitErr());
    if (mooncake->getInitErr()) {
        std::cout << "Failed to initialize worker1" << std::endl;
        exit(1);
    }

    return mooncake;
}

void
releaseEngine(nixlBackendEngine *mooncake) {
    delete mooncake;
}

std::string
memType2Str(nixl_mem_t mem_type) {
    switch (mem_type) {
    case DRAM_SEG:
        return std::string("DRAM");
    case VRAM_SEG:
        return std::string("VRAM");
    case BLK_SEG:
        return std::string("BLOCK");
    case FILE_SEG:
        return std::string("FILE");
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}


#ifdef HAVE_CUDA

static int
cudaQueryAddr(void *address, bool &is_dev, CUdevice &dev, CUcontext &ctx) {
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    uint32_t is_managed = 0;
#define NUM_ATTRS 4
    CUpointer_attribute attr_type[NUM_ATTRS];
    void *attr_data[NUM_ATTRS];
    CUresult result;

    attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    attr_data[0] = &mem_type;
    attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
    attr_data[1] = &is_managed;
    attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;

    attr_data[2] = &dev;
    attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[3] = &ctx;

    result = cuPointerGetAttributes(4, attr_type, attr_data, (CUdeviceptr)address);

    is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

    return (CUDA_SUCCESS != result);
}

#endif


void
allocateBuffer(nixl_mem_t mem_type, int dev_id, size_t len, void *&addr) {
    switch (mem_type) {
    case DRAM_SEG:
        addr = calloc(1, len);
        break;
#ifdef HAVE_CUDA
    case VRAM_SEG: {
        bool is_dev;
        CUdevice dev;
        CUcontext ctx;

        checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
        checkCudaError(cudaMalloc(&addr, len), "Failed to allocate CUDA buffer 0");
        cudaQueryAddr(addr, is_dev, dev, ctx);
        std::cout << "CUDA addr: " << std::hex << addr << " dev=" << std::dec << dev
                  << " ctx=" << std::hex << ctx << std::dec << std::endl;
        break;
    }
#endif
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
    assert(addr);
}

void
releaseBuffer(nixl_mem_t mem_type, int dev_id, void *&addr) {
    switch (mem_type) {
    case DRAM_SEG:
        free(addr);
        break;
#ifdef HAVE_CUDA
    case VRAM_SEG:
        checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
        checkCudaError(cudaFree(addr), "Failed to allocate CUDA buffer 0");
        break;
#endif
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void
doMemset(nixl_mem_t mem_type, int dev_id, void *addr, char byte, size_t len) {
    switch (mem_type) {
    case DRAM_SEG:
        memset(addr, byte, len);
        break;
#ifdef HAVE_CUDA
    case VRAM_SEG:
        checkCudaError(cudaSetDevice(dev_id), "Failed to set device");
        checkCudaError(cudaMemset(addr, byte, len), "Failed to memset");
        break;
#endif
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void *
getValidationPtr(nixl_mem_t mem_type, void *addr, size_t len) {
    switch (mem_type) {
    case DRAM_SEG:
        return addr;
        break;
#ifdef HAVE_CUDA
    case VRAM_SEG: {
        void *ptr = calloc(len, 1);
        checkCudaError(cudaMemcpy(ptr, addr, len, cudaMemcpyDeviceToHost), "Failed to memcpy");
        return ptr;
    }
#endif
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void *
releaseValidationPtr(nixl_mem_t mem_type, void *addr) {
    switch (mem_type) {
    case DRAM_SEG:
        break;
#ifdef HAVE_CUDA
    case VRAM_SEG:
        free(addr);
        break;
#endif
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
    return NULL;
}

void
allocateWrongGPUTest(nixlBackendEngine *mooncake, int dev_id) {
    nixlBlobDesc desc;
    nixlBackendMD *md;
    void *buf;

    allocateBuffer(VRAM_SEG, dev_id, desc.len, buf);

    desc.devId = dev_id;
    desc.addr = (uint64_t)buf;

    int ret = mooncake->registerMem(desc, VRAM_SEG, md);

    assert(ret == NIXL_ERR_NOT_SUPPORTED);

    releaseBuffer(VRAM_SEG, dev_id, buf);
}

void
allocateAndRegister(nixlBackendEngine *mooncake,
                    int dev_id,
                    nixl_mem_t mem_type,
                    void *&addr,
                    size_t len,
                    nixlBackendMD *&md) {
    nixlBlobDesc desc;

    allocateBuffer(mem_type, dev_id, len, addr);

    desc.addr = (uintptr_t)addr;
    desc.len = len;
    desc.devId = dev_id;

    int ret = mooncake->registerMem(desc, mem_type, md);

    assert(ret == NIXL_SUCCESS);
}

void
deallocateAndDeregister(nixlBackendEngine *mooncake,
                        int dev_id,
                        nixl_mem_t mem_type,
                        void *&addr,
                        nixlBackendMD *&md) {
    mooncake->deregisterMem(md);
    releaseBuffer(mem_type, dev_id, addr);
}

void
loadRemote(nixlBackendEngine *mooncake,
           int dev_id,
           std::string agent,
           nixl_mem_t mem_type,
           void *addr,
           size_t len,
           nixlBackendMD *&lmd,
           nixlBackendMD *&rmd) {
    nixlBlobDesc info;
    info.addr = (uintptr_t)addr;
    info.len = len;
    info.devId = dev_id;
    mooncake->getPublicData(lmd, info.metaInfo);

    // Not applicable to Mooncake backend
    // assert(info.metaInfo.size() > 0);

    // We get the data from the cetnral location and populate the backend, and receive remote_meta
    int ret = mooncake->loadRemoteMD(info, mem_type, agent, rmd);
    assert(NIXL_SUCCESS == ret);
}

void
populateDescs(nixl_meta_dlist_t &descs,
              int dev_id,
              void *addr,
              int desc_cnt,
              size_t desc_size,
              nixlBackendMD *&md) {
    for (int i = 0; i < desc_cnt; i++) {
        nixlMetaDesc req;
        req.addr = (uintptr_t)(((char *)addr) + i * desc_size); // random offset
        req.len = desc_size;
        req.devId = dev_id;
        req.metadataP = md;
        descs.addDesc(req);
    }
}

static string
op2string(nixl_xfer_op_t op, bool hasNotif) {
    if (op == NIXL_READ && !hasNotif) return string("READ");
    if (op == NIXL_WRITE && !hasNotif) return string("WRITE");
    if (op == NIXL_READ && hasNotif) return string("READ/NOTIF");
    if (op == NIXL_WRITE && hasNotif) return string("WRITE/NOTIF");

    return string("ERR-OP");
}

void
performTransfer(nixlBackendEngine *mooncake1,
                nixlBackendEngine *mooncake2,
                nixl_meta_dlist_t &req_src_descs,
                nixl_meta_dlist_t &req_dst_descs,
                void *addr1,
                void *addr2,
                size_t len,
                nixl_xfer_op_t op,
                testHndlIterator &hiter,
                bool progress,
                bool use_notif) {
    int ret2;
    nixl_status_t ret3;
    void *chkptr1, *chkptr2;

    std::string remote_agent("Agent2");

    if (mooncake1 == mooncake2) remote_agent = "Agent1";

    std::string test_str("test");
    std::cout << "\t" << op2string(op, use_notif) << " from " << addr1 << " to " << addr2 << "\n";

    nixl_opt_b_args_t opt_args;
    opt_args.notifMsg = test_str;
    opt_args.hasNotif = use_notif;


    // Posting a request, to be updated to return an async handler,
    // or an ID that later can be used to check the status as a new method
    // Also maybe we would remove the WRITE and let the backend class decide the op
    if (hiter.needPrep()) {
        nixlBackendReqH *new_handle = nullptr;
        ret3 = mooncake1->prepXfer(
            op, req_src_descs, req_dst_descs, remote_agent, new_handle, &opt_args);
        assert(ret3 == NIXL_SUCCESS);
        hiter.setHandle(new_handle);
    }
    nixlBackendReqH *&handle = hiter.getHandle();
    ret3 = mooncake1->postXfer(op, req_src_descs, req_dst_descs, remote_agent, handle, &opt_args);
    assert(ret3 == NIXL_SUCCESS || ret3 == NIXL_IN_PROG);


    if (ret3 == NIXL_SUCCESS) {
        cout << "\t\tWARNING: Tansfer request completed immediately - no testing non-inline path"
             << endl;
    } else {
        cout << "\t\tNOTE: Testing non-inline Transfer path!" << endl;

        while (ret3 == NIXL_IN_PROG) {
            ret3 = mooncake1->checkXfer(handle);
            assert(ret3 == NIXL_SUCCESS || ret3 == NIXL_IN_PROG);
        }
    }

    if (hiter.needRelease()) {
        hiter.unsetHandle();
        mooncake1->releaseReqH(handle);
    }

    if (use_notif) {
        /* Test notification path */
        notif_list_t target_notifs;

        cout << "\t\tChecking notification flow: " << flush;
        ret2 = 0;

        while (ret2 == 0) {
            ret3 = mooncake2->getNotifs(target_notifs);
            ret2 = target_notifs.size();
            assert(ret3 == NIXL_SUCCESS);
        }

        assert(ret2 == 1);

        assert(target_notifs.front().first == "Agent1");
        assert(target_notifs.front().second == test_str);

        cout << "OK" << endl;
    }

    cout << "\t\tData verification: " << flush;

    chkptr1 = getValidationPtr(req_src_descs.getType(), addr1, len);
    chkptr2 = getValidationPtr(req_dst_descs.getType(), addr2, len);

    // Perform correctness check.
    for (size_t i = 0; i < len; i++) {
        assert(((uint8_t *)chkptr1)[i] == ((uint8_t *)chkptr2)[i]);
    }

    releaseValidationPtr(req_src_descs.getType(), chkptr1);
    releaseValidationPtr(req_dst_descs.getType(), chkptr2);

    cout << "OK" << endl;
}

void
test_intra_agent_transfer(bool p_thread, nixlBackendEngine *mooncake, nixl_mem_t mem_type) {

    std::cout << std::endl << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << "   Intra-agent memory transfer test: " << "P-Thr=" << (p_thread ? "ON" : "OFF")
              << ", " << memType2Str(mem_type) << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << std::endl << std::endl;

    std::string agent1("Agent1");
    nixl_status_t ret1;

    int iter = 10;

    assert(mooncake->supportsLocal());

    // connection info is still a string
    std::string conn_info1;
    ret1 = mooncake->getConnInfo(conn_info1);
    assert(ret1 == NIXL_SUCCESS);
    ret1 = mooncake->loadRemoteConnInfo(agent1, conn_info1);
    assert(ret1 == NIXL_SUCCESS);

    std::cout << "Local connection complete\n";

    // Number of transfer descriptors
    int desc_cnt = 64;
    // Size of a single descriptor
    size_t desc_size = 1 * 1024 * 1024;
    size_t len = desc_cnt * desc_size;

    void *addr1, *addr2;
    nixlBackendMD *lmd1, *lmd2;
    allocateAndRegister(mooncake, 0, mem_type, addr1, len, lmd1);
    allocateAndRegister(mooncake, 0, mem_type, addr2, len, lmd2);

    // string descs unnecessary, convert meta locally
    nixlBackendMD *rmd2;
    ret1 = mooncake->loadLocalMD(lmd2, rmd2);
    assert(ret1 == NIXL_SUCCESS);

    nixl_meta_dlist_t req_src_descs(mem_type);
    populateDescs(req_src_descs, 0, addr1, desc_cnt, desc_size, lmd1);

    nixl_meta_dlist_t req_dst_descs(mem_type);
    populateDescs(req_dst_descs, 0, addr2, desc_cnt, desc_size, rmd2);

    nixl_xfer_op_t ops[] = {NIXL_READ, NIXL_WRITE};
    bool use_notifs[] = {true, false};

    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[i]); i++) {

        for (bool use_notif : use_notifs) {
            cout << endl
                 << op2string(ops[i], use_notif) << " test (" << iter << ") iterations" << endl;
            for (int k = 0; k < iter; k++) {
                /* Init data */
                doMemset(mem_type, 0, addr1, 0xbb, len);
                doMemset(mem_type, 0, addr2, 0, len);

                /* Test */
                testHndlIterator hiter(false);
                performTransfer(mooncake,
                                mooncake,
                                req_src_descs,
                                req_dst_descs,
                                addr1,
                                addr2,
                                len,
                                ops[i],
                                hiter,
                                p_thread,
                                use_notif);
            }
        }
    }

    mooncake->unloadMD(rmd2);
    deallocateAndDeregister(mooncake, 0, mem_type, addr1, lmd1);
    deallocateAndDeregister(mooncake, 0, mem_type, addr2, lmd2);

    mooncake->disconnect(agent1);
}

void
test_inter_agent_transfer(bool p_thread,
                          bool reuse_hndl,
                          nixlBackendEngine *mooncake1,
                          nixl_mem_t src_mem_type,
                          int src_dev_id,
                          nixlBackendEngine *mooncake2,
                          nixl_mem_t dst_mem_type,
                          int dst_dev_id) {
    int ret;
    int iter = 10;

    std::cout << std::endl << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << "    Inter-agent memory transfer test " << std::endl;
    std::cout << "         P-Thr=" << (p_thread ? "ON" : "OFF") << std::endl;
    std::cout << "         Handler-reuse=" << (reuse_hndl ? "ON" : "OFF") << std::endl;
    std::cout << "         (" << memType2Str(src_mem_type) << " -> " << memType2Str(dst_mem_type)
              << ")" << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << std::endl << std::endl;

    // Example: assuming two agents running on the same machine,
    // with separate memory regions in DRAM
    std::string agent1("Agent1");
    std::string agent2("Agent2");

    // We get the required connection info from Mooncake to be put on the central
    // location and ask for it for a remote node
    std::string conn_info1, conn_info2;
    ret = mooncake1->getConnInfo(conn_info1);
    assert(ret == NIXL_SUCCESS);
    ret = mooncake2->getConnInfo(conn_info2);
    assert(ret == NIXL_SUCCESS);

    // We assumed we put them to central location and now receiving it on the other process
    ret = mooncake1->loadRemoteConnInfo(agent2, conn_info2);
    assert(ret == NIXL_SUCCESS);

    // TODO: Causes race condition - investigate conn management implementation
    // ret = mooncake2->loadRemoteConnInfo (agent1, conn_info1);

    std::cout << "Synchronous handshake complete\n";

    std::string test_str("test");
    mooncake1->genNotif(agent2, test_str);
    int ret_gen = 0;
    notif_list_t target_notif_gen;
    while (ret_gen == 0) {
        int ret3_gen = mooncake2->getNotifs(target_notif_gen);
        ret_gen = target_notif_gen.size();
        assert(ret3_gen == NIXL_SUCCESS);
    }
    assert(target_notif_gen.front().second == test_str);
    cout << "\t\tGenNotify Data verification success!" << flush;

    // Number of transfer descriptors
    int desc_cnt = 64;
    // Size of a single descriptor
    size_t desc_size = 1 * 1024 * 1024;
    size_t len = desc_cnt * desc_size;

    void *addr1 = NULL, *addr2 = NULL;
    nixlBackendMD *lmd1, *lmd2;
    allocateAndRegister(mooncake1, src_dev_id, src_mem_type, addr1, len, lmd1);
    allocateAndRegister(mooncake2, dst_dev_id, dst_mem_type, addr2, len, lmd2);

    nixlBackendMD *rmd1 /*, *rmd2*/;
    loadRemote(mooncake1, dst_dev_id, agent2, dst_mem_type, addr2, len, lmd2, rmd1);
    // loadRemote(mooncake2, src_dev_id, agent1, src_mem_type, addr1, len, lmd1, rmd2);

    nixl_meta_dlist_t req_src_descs(src_mem_type);
    populateDescs(req_src_descs, src_dev_id, addr1, desc_cnt, desc_size, lmd1);

    nixl_meta_dlist_t req_dst_descs(dst_mem_type);
    populateDescs(req_dst_descs, dst_dev_id, addr2, desc_cnt, desc_size, rmd1);

    nixl_xfer_op_t ops[] = {NIXL_READ, NIXL_WRITE};
    bool use_notifs[] = {true, false};

    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[i]); i++) {

        for (bool use_notif : use_notifs) {
            cout << endl
                 << op2string(ops[i], use_notif) << " test (" << iter << ") iterations" << endl;
            testHndlIterator hiter(reuse_hndl);
            for (int k = 0; k < iter; k++) {
                /* Init data */
                doMemset(src_mem_type, src_dev_id, addr1, 0xbb, len);
                doMemset(dst_mem_type, dst_dev_id, addr2, 0xda, len);

                /* Test */
                if ((k + 1) == iter) {
                    /* If this is the last iteration */
                    hiter.isLast();
                }
                performTransfer(mooncake1,
                                mooncake2,
                                req_src_descs,
                                req_dst_descs,
                                addr1,
                                addr2,
                                len,
                                ops[i],
                                hiter,
                                !p_thread,
                                use_notif);
            }
        }
    }

    // As well as all the remote notes, asking to remove them one by one
    // need to provide list of descs
    mooncake1->unloadMD(rmd1);
    // mooncake2->unloadMD (rmd2);

    // Release memory regions
    deallocateAndDeregister(mooncake1, src_dev_id, src_mem_type, addr1, lmd1);
    deallocateAndDeregister(mooncake2, dst_dev_id, dst_mem_type, addr2, lmd2);

    // Test one-sided disconnect (initiator only)
    mooncake1->disconnect(agent2);

    // TODO: Causes race condition - investigate conn management implementation
    // mooncake2->disconnect(agent1);
}

int
main() {
    bool thread_on[2] = {false, true};
    nixlBackendEngine *mooncake[2][2] = {0};

    // Allocate Mooncake engines
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::stringstream s;
            s << "Agent" << (j + 1);
            mooncake[i][j] = createEngine(s.str(), thread_on[i]);
        }
    }

#ifdef HAVE_CUDA
    int dev_ids[2] = {0, 0};
    int n_vram_dev;
    if (cudaGetDeviceCount(&n_vram_dev) != cudaSuccess) {
        std::cout << "Call to cudaGetDeviceCount failed, assuming 0 devices";
        n_vram_dev = 0;
    }

    std::cout << "Detected " << n_vram_dev << " CUDA devices" << std::endl;
    if (n_vram_dev > 1) {
        dev_ids[1] = 1;
        dev_ids[0] = 0;
    }
#endif
    test_inter_agent_transfer(
        thread_on[0], false, mooncake[0][0], DRAM_SEG, 0, mooncake[0][1], DRAM_SEG, 0);

    for (int i = 0; i < 2; i++) {
        // Test local memory to local memory transfer
        //  std::cout << "thread_on" <<i<<thread_on[i]<<endl;
        //  test_intra_agent_transfer(thread_on[i], mooncake[i][0], DRAM_SEG);
#ifdef HAVE_CUDA
        if (n_vram_dev > 0) {
            test_intra_agent_transfer(thread_on[i], mooncake[i][0], VRAM_SEG);
        }
#endif
    }

    for (int i = 0; i < 2; i++) {
        test_inter_agent_transfer(
            thread_on[i], false, mooncake[i][0], DRAM_SEG, 0, mooncake[i][1], DRAM_SEG, 0);
        test_inter_agent_transfer(
            thread_on[i], true, mooncake[i][0], DRAM_SEG, 0, mooncake[i][1], DRAM_SEG, 0);

#ifdef HAVE_CUDA
        if (n_vram_dev > 1) {
            test_inter_agent_transfer(thread_on[i],
                                      false,
                                      mooncake[i][0],
                                      VRAM_SEG,
                                      dev_ids[0],
                                      mooncake[i][1],
                                      VRAM_SEG,
                                      dev_ids[1]);
            test_inter_agent_transfer(thread_on[i],
                                      true,
                                      mooncake[i][0],
                                      VRAM_SEG,
                                      dev_ids[0],
                                      mooncake[i][1],
                                      VRAM_SEG,
                                      dev_ids[1]);
            test_inter_agent_transfer(thread_on[i],
                                      true,
                                      mooncake[i][0],
                                      DRAM_SEG,
                                      dev_ids[0],
                                      mooncake[i][1],
                                      VRAM_SEG,
                                      dev_ids[1]);
            test_inter_agent_transfer(thread_on[i],
                                      true,
                                      mooncake[i][0],
                                      VRAM_SEG,
                                      dev_ids[0],
                                      mooncake[i][1],
                                      DRAM_SEG,
                                      dev_ids[1]);
        }
#endif
    }

#ifdef HAVE_CUDA
    if (n_vram_dev > 1) {
        // Test if registering on a different GPU fails correctly
        allocateWrongGPUTest(mooncake[0][0], 1);
        std::cout << "Verified registration on wrong GPU fails correctly\n";
    }
#endif

    // Deallocate Mooncake engines
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            releaseEngine(mooncake[i][j]);
        }
    }
}
