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

#include "ucx_backend.h"
#include "test_utils.h"

using namespace std;


#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda.h>

int gpu_id = 0;

static void checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif


class testHndlIterator {
private:
    bool reuse;
    bool set;
    bool prepare;
    bool release;
    nixlBackendReqH* handle;
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
        nixl_exit_on_failure(!set, "Handler was not released");
    }

    bool needPrep() {
        if (reuse) {
            if (!prepare) {
                return false;
            }
        }
        return true;
    }

    bool needRelease() {
        return release;
    }

    void isLast() {
        if (reuse) {
            release = true;
        }
    }

    void setHandle(nixlBackendReqH *_handle)
    {
        nixl_exit_on_failure(!set, "Handler was not released");
        handle = _handle;
        set = true;
        if (reuse) {
            prepare = false;
        }
    }

    void unsetHandle() {
        nixl_exit_on_failure(set, "Handler was not set");
        set = false;
    }

    nixlBackendReqH *&getHandle() {
        nixl_exit_on_failure(set, "Handler was not set");
        return handle;
    }
};

nixlUcxEngine *
createEngine(std::string name, bool p_thread) {
    nixlBackendInitParams init;
    nixl_b_params_t       custom_params;

    init.enableProgTh = p_thread;
    init.pthrDelay    = 100;
    init.localAgent   = name;
    init.customParams = &custom_params;
    init.type         = "UCX";

    auto ucx = nixlUcxEngine::create(init).release();
    nixl_exit_on_failure(!ucx->getInitErr(), "Failed to initialize worker1");

    return ucx;
}

void
releaseEngine(nixlUcxEngine *ucx) {
    delete ucx;
}

std::string memType2Str(nixl_mem_t mem_type)
{
    switch(mem_type) {
    case DRAM_SEG:
        return std::string("DRAM");
    case VRAM_SEG:
        return std::string("VRAM");
    case BLK_SEG:
        return std::string("BLOCK");
    case FILE_SEG:
        return std::string("FILE");
    default:
        nixl_exit_on_failure(false, "Unsupported memory type!");
    }
    return std::string("");
}


#ifdef HAVE_CUDA

static int cudaQueryAddr(void *address, bool &is_dev,
                         CUdevice &dev, CUcontext &ctx)
{
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


void allocateBuffer(nixl_mem_t mem_type, int dev_id, size_t len, void* &addr)
{
    switch(mem_type) {
    case DRAM_SEG:
        addr = calloc(1, len);
        break;
#ifdef HAVE_CUDA
    case VRAM_SEG:{
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
        nixl_exit_on_failure(false, "Unsupported memory type!");
    }
    nixl_exit_on_failure((addr != nullptr), "Failed to allocate buffer");
}

void releaseBuffer(nixl_mem_t mem_type, int dev_id, void* &addr)
{
    switch(mem_type) {
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
        nixl_exit_on_failure(false, "Unsupported memory type!");
    }
}

void doMemset(nixl_mem_t mem_type, int dev_id, void *addr, char byte, size_t len)
{
    switch(mem_type) {
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
        nixl_exit_on_failure(false, "Unsupported memory type!");
    }
}

void *getValidationPtr(nixl_mem_t mem_type, void *addr, size_t len)
{
    switch(mem_type) {
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
        nixl_exit_on_failure(false, "Unsupported memory type!");
    }
    return nullptr;
}

void *releaseValidationPtr(nixl_mem_t mem_type, void *addr)
{
    switch(mem_type) {
    case DRAM_SEG:
        break;
#ifdef HAVE_CUDA
    case VRAM_SEG:
        free(addr);
        break;
#endif
    default:
        nixl_exit_on_failure(false, "Unsupported memory type!");
    }
    return nullptr;
}

void
allocateWrongGPUTest(nixlUcxEngine *ucx, int dev_id) {
    nixlBlobDesc desc = {0};
    nixlBackendMD* md;
    void* buf;

    allocateBuffer(VRAM_SEG, dev_id, desc.len, buf);

    desc.devId = dev_id;
    desc.addr = (uint64_t) buf;

    int ret = ucx->registerMem(desc, VRAM_SEG, md);

    nixl_exit_on_failure((ret == NIXL_ERR_NOT_SUPPORTED), "Failed to register memory", "test");

    releaseBuffer(VRAM_SEG, dev_id, buf);
}

void
allocateAndRegister(nixlUcxEngine *ucx,
                    int dev_id,
                    nixl_mem_t mem_type,
                    void *&addr,
                    size_t len,
                    nixlBackendMD *&md) {
    nixlBlobDesc desc;

    allocateBuffer(mem_type, dev_id, len, addr);

    desc.addr   = (uintptr_t) addr;
    desc.len    = len;
    desc.devId = dev_id;

    int ret = ucx->registerMem(desc, mem_type, md);

    nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to allocate and register memory");
}

void
deallocateAndDeregister(nixlUcxEngine *ucx,
                        int dev_id,
                        nixl_mem_t mem_type,
                        void *&addr,
                        nixlBackendMD *&md) {
    ucx->deregisterMem(md);
    releaseBuffer(mem_type, dev_id, addr);
}

void
loadRemote(nixlUcxEngine *ucx,
           int dev_id,
           std::string agent,
           nixl_mem_t mem_type,
           void *addr,
           size_t len,
           nixlBackendMD *&lmd,
           nixlBackendMD *&rmd) {
    nixlBlobDesc info;
    info.addr     = (uintptr_t) addr;
    info.len      = len;
    info.devId    = dev_id;
    ucx->getPublicData(lmd, info.metaInfo);

    nixl_exit_on_failure((info.metaInfo.size() > 0), "Failed to get public data");

    // We get the data from the cetnral location and populate the backend, and receive remote_meta
    int ret = ucx->loadRemoteMD(info, mem_type, agent, rmd);
    nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to load remote MD");
}

void populateDescs(nixl_meta_dlist_t &descs, int dev_id, void *addr, int desc_cnt, size_t desc_size, nixlBackendMD* &md)
{
    for(int i = 0; i < desc_cnt; i++) {
        nixlMetaDesc req;
        req.addr     = (uintptr_t) (((char*) addr) + i * desc_size); //random offset
        req.len      = desc_size;
        req.devId    = dev_id;
        req.metadataP = md;
        descs.addDesc(req);
    }
}

static string op2string(nixl_xfer_op_t op, bool hasNotif)
{
    if(op == NIXL_READ && !hasNotif)
        return string("READ");
    if(op == NIXL_WRITE && !hasNotif)
        return string("WRITE");
    if(op == NIXL_READ && hasNotif)
        return string("READ/NOTIF");
    if(op == NIXL_WRITE && hasNotif)
        return string("WRITE/NOTIF");

    return string("ERR-OP");
}

void
performTransfer(nixlUcxEngine *ucx1,
                nixlUcxEngine *ucx2,
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

    std::string remote_agent ("Agent2");

    if(ucx1 == ucx2) remote_agent = "Agent1";

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
        ret3 =
            ucx1->prepXfer(op, req_src_descs, req_dst_descs, remote_agent, new_handle, &opt_args);
        nixl_exit_on_failure(ret3, "Failed to prep xfer");
        hiter.setHandle(new_handle);
    }
    nixlBackendReqH *&handle = hiter.getHandle();
    ret3 = ucx1->postXfer(op, req_src_descs, req_dst_descs, remote_agent, handle, &opt_args);
    nixl_exit_on_failure(ret3 >= NIXL_SUCCESS, "Failed to post xfer");

    if (ret3 == NIXL_SUCCESS) {
        cout << "\t\tWARNING: Tansfer request completed immediately - no testing non-inline path" << endl;
    } else {
        cout << "\t\tNOTE: Testing non-inline Transfer path!" << endl;

        while(ret3 == NIXL_IN_PROG) {
            ret3 = ucx1->checkXfer(handle);
            if(progress){
                ucx2->progress();
            }
            nixl_exit_on_failure(ret3 >= NIXL_SUCCESS, "Failed to check xfer");
        }
    }

    if (hiter.needRelease()) {
        hiter.unsetHandle();
        ucx1->releaseReqH(handle);
    }

    if(use_notif) {
            /* Test notification path */
        notif_list_t target_notifs;

        cout << "\t\tChecking notification flow: " << flush;
        ret2 = 0;

        while(ret2 == 0){
            ret3 = ucx2->getNotifs(target_notifs);
            ret2 = target_notifs.size();
            if(progress){
                ucx1->progress();
            }
            nixl_exit_on_failure(ret3, "Failed to get notifs");
        }

        nixl_exit_on_failure((ret2 == 1), "Incorrect number of target notifs");

        nixl_exit_on_failure((target_notifs.front().first == "Agent1"),
                             "Incorrect front notif source");
        nixl_exit_on_failure((target_notifs.front().second == test_str),
                             "Incorrect front notif message");

        cout << "OK" << endl;
    }

    cout << "\t\tData verification: " << flush;

    chkptr1 = getValidationPtr(req_src_descs.getType(), addr1, len);
    chkptr2 = getValidationPtr(req_dst_descs.getType(), addr2, len);

    // Perform correctness check.
    for (size_t i = 0; i < len; i++) {
        nixl_exit_on_failure((((uint8_t *)chkptr1)[i] == ((uint8_t *)chkptr2)[i]), "Data mismatch");
    }

    releaseValidationPtr(req_src_descs.getType(), chkptr1);
    releaseValidationPtr(req_dst_descs.getType(), chkptr2);

    cout << "OK" << endl;
}

void
test_intra_agent_transfer(bool p_thread, nixlUcxEngine *ucx, nixl_mem_t mem_type) {

    std::cout << std::endl << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << "   Intra-agent memory transfer test: "
              << "P-Thr=" << (p_thread ? "ON" : "OFF") << ", " << memType2Str(mem_type)
              << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << std::endl << std::endl;

    std::string agent1("Agent1");
    nixl_status_t ret1;

    int iter = 10;

    nixl_exit_on_failure(ucx->supportsLocal(), "Failed to get conn info");

    //connection info is still a string
    std::string conn_info1;
    ret1 = ucx->getConnInfo(conn_info1);
    nixl_exit_on_failure((ret1 == NIXL_SUCCESS), "Failed to get conn info");
    ret1 = ucx->loadRemoteConnInfo(agent1, conn_info1);
    nixl_exit_on_failure((ret1 == NIXL_SUCCESS), "Failed to load remote conn info");

    std::cout << "Local connection complete\n";

    // Number of transfer descriptors
    int desc_cnt = 64;
    // Size of a single descriptor
    size_t desc_size = 1 * 1024 * 1024;
    size_t len = desc_cnt * desc_size;

    void *addr1, *addr2;
    nixlBackendMD *lmd1, *lmd2;
    allocateAndRegister(ucx, 0, mem_type, addr1, len, lmd1);
    allocateAndRegister(ucx, 0, mem_type, addr2, len, lmd2);

    //string descs unnecessary, convert meta locally
    nixlBackendMD* rmd2;
    ret1 = ucx->loadLocalMD(lmd2, rmd2);
    nixl_exit_on_failure((ret1 == NIXL_SUCCESS), "Failed to load local MD");
    nixl_meta_dlist_t req_src_descs (mem_type);
    populateDescs(req_src_descs, 0, addr1, desc_cnt, desc_size, lmd1);

    nixl_meta_dlist_t req_dst_descs (mem_type);
    populateDescs(req_dst_descs, 0, addr2, desc_cnt, desc_size, rmd2);

    nixl_xfer_op_t ops[] = {  NIXL_READ, NIXL_WRITE };
    bool use_notifs[] = { true, false };

    for (size_t i = 0; i < sizeof(ops)/sizeof(ops[i]); i++) {

        for(bool use_notif : use_notifs) {
            cout << endl << op2string(ops[i], use_notif) << " test (" << iter << ") iterations" <<endl;
            for(int k = 0; k < iter; k++ ) {
                /* Init data */
                doMemset(mem_type, 0, addr1, 0xbb, len);
                doMemset(mem_type, 0, addr2, 0, len);

                /* Test */
                testHndlIterator hiter(false);
                performTransfer(ucx, ucx, req_src_descs, req_dst_descs,
                                addr1, addr2, len, ops[i], hiter, p_thread, use_notif);
            }
        }
    }

    ucx->unloadMD (rmd2);
    deallocateAndDeregister(ucx, 0, mem_type, addr1, lmd1);
    deallocateAndDeregister(ucx, 0, mem_type, addr2, lmd2);

    ucx->disconnect(agent1);
}

void
test_inter_agent_transfer(bool p_thread,
                          bool reuse_hndl,
                          nixlUcxEngine *ucx1,
                          nixl_mem_t src_mem_type,
                          int src_dev_id,
                          nixlUcxEngine *ucx2,
                          nixl_mem_t dst_mem_type,
                          int dst_dev_id) {
    int ret;
    int iter = 10;

    std::cout << std::endl << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << "    Inter-agent memory transfer test " << std::endl;
    std::cout << "         P-Thr=" << (p_thread ? "ON" : "OFF") << std::endl;
    std::cout << "         Handler-reuse=" << (reuse_hndl ? "ON" : "OFF") << std::endl;
    std::cout << "         (" << memType2Str(src_mem_type) << " -> "
                << memType2Str(dst_mem_type) << ")" << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << std::endl << std::endl;

    // Example: assuming two agents running on the same machine,
    // with separate memory regions in DRAM
    std::string agent1("Agent1");
    std::string agent2("Agent2");

    // We get the required connection info from UCX to be put on the central
    // location and ask for it for a remote node
    std::string conn_info1, conn_info2;
    ret = ucx1->getConnInfo(conn_info1);
    nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to get conn info");
    ret = ucx2->getConnInfo(conn_info2);
    nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to get conn info");

    // We assumed we put them to central location and now receiving it on the other process
    ret = ucx1->loadRemoteConnInfo(agent2, conn_info2);
    nixl_exit_on_failure((ret == NIXL_SUCCESS), "Failed to load remote conn info");

    // TODO: Causes race condition - investigate conn management implementation
    // ret = ucx2->loadRemoteConnInfo (agent1, conn_info1);

    std::cout << "Synchronous handshake complete\n";

    // Number of transfer descriptors
    int desc_cnt = 64;
    // Size of a single descriptor
    size_t desc_size = 1 * 1024 * 1024;
    size_t len = desc_cnt * desc_size;

    void *addr1 = NULL, *addr2 = NULL;
    nixlBackendMD *lmd1, *lmd2;
    allocateAndRegister(ucx1, src_dev_id, src_mem_type, addr1, len, lmd1);
    allocateAndRegister(ucx2, dst_dev_id, dst_mem_type, addr2, len, lmd2);

    nixlBackendMD *rmd1 /*, *rmd2*/;
    loadRemote(ucx1, dst_dev_id,  agent2, dst_mem_type, addr2, len, lmd2, rmd1);
    //loadRemote(ucx2, src_dev_id, agent1, src_mem_type, addr1, len, lmd1, rmd2);

    nixl_meta_dlist_t req_src_descs (src_mem_type);
    populateDescs(req_src_descs, src_dev_id, addr1, desc_cnt, desc_size, lmd1);

    nixl_meta_dlist_t req_dst_descs (dst_mem_type);
    populateDescs(req_dst_descs, dst_dev_id, addr2, desc_cnt, desc_size, rmd1);

    nixl_xfer_op_t ops[] = {  NIXL_READ, NIXL_WRITE };
    bool use_notifs[] = { true, false };

    for (size_t i = 0; i < sizeof(ops)/sizeof(ops[i]); i++) {

        for(bool use_notif : use_notifs) {
            cout << endl << op2string(ops[i], use_notif) << " test (" << iter << ") iterations" <<endl;
            testHndlIterator hiter(reuse_hndl);
            for(int k = 0; k < iter; k++ ) {
                /* Init data */
                doMemset(src_mem_type, src_dev_id, addr1, 0xbb, len);
                doMemset(dst_mem_type, dst_dev_id, addr2, 0xda, len);

                /* Test */
                if ((k+1) == iter) {
                    /* If this is the last iteration */
                    hiter.isLast();
                }
                performTransfer(ucx1, ucx2, req_src_descs, req_dst_descs,
                                addr1, addr2, len, ops[i], hiter, !p_thread, use_notif);
            }
        }
    }

    cout << endl << "Test genNotif operation" << endl;

    for(int k = 0; k < iter; k++) {
        std::string test_str("test");
        std::string tgt_agent("Agent2");
        notif_list_t target_notifs;

        cout << "\t gnNotif to Agent2" <<endl;

        ucx1->genNotif(tgt_agent, test_str);

        cout << "\t\tChecking notification flow: " << flush;
        ret = 0;

        nixl_status_t ret2;

        while(ret == 0){
            ret2 = ucx2->getNotifs(target_notifs);
            ret = target_notifs.size();
            nixl_exit_on_failure((ret2 == NIXL_SUCCESS), "Failed to get notifs");
        }

        nixl_exit_on_failure((ret == 1), "Incorrect number of target notifs");
        nixl_exit_on_failure((target_notifs.front().first == "Agent1"),
                             "Incorrect front notif source");
        nixl_exit_on_failure((target_notifs.front().second == test_str),
                             "Incorrect front notif message");

        cout << "OK" << endl;
    }

    // As well as all the remote notes, asking to remove them one by one
    // need to provide list of descs
    ucx1->unloadMD (rmd1);
    //ucx2->unloadMD (rmd2);

    // Release memory regions
    deallocateAndDeregister(ucx1, src_dev_id, src_mem_type, addr1, lmd1);
    deallocateAndDeregister(ucx2, dst_dev_id, dst_mem_type, addr2, lmd2);

    // Test one-sided disconnect (initiator only)
    ucx1->disconnect(agent2);

    // TODO: Causes race condition - investigate conn management implementation
    //ucx2->disconnect(agent1);
}

int main()
{
    bool thread_on[2] = {false, true};
    nixlUcxEngine *ucx[2][2] = {0};

    // Allocate UCX engines
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            std::stringstream s;
            s << "Agent" << (j + 1);
            ucx[i][j] = createEngine(s.str(), thread_on[i]);
        }
    }

#ifdef HAVE_CUDA
    int dev_ids[2] = { 0 , 0 };
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

    for(int i = 0; i < 2; i++) {
        //Test local memory to local memory transfer
        test_intra_agent_transfer(thread_on[i], ucx[i][0], DRAM_SEG);
#ifdef HAVE_CUDA
        if (n_vram_dev > 0) {
            test_intra_agent_transfer(thread_on[i], ucx[i][0], VRAM_SEG);
        }
#endif
    }

    for(int i = 0; i < 2; i++) {
        test_inter_agent_transfer(thread_on[i], false,
                                  ucx[i][0], DRAM_SEG, 0,
                                  ucx[i][1], DRAM_SEG, 0);
        test_inter_agent_transfer(thread_on[i], true,
                                  ucx[i][0], DRAM_SEG, 0,
                                  ucx[i][1], DRAM_SEG, 0);

#ifdef HAVE_CUDA
        if (n_vram_dev > 1) {
            test_inter_agent_transfer(thread_on[i], false,
                                      ucx[i][0], VRAM_SEG, dev_ids[0],
                                      ucx[i][1], VRAM_SEG, dev_ids[1]);
            test_inter_agent_transfer(thread_on[i], true,
                                      ucx[i][0], VRAM_SEG, dev_ids[0],
                                      ucx[i][1], VRAM_SEG, dev_ids[1]);
            test_inter_agent_transfer(thread_on[i], true,
                                      ucx[i][0], DRAM_SEG, dev_ids[0],
                                      ucx[i][1], VRAM_SEG, dev_ids[1]);
            test_inter_agent_transfer(thread_on[i], true,
                                      ucx[i][0], VRAM_SEG, dev_ids[0],
                                      ucx[i][1], DRAM_SEG, dev_ids[1]);
        }
#endif
    }

#ifdef HAVE_CUDA
    if (n_vram_dev > 1) {
		//Test if registering on a different GPU fails correctly
		allocateWrongGPUTest(ucx[0][0], 1);
		std::cout << "Verified registration on wrong GPU fails correctly\n";
	}
#endif

    // Deallocate UCX engines
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            releaseEngine(ucx[i][j]);
        }
    }
}
