#include "common.h"
#include "gtest/gtest.h"

#include "nixl.h"
#include "nixl_types.h"

#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <thread>
#include <mutex>
#include <optional>

#include <cuda_runtime.h>
#include "ucx/nixl_gdaki_device.cuh"

#define UCS_NSEC_PER_SEC   1000000000ul
#define NS_TO_SEC(ns)      ((ns)*1.0 / (UCS_NSEC_PER_SEC))
#define MAX_THREADS        1024

__device__ inline unsigned long long GdakiGetTimeNs()
{
    unsigned long long globaltimer;
    // 64-bit GPU global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

template<nixl_gpu_xfer_coordination_level_t level>
__device__ constexpr size_t GgakiTestMyReqIdx()
{
    switch (level) {
        case NIXL_GPU_XFER_COORDINATION_THREAD: return threadIdx.x;
        case NIXL_GPU_XFER_COORDINATION_WARP:   return threadIdx.x / warpSize;
        case NIXL_GPU_XFER_COORDINATION_BLOCK:  return 0;
        default:                                return 0;
    }
}

template<nixl_gpu_xfer_coordination_level_t level>
__global__ void
GdakiPostGpuXferReqLaunchKernel(nixlGpuXferReqH *req_hdnl, size_t num_iters,
                                uint64_t signal_inc,
                                unsigned long long *start_time_ptr,
                                unsigned long long *end_time_ptr) {
    __shared__ nixlGpuXferStatusH xfer_status[MAX_THREADS];
    nixlGpuXferStatusH *xfer_status_ptr = &xfer_status[GgakiTestMyReqIdx<level>()];
    nixl_status_t status;

    assert(GgakiTestMyReqIdx<level>() < MAX_THREADS);

    if (threadIdx.x == 0) {
        unsigned long long start_time = GdakiGetTimeNs();
        *start_time_ptr = start_time;
    }

    for (size_t i = 0; i < num_iters; ++i) {
        status = nixlPostGpuXferReq<level>(req_hdnl, signal_inc, xfer_status_ptr);
        if (status != NIXL_SUCCESS) {
            printf("Failed to execute batch %d\n", status);
            return;
        }

        status = nixlGetGpuXferReqStatus<level>(xfer_status_ptr);
        if (status != NIXL_SUCCESS) {
            printf("Failed to progress request %d\n", status);
            return;
        }
    }

    if (threadIdx.x == 0) {
        unsigned long long end_time = GdakiGetTimeNs();
        *end_time_ptr = end_time;
    }
}

template<size_t indcnt, nixl_gpu_xfer_coordination_level_t level, bool is_null_status = false>
__global__ void
GdakiPostPartialGpuXferReqLaunchKernel(nixlGpuXferReqH *req_hdnl,
                                       size_t num_iters,
                                       uint64_t signal_inc,
                                       unsigned long long *start_time_ptr,
                                       unsigned long long *end_time_ptr,
                                       bool has_iovs = true) {
    __shared__ nixlGpuXferStatusH xfer_status[MAX_THREADS];
    nixlGpuXferStatusH *xfer_status_ptr = is_null_status ? nullptr : &xfer_status[GgakiTestMyReqIdx<level>()];
    nixl_status_t status;
    __shared__ int indices[indcnt];
    __shared__ size_t sizes[indcnt];
    __shared__ size_t src_offsets[indcnt];
    __shared__ size_t dst_offsets[indcnt];

    assert(GgakiTestMyReqIdx<level>() < MAX_THREADS);

    if (threadIdx.x == 0) {
        unsigned long long start_time = GdakiGetTimeNs();
        *start_time_ptr = start_time;

        if (has_iovs) {
            // Initialize shared arrays
            for (size_t i = 0; i < indcnt; ++i) {
                indices[i] = i * 4; // Buffers 0, 4, 8, 12, 16, 20, 24, 28
                sizes[i] = ((1024 * 16) / 2) + (i * 512); // Varying sizes: 8KB, 8.5KB, 9KB, 9.5KB, 10KB, 10.5KB, 11KB, 11.5KB
                src_offsets[i] = 0; // Start from beginning of each source buffer
                dst_offsets[i] = (1024 / 2) * i; // Staggered destination offsets
            }
        }
    }

    for (size_t i = 0; i < num_iters; ++i) {
        if (has_iovs) {
            status = nixlPostPartialGpuXferReq<level>(req_hdnl, signal_inc, indcnt,
                                                      indices, sizes, src_offsets,
                                                      dst_offsets, xfer_status_ptr);
        } else {
            status = nixlPostPartialGpuXferReq<level>(req_hdnl, signal_inc, 0,
                                                      nullptr, nullptr, nullptr,
                                                      nullptr, xfer_status_ptr);
        }

        if (status != NIXL_SUCCESS) {
            printf("Failed to execute batch %d\n", status);
            return;
        }

        if (!is_null_status) {
            status = nixlGetGpuXferReqStatus<level>(xfer_status_ptr);
            if (status != NIXL_SUCCESS) {
                printf("Failed to progress request %d\n", status);
                return;
            }
        }
    }

    if (threadIdx.x == 0) {
        unsigned long long end_time = GdakiGetTimeNs();
        *end_time_ptr = end_time;
    }
}

template<nixl_gpu_xfer_coordination_level_t level>
nixl_status_t
GdakiPostGpuXferReqwKernel(unsigned num_threads, nixlGpuXferReqH *req_hdnl,
                           size_t num_iters, uint64_t signal_inc,
                           unsigned long long *start_time_ptr,
                           unsigned long long *end_time_ptr) {
    nixl_status_t ret = NIXL_SUCCESS;
    cudaError_t err;

    GdakiPostGpuXferReqLaunchKernel<level><<<1, num_threads>>>(req_hdnl, num_iters, signal_inc, start_time_ptr, end_time_ptr);

    err = cudaDeviceSynchronize(); // Wait until kernel prints are visible
    if (err != cudaSuccess) {
        printf("Failed to synchronize: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    return ret;
}

template<size_t indcnt, nixl_gpu_xfer_coordination_level_t level, bool is_null_status = false>
nixl_status_t
GdakiPostGpuXferReqPartialwKernel(unsigned num_threads,
                                  nixlGpuXferReqH *req_hdnl, size_t num_iters,
                                  uint64_t signal_inc,
                                  unsigned long long *start_time_ptr,
                                  unsigned long long *end_time_ptr,
                                  bool has_iovs = true) {
    nixl_status_t ret = NIXL_SUCCESS;
    cudaError_t err;

    GdakiPostPartialGpuXferReqLaunchKernel<indcnt, level, is_null_status><<<1, num_threads>>>(
            req_hdnl, num_iters, signal_inc, start_time_ptr, end_time_ptr, has_iovs);

    err = cudaDeviceSynchronize(); // Wait until kernel prints are visible
    if (err != cudaSuccess) {
        printf("Failed to synchronize: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Failed to launch kernel: %s\n", cudaGetErrorString(err));
        ret = NIXL_ERR_BACKEND;
    }

    return ret;
}

namespace gtest {

class MemBuffer : std::shared_ptr<void> {
public:
    MemBuffer(size_t size, nixl_mem_t mem_type) :
        std::shared_ptr<void>(allocate(size, mem_type),
                              [mem_type](void *ptr) {
                                  release(ptr, mem_type);
                              }),
        size(size)
    {
    }

    operator uintptr_t() const
    {
        return reinterpret_cast<uintptr_t>(get());
    }

    size_t getSize() const
    {
        return size;
    }

private:
    static void *allocate(size_t size, nixl_mem_t mem_type)
    {
        void *ptr;
        return cudaSuccess == cudaMalloc(&ptr, size)? ptr : nullptr;
    }

    static void release(void *ptr, nixl_mem_t mem_type)
    {
        cudaFree(ptr);
    }

    const size_t size;
};

static const std::string NOTIF_MSG = "notification";

struct TestGdakiTransferParam {
    nixl_gpu_xfer_coordination_level_t level;
    std::string name;
    bool is_full_post_implemented;
    bool is_partial_post_implemented;
};

const char* gpuXferCoordinationLevelStr(nixl_gpu_xfer_coordination_level_t level) {
    switch (level) {
        case NIXL_GPU_XFER_COORDINATION_WARP:   return "WARP";
        case NIXL_GPU_XFER_COORDINATION_BLOCK:  return "BLOCK";
        case NIXL_GPU_XFER_COORDINATION_THREAD: return "THREAD";
// TODO       case NIXL_GPU_XFER_COORDINATION_GRID:   return "GRID";
        default:                                return "UNKNOWN";
    }
}

std::ostream& operator<<(std::ostream& os, const TestGdakiTransferParam& param) {
    return os << "TestGdakiTransferParam{coordination_level=" << gpuXferCoordinationLevelStr(param.level)
              << ", full_post=" << (param.is_full_post_implemented ? "true" : "false")
              << ", partial_post=" << (param.is_partial_post_implemented ? "true" : "false") << "}";
}
class TestGdakiTransfer : public testing::TestWithParam<std::tuple<std::string, TestGdakiTransferParam>> {
protected:
    static nixlAgentConfig getConfig()
    {
        return nixlAgentConfig(true, false, 0,  // Set listen=false, port=0
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_RW, 0,
                               100000);
    }

    nixl_b_params_t getBackendParams()
    {
        nixl_b_params_t params;

        if (getBackendName() == "UCX") {
            params["num_workers"] = "2";
        }

        return params;
    }

    void SetUp() override
    {
        if (cudaSetDevice(0) != cudaSuccess) {
            FAIL() << "Failed to set CUDA device 0";
        }

        // Create two agents
        for (size_t i = 0; i < 2; i++) {
            agents.emplace_back(std::make_unique<nixlAgent>(getAgentName(i),
                                                            getConfig()));
            nixlBackendH *backend_handle = nullptr;
            nixl_status_t status = agents.back()->createBackend(
                    getBackendName(), getBackendParams(), backend_handle);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_NE(backend_handle, nullptr);
            backend_handles.push_back(backend_handle);
        }
    }

    void TearDown() override
    {
        agents.clear();
        backend_handles.clear();
    }

    std::string getBackendName() const
    {
        return std::get<0>(GetParam());
    }

    nixl_gpu_xfer_coordination_level_t getFullTransferCoordinationLevel() const
    {
        return std::get<1>(GetParam()).level;
    }

    nixl_gpu_xfer_coordination_level_t getPartialTransferCoordinationLevel() const
    {
        return std::get<1>(GetParam()).level;
    }

    std::string getCoordinationLevelName() const
    {
        return std::get<1>(GetParam()).name;
    }

    bool isFullTransferCoordinationLevelImplemented() const
    {
        return std::get<1>(GetParam()).is_full_post_implemented;
    }

    bool isPartialTransferCoordinationLevelImplemented() const
    {
        return std::get<1>(GetParam()).is_partial_post_implemented;
    }

    static nixl_opt_args_t extra_params_ip(int remote)
    {
        nixl_opt_args_t extra_params;

        extra_params.ipAddr = "127.0.0.1";
        extra_params.port   = 0;
        return extra_params;
    }

    nixl_status_t fetchRemoteMD(int local = 0, int remote = 1)
    {
        auto extra_params = extra_params_ip(remote);

        return agents[local]->fetchRemoteMD(getAgentName(remote),
                                            &extra_params);
    }

    nixl_status_t checkRemoteMD(int local = 0, int remote = 1)
    {
        nixl_xfer_dlist_t descs(DRAM_SEG);
        return agents[local]->checkRemoteMD(getAgentName(remote), descs);
    }

    template<typename Desc>
    nixlDescList<Desc>
    makeDescList(const std::vector<MemBuffer> &buffers, nixl_mem_t mem_type)
    {
        nixlDescList<Desc> desc_list(mem_type);
        for (const auto &buffer : buffers) {
            desc_list.addDesc(Desc(buffer, buffer.getSize(), uint64_t(DEV_ID)));
        }
        return desc_list;
    }

    void registerMem(nixlAgent &agent, const std::vector<MemBuffer> &buffers,
                     nixl_mem_t mem_type)
    {
        auto reg_list = makeDescList<nixlBlobDesc>(buffers, mem_type);
        agent.registerMem(reg_list);
    }

    static bool wait_until_true(std::function<bool()> func, int retries = 500) {
        bool result;

        while (!(result = func()) && retries-- > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        return result;
    }

    void completeWireup(size_t from_agent, size_t to_agent) {
        nixl_notifs_t notifs;
        nixl_status_t status = getAgent(from_agent).genNotif(getAgentName(to_agent), NOTIF_MSG);
        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to complete wireup";

        do {
            nixl_status_t ret = getAgent(to_agent).getNotifs(notifs);
            ASSERT_EQ(ret, NIXL_SUCCESS) << "Failed to get notifications during wireup";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (notifs.size() == 0);
    }

    void exchangeMD(size_t from_agent, size_t to_agent)
    {
        // Connect the existing agents and exchange metadata
        for (size_t i = 0; i < agents.size(); i++) {
            nixl_blob_t md;
            nixl_status_t status = agents[i]->getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);

            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                std::string remote_agent_name;
                status = agents[j]->loadRemoteMD(md, remote_agent_name);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_EQ(remote_agent_name, getAgentName(i));
            }
        }

        completeWireup(from_agent, to_agent);
    }

    void invalidateMD()
    {
        // Disconnect the agents and invalidate remote metadata
        for (size_t i = 0; i < agents.size(); i++) {
            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                nixl_status_t status = agents[j]->invalidateRemoteMD(
                        getAgentName(i));
                ASSERT_EQ(status, NIXL_SUCCESS);
            }
        }
    }

    void createRegisteredMem(nixlAgent& agent,
                             size_t size, size_t count,
                             nixl_mem_t mem_type,
                             std::vector<MemBuffer>& out)
    {
        while (count-- != 0) {
            out.emplace_back(size, mem_type);
        }

        registerMem(agent, out, mem_type);
    }

    template<nixl_gpu_xfer_coordination_level_t level>
    void doTransfer(nixlAgent &from, const std::string &from_name,
                    nixlAgent &to, const std::string &to_name, size_t size,
                    size_t count, size_t num_threads,
                    nixl_mem_t src_mem_type,
                    std::optional<std::vector<MemBuffer>> src_buffers,
                    nixl_mem_t dst_mem_type,
                    std::optional<std::vector<MemBuffer>> dst_buffers,
                    std::optional<std::vector<MemBuffer>> signal_buffers)
    {
        nixl_opt_args_t extra_params = {};
        extra_params.hasNotif = true;
        extra_params.notifMsg = NOTIF_MSG;
        unsigned long long start_time_cpu = 0;
        unsigned long long end_time_cpu = 0;
        unsigned long long *start_time_ptr = nullptr;
        unsigned long long *end_time_ptr = nullptr;
        nixl_status_t status;
        uint64_t signal_inc;

        // TODO define one signal buffer
        if (signal_buffers.has_value()) {
            Logger() << "Setting signal buffer address " << static_cast<uintptr_t>(signal_buffers.value()[0]) << " and device id " << DEV_ID;
            extra_params.signal_addr = static_cast<uintptr_t>(signal_buffers.value()[0]);
            extra_params.signal_dev_id = DEV_ID;
            if (!src_buffers.has_value() && !dst_buffers.has_value()) {
                extra_params.backends.push_back(backend_handles[0]);
            }
            signal_inc = 1;
        } else {
            // Explicitly ensure signal fields are zero
            extra_params.signal_addr = 0;
            extra_params.signal_dev_id = 0;
            signal_inc = 0;
        }

        nixlXferReqH *xfer_req = nullptr;
        if (src_buffers.has_value() && dst_buffers.has_value()) {
            Logger() << "Creating xfer request with source and destination buffers";
            status = from.createXferReq(
                    NIXL_WRITE,
                    makeDescList<nixlBasicDesc>(src_buffers.value(), src_mem_type),
                    makeDescList<nixlBasicDesc>(dst_buffers.value(), dst_mem_type), to_name,
                    xfer_req, &extra_params);
        } else if (signal_buffers.has_value()) {
            Logger() << "Creating signal xfer request";
            status = from.createSignalXferReq(to_name, xfer_req, &extra_params);
        } else {
            FAIL() << "Either source/destination buffers or signal buffer must be provided";
        }

        ASSERT_EQ(status, NIXL_SUCCESS) << "Failed to create xfer request " << nixlEnumStrings::statusStr(status);
        EXPECT_NE(xfer_req, nullptr);
        Logger() << "Prep xfer request";

        nixlGpuXferReqH *gpu_req_hndl = from.exportXferReqtoGPU(xfer_req);
        ASSERT_NE(gpu_req_hndl, nullptr) << "Failed to export xfer request to GPU";
        Logger() << "Exported gpu request: " << static_cast<void*>(gpu_req_hndl);

        // Set default batch execution parameters
        const size_t num_iters = 10;

        initTiming(&start_time_ptr, &end_time_ptr);

        status = GdakiPostGpuXferReqwKernel<level>(num_threads, gpu_req_hndl, num_iters, signal_inc, start_time_ptr, end_time_ptr);
        ASSERT_TRUE(status == NIXL_SUCCESS);

        if (signal_buffers.has_value()) {
            getTiming(start_time_ptr, end_time_ptr, start_time_cpu, end_time_cpu);
            logResults(size, count, num_iters, start_time_cpu, end_time_cpu);
        }

        cudaFree(start_time_ptr);
        cudaFree(end_time_ptr);

        status = from.releaseXferReqtoGPU(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        status = from.releaseXferReq(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        invalidateMD();
    }

    template<size_t indcnt, nixl_gpu_xfer_coordination_level_t level, bool is_null_status = false>
    void doPartialTransfer(nixlAgent &from, const std::string &from_name,
                           nixlAgent &to, const std::string &to_name, size_t size,
                           size_t count, size_t num_threads,
                           nixl_mem_t src_mem_type,
                           std::optional<std::vector<MemBuffer>> src_buffers,
                           nixl_mem_t dst_mem_type,
                           std::optional<std::vector<MemBuffer>> dst_buffers,
                           std::optional<std::vector<MemBuffer>> signal_buffers)
    {
        std::vector<std::thread> threads;
        nixl_opt_args_t extra_params = {};
        extra_params.hasNotif = true;
        extra_params.notifMsg = NOTIF_MSG;
        unsigned long long start_time_cpu = 0;
        unsigned long long end_time_cpu = 0;
        unsigned long long *start_time_ptr = nullptr;
        unsigned long long *end_time_ptr = nullptr;
        nixl_status_t status;
        nixlXferReqH *xfer_req = nullptr;
        uint64_t signal_inc;

        // Only set signal buffer if provided
        if (signal_buffers.has_value()) {
            extra_params.signal_addr = static_cast<uintptr_t>(signal_buffers.value()[0]);
            extra_params.signal_dev_id = DEV_ID;
            if (!src_buffers.has_value() && !dst_buffers.has_value()) {
                extra_params.backends.push_back(backend_handles[0]);
            }
            signal_inc = 1;
        } else {
            // Explicitly set signal_addr to 0 when no signal is used
            extra_params.signal_addr = 0;
            extra_params.signal_dev_id = 0;
            signal_inc = 0;
        }

        bool has_iovs = src_buffers.has_value() && dst_buffers.has_value();

        if (src_buffers.has_value() && dst_buffers.has_value()) {
            Logger() << "Creating xfer request with source and destination buffers";
            status = from.createXferReq(
                    NIXL_WRITE,
                    makeDescList<nixlBasicDesc>(src_buffers.value(), src_mem_type),
                    makeDescList<nixlBasicDesc>(dst_buffers.value(), dst_mem_type), to_name,
                    xfer_req, &extra_params);
        } else {
            Logger() << "Creating signal xfer request with signal buffer";
            status = from.createSignalXferReq(to_name, xfer_req, &extra_params);
        }

        ASSERT_EQ(status, NIXL_SUCCESS);
        EXPECT_NE(xfer_req, nullptr);
        Logger() << "Prep partial xfer request";

        nixlGpuXferReqH *gpu_req_hndl = from.exportXferReqtoGPU(xfer_req);
        ASSERT_NE(gpu_req_hndl, nullptr);
        Logger() << "Exported gpu request for partial transfer: " << static_cast<void*>(gpu_req_hndl);

        // Set default batch execution parameters
        const size_t num_iters = 10;

        initTiming(&start_time_ptr, &end_time_ptr);

        status = GdakiPostGpuXferReqPartialwKernel<indcnt, level, is_null_status>(num_threads, gpu_req_hndl, num_iters, signal_inc, start_time_ptr, end_time_ptr, has_iovs);
        ASSERT_TRUE(status == NIXL_SUCCESS);

        if (signal_buffers.has_value()) {
            getTiming(start_time_ptr, end_time_ptr, start_time_cpu, end_time_cpu);
            size_t total_partial_size = 0;
            for (size_t i = 0; i < indcnt; ++i) {
                total_partial_size += ((1024 * 16) / 2) + (i * 512);
            }
            logResults(size, indcnt, num_iters, start_time_cpu, end_time_cpu);
        }

        cudaFree(start_time_ptr);
        cudaFree(end_time_ptr);

        status = from.releaseXferReqtoGPU(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        status = from.releaseXferReq(xfer_req);
        EXPECT_EQ(status, NIXL_SUCCESS);

        invalidateMD();
    }

    nixlAgent &getAgent(size_t idx)
    {
        return *agents[idx];
    }

    std::string getAgentName(size_t idx)
    {
        return absl::StrFormat("agent_%d", idx);
    }

    // Helper function to dispatch doTransfer based on coordination level
   void dispatch_doTransfer(nixl_gpu_xfer_coordination_level_t level,
                            nixlAgent &from, const std::string &from_name,
                            nixlAgent &to, const std::string &to_name, size_t size,
                            size_t count, size_t num_threads,
                            nixl_mem_t src_mem_type,
                            std::optional<std::vector<MemBuffer>> src_buffers,
                            nixl_mem_t dst_mem_type,
                            std::optional<std::vector<MemBuffer>> dst_buffers,
                            std::optional<std::vector<MemBuffer>> signal_buffers)
    {
        switch (level) {
            case NIXL_GPU_XFER_COORDINATION_BLOCK:
                doTransfer<NIXL_GPU_XFER_COORDINATION_BLOCK>(
                        from, from_name, to, to_name, size, count, num_threads,
                        src_mem_type, src_buffers, dst_mem_type, dst_buffers,
                        signal_buffers);
                break;
            case NIXL_GPU_XFER_COORDINATION_WARP:
            case NIXL_GPU_XFER_COORDINATION_THREAD:
                FAIL() << "Coordination level " << getCoordinationLevelName()
                    << " is not implemented yet";
            // case NIXL_GPU_XFER_COORDINATION_GRID:
                break;
            default:
                FAIL() << "Unknown coordination level: " << level;
        }
    }

    // Helper function to dispatch doPartialTransfer based on coordination level
    template<size_t indcnt, bool is_null_status = false>
    void dispatch_doPartialTransfer(nixl_gpu_xfer_coordination_level_t level,
                                    nixlAgent &from, const std::string &from_name,
                                    nixlAgent &to, const std::string &to_name, size_t size,
                                    size_t count, size_t num_threads,
                                    nixl_mem_t src_mem_type,
                                    std::optional<std::vector<MemBuffer>> src_buffers,
                                    nixl_mem_t dst_mem_type,
                                    std::optional<std::vector<MemBuffer>> dst_buffers,
                                    std::optional<std::vector<MemBuffer>> signal_buffers)
    {
        switch (level) {
            case NIXL_GPU_XFER_COORDINATION_BLOCK:
                doPartialTransfer<indcnt, NIXL_GPU_XFER_COORDINATION_BLOCK, is_null_status>(
                        from, from_name, to, to_name, size, count, num_threads,
                        src_mem_type, src_buffers, dst_mem_type, dst_buffers,
                        signal_buffers);
                break;
            case NIXL_GPU_XFER_COORDINATION_WARP:
                doPartialTransfer<indcnt, NIXL_GPU_XFER_COORDINATION_WARP, is_null_status>(
                        from, from_name, to, to_name, size, count, num_threads,
                        src_mem_type, src_buffers, dst_mem_type, dst_buffers,
                        signal_buffers);
                break;
            case NIXL_GPU_XFER_COORDINATION_THREAD:
                doPartialTransfer<indcnt, NIXL_GPU_XFER_COORDINATION_THREAD, is_null_status>(
                        from, from_name, to, to_name, size, count, num_threads,
                        src_mem_type, src_buffers, dst_mem_type, dst_buffers,
                        signal_buffers);
                break;
            // case NIXL_GPU_XFER_COORDINATION_GRID:
            default:
                FAIL() << "Unknown coordination level: " << level;
        }
    }

protected:
    static constexpr size_t SENDER_AGENT = 0;
    static constexpr size_t RECEIVER_AGENT = 1;

private:
    static constexpr uint64_t DEV_ID = 0;

    std::vector<std::unique_ptr<nixlAgent>> agents;
    std::vector<nixlBackendH*> backend_handles;

    // Timing helper functions
    void initTiming(unsigned long long** start_time_ptr, unsigned long long** end_time_ptr) {
        cudaMalloc(start_time_ptr, sizeof(unsigned long long));
        cudaMalloc(end_time_ptr, sizeof(unsigned long long));
        cudaMemset(*start_time_ptr, 0, sizeof(unsigned long long));
        cudaMemset(*end_time_ptr, 0, sizeof(unsigned long long));
    }

    void getTiming(unsigned long long* start_time_ptr, unsigned long long* end_time_ptr,
                   unsigned long long& start_time_cpu, unsigned long long& end_time_cpu) {
        cudaMemcpy(&start_time_cpu, start_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(&end_time_cpu, end_time_ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }

    void logResults(size_t size, size_t count, size_t num_iters,
                   unsigned long long start_time_cpu, unsigned long long end_time_cpu) {
        auto total_time = NS_TO_SEC(end_time_cpu - start_time_cpu);
        double total_size = size * count * num_iters;
        auto bandwidth = total_size / total_time / (1024 * 1024);
        Logger() << "Average Results: " << size << "x" << count
                 << "x" << num_iters << "=" << total_size << " bytes in "
                 << total_time << " seconds " << "(" << bandwidth << " MB/s)";
    }
};

TEST_P(TestGdakiTransfer, fullGpuTransferIntraNode)
{
    if (!isFullTransferCoordinationLevelImplemented()) {
        GTEST_SKIP() << "Coordination level " << getCoordinationLevelName()
                     << " is not implemented yet";
    }

    std::vector<MemBuffer> src_buffers, dst_buffers;
    std::vector<MemBuffer> signal_buffers;
    constexpr size_t size = 16 * 1024;
    constexpr size_t count = 32;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;

    createRegisteredMem(getAgent(SENDER_AGENT), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), size, count, mem_type, dst_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), sizeof(uint64_t), 1, mem_type, signal_buffers);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    dispatch_doTransfer(getFullTransferCoordinationLevel(),
        getAgent(SENDER_AGENT), getAgentName(SENDER_AGENT),
        getAgent(RECEIVER_AGENT), getAgentName(RECEIVER_AGENT),
        size, count, num_threads, mem_type, src_buffers,
        mem_type, dst_buffers, signal_buffers);
}

TEST_P(TestGdakiTransfer, partialGpuTransferIntraNode)
{
    if (!isPartialTransferCoordinationLevelImplemented()) {
        GTEST_SKIP() << "Coordination level " << getCoordinationLevelName()
                     << " is not implemented yet";
    }

    std::vector<MemBuffer> src_buffers, dst_buffers;
    std::vector<MemBuffer> signal_buffers;
    constexpr size_t size = 16 * 1024;
    constexpr size_t count = 32;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;

    createRegisteredMem(getAgent(SENDER_AGENT), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), size, count, mem_type, dst_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), sizeof(uint64_t), 1, mem_type, signal_buffers);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    constexpr size_t indcnt = 8;

    dispatch_doPartialTransfer<indcnt>(getPartialTransferCoordinationLevel(),
        getAgent(SENDER_AGENT), getAgentName(SENDER_AGENT),
        getAgent(RECEIVER_AGENT), getAgentName(RECEIVER_AGENT),
        size, count, num_threads, mem_type, src_buffers,
        mem_type, dst_buffers, signal_buffers);
}

TEST_P(TestGdakiTransfer, partialGpuTransferIntraNodeWithNullStatus)
{
    if (!isPartialTransferCoordinationLevelImplemented()) {
        GTEST_SKIP() << "Coordination level " << getCoordinationLevelName()
                     << " is not implemented yet";
    }

    std::vector<MemBuffer> src_buffers, dst_buffers;
    std::vector<MemBuffer> signal_buffers;
    constexpr size_t size = 16 * 1024;
    constexpr size_t count = 32;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;

    createRegisteredMem(getAgent(SENDER_AGENT), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), size, count, mem_type, dst_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), sizeof(uint64_t), 1, mem_type, signal_buffers);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    constexpr size_t indcnt = 8;

    dispatch_doPartialTransfer<indcnt, true>(
        getPartialTransferCoordinationLevel(),
        getAgent(SENDER_AGENT), getAgentName(SENDER_AGENT),
        getAgent(RECEIVER_AGENT), getAgentName(RECEIVER_AGENT),
        size, count, num_threads, mem_type, src_buffers,
        mem_type, dst_buffers, signal_buffers);
}

TEST_P(TestGdakiTransfer, partialGpuTransferIntraNodeIovsWithoutSignal)
{
    if (!isPartialTransferCoordinationLevelImplemented()) {
        GTEST_SKIP() << "Coordination level " << getCoordinationLevelName()
                     << " is not implemented yet";
    }

    std::vector<MemBuffer> src_buffers, dst_buffers;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;
    constexpr size_t indcnt = 8; // Transfer 8 out of 32 buffers
    constexpr size_t size = 16 * 1024;
    constexpr size_t count = 32;

    createRegisteredMem(getAgent(SENDER_AGENT), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), size, count, mem_type, dst_buffers);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    dispatch_doPartialTransfer<indcnt>(getPartialTransferCoordinationLevel(),
        getAgent(SENDER_AGENT), getAgentName(SENDER_AGENT),
        getAgent(RECEIVER_AGENT), getAgentName(RECEIVER_AGENT),
        size, count, num_threads, mem_type, src_buffers,
        mem_type, dst_buffers, std::nullopt);
}

TEST_P(TestGdakiTransfer, partialGpuTransferIntraNodeIovsWithoutSignalWithNullStatus)
{
    if (!isPartialTransferCoordinationLevelImplemented()) {
        GTEST_SKIP() << "Coordination level " << getCoordinationLevelName()
                     << " is not implemented yet";
    }

    std::vector<MemBuffer> src_buffers, dst_buffers;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;
    constexpr size_t indcnt = 8; // Transfer 8 out of 32 buffers
    constexpr size_t size = 16 * 1024;
    constexpr size_t count = 32;

    createRegisteredMem(getAgent(SENDER_AGENT), size, count, mem_type, src_buffers);
    createRegisteredMem(getAgent(RECEIVER_AGENT), size, count, mem_type, dst_buffers);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    dispatch_doPartialTransfer<indcnt, true>(
        getPartialTransferCoordinationLevel(),
        getAgent(SENDER_AGENT), getAgentName(SENDER_AGENT),
        getAgent(RECEIVER_AGENT), getAgentName(RECEIVER_AGENT),
        size, count, num_threads, mem_type, src_buffers,
        mem_type, dst_buffers, std::nullopt);
}

TEST_P(TestGdakiTransfer, fullGpuTransferIntraNodeSignalWithoutIovs)
{
    if (!isFullTransferCoordinationLevelImplemented()) {
        GTEST_SKIP() << "Coordination level " << getCoordinationLevelName()
                     << " is not implemented yet";
    }

    std::vector<MemBuffer> signal_buffers;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;

    createRegisteredMem(getAgent(RECEIVER_AGENT), sizeof(uint64_t), 1, mem_type,
                        signal_buffers);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    dispatch_doTransfer(getFullTransferCoordinationLevel(),
               getAgent(SENDER_AGENT), getAgentName(SENDER_AGENT),
               getAgent(RECEIVER_AGENT), getAgentName(RECEIVER_AGENT),
               0, 0, num_threads, mem_type, std::nullopt,
               mem_type, std::nullopt, std::make_optional(signal_buffers));
}

TEST_P(TestGdakiTransfer, partialGpuTransferIntraNodeSignalWithoutIovs)
{
    if (!isPartialTransferCoordinationLevelImplemented()) {
        GTEST_SKIP() << "Coordination level " << getCoordinationLevelName()
                     << " is not implemented yet";
    }

    std::vector<MemBuffer> signal_buffers;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;

    createRegisteredMem(getAgent(RECEIVER_AGENT), sizeof(uint64_t), 1, mem_type,
                        signal_buffers);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    dispatch_doPartialTransfer<1>(getPartialTransferCoordinationLevel(),
               getAgent(SENDER_AGENT), getAgentName(SENDER_AGENT),
               getAgent(RECEIVER_AGENT), getAgentName(RECEIVER_AGENT),
               0, 0, num_threads, mem_type, std::nullopt,
               mem_type, std::nullopt, std::make_optional(signal_buffers));
}

TEST_P(TestGdakiTransfer, partialGpuTransferIntraNodeSignalWithoutIovsWithNullStatus)
{
    if (!isPartialTransferCoordinationLevelImplemented()) {
        GTEST_SKIP() << "Coordination level " << getCoordinationLevelName()
                     << " is not implemented yet";
    }

    std::vector<MemBuffer> signal_buffers;
    nixl_mem_t mem_type = VRAM_SEG;
    size_t num_threads = 32;

    createRegisteredMem(getAgent(RECEIVER_AGENT), sizeof(uint64_t), 1, mem_type,
                        signal_buffers);

    exchangeMD(SENDER_AGENT, RECEIVER_AGENT);

    dispatch_doPartialTransfer<1, true>(getPartialTransferCoordinationLevel(),
               getAgent(SENDER_AGENT), getAgentName(SENDER_AGENT),
               getAgent(RECEIVER_AGENT), getAgentName(RECEIVER_AGENT),
               0, 0, num_threads, mem_type, std::nullopt,
               mem_type, std::nullopt, std::make_optional(signal_buffers));
}


// Define coordination levels to test
static const std::vector<TestGdakiTransferParam> coordination_levels = {
    {NIXL_GPU_XFER_COORDINATION_BLOCK,  "BLOCK",  true,  true},   // Fully implemented
    {NIXL_GPU_XFER_COORDINATION_WARP,   "WARP",   false, true},   // Implemented for partial transfers
    {NIXL_GPU_XFER_COORDINATION_THREAD, "THREAD", false, true},  // Implemented for partial transfers
    // {NIXL_GPU_XFER_COORDINATION_GRID,   "GRID",   false, false}   // Not implemented
};

INSTANTIATE_TEST_SUITE_P(
    ucx,
    TestGdakiTransfer,
    testing::Combine(
        testing::Values("UCX"),
        testing::ValuesIn(coordination_levels)
    ),
    [](const testing::TestParamInfo<std::tuple<std::string, TestGdakiTransferParam>>& info) {
        return std::get<0>(info.param) + "_" + std::get<1>(info.param).name;
    }
);

} // namespace gtest
