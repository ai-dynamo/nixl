/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <gpu/nixl_device.cuh>
#include <nixl.h>

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <netinet/in.h>
#include <optional>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) || !NIXL_ENABLE_GPUNETIO_DEVICE_API
#error "Build this test with NIXL_ENABLE_GPUNETIO_DEVICE_API=1"
#endif

#if !defined(NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31) || !NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31
#error "Build this test with NIXL_GPUNETIO_DEVICE_API_LEGACY_DOCA31=1"
#endif

namespace {

constexpr uint64_t kArenaBytes = 40ull * 1024 * 1024;
constexpr uint64_t kDescriptorPrefix = 256;
constexpr uint64_t kDescriptorSuffix = 256;
constexpr uint64_t kGuardBytes = 64;
constexpr uint8_t kSourceGuard = 0x5a;
constexpr uint8_t kDestinationGuard = 0xa5;
constexpr uint64_t kWrapRequests = 8193;
constexpr uint64_t kWrapPayloadBytes = 4096;
constexpr uint64_t kWrapSlotBytes = 4352;
constexpr auto kWaitTimeout = std::chrono::seconds(45);
constexpr auto kPollTimeout = std::chrono::seconds(5);
constexpr size_t kStatusModeTagOffset = sizeof(nixlGpuXferStatusH) - sizeof(uint32_t);

struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t size) : bytes(size) {
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
            ptr = nullptr;
            bytes = 0;
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &
    operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

struct DeviceResult {
    int32_t submission_status = NIXL_ERR_UNKNOWN;
    int32_t completion_status = NIXL_ERR_UNKNOWN;
    int32_t old_status = NIXL_ERR_UNKNOWN;
    uint64_t expected = 0;
    uint64_t accepted = 0;
    uint64_t completed = 0;
    uint64_t first_failure = std::numeric_limits<uint64_t>::max();
    uint64_t poll_count = 0;
    uint32_t timed_out = 0;
    uint32_t status_untouched = 0;
};

struct PutSpec {
    nixlMemViewElem src;
    nixlMemViewElem dst;
    uint64_t bytes;
    uint64_t seed;
    uint64_t count;
};

struct InvalidSpec {
    nixlMemViewElem src;
    nixlMemViewElem dst;
    uint64_t bytes;
    uint64_t flags;
    unsigned channel;
    nixl_status_t expected_status;
    uint32_t kind;
};

struct RunCase {
    const char *name;
    uint64_t bytes;
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t seed;
};

__device__ uint64_t
deviceNow() {
    return clock64();
}

template<nixl_gpu_level_t level>
__global__ void
putKernel(PutSpec spec, DeviceResult *result, uint64_t timeout_cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    result->expected = spec.count;
    nixlGpuXferStatusH status{};
    nixlGpuXferStatusH old_status{};
    bool have_old_status = false;

    for (uint64_t i = 0; i < spec.count; ++i) {
        const nixlMemViewElem src{
            spec.src.mvh, spec.src.index, spec.src.offset + i * kWrapSlotBytes};
        const nixlMemViewElem dst{
            spec.dst.mvh, spec.dst.index, spec.dst.offset + i * kWrapSlotBytes};

        status = nixlGpuXferStatusH{};
        const auto submitted = nixlPut<level>(src, dst, spec.bytes, 0, 0, &status);
        if (i == 0) {
            result->submission_status = submitted;
        }
        if (submitted != NIXL_IN_PROG) {
            result->first_failure = i;
            result->completion_status = submitted;
            return;
        }
        ++result->accepted;

        const uint64_t deadline = deviceNow() + timeout_cycles;
        nixl_status_t completed = NIXL_IN_PROG;
        do {
            completed = nixlGpuGetXferStatus<level>(status);
            ++result->poll_count;
            if (completed == NIXL_IN_PROG && deviceNow() >= deadline) {
                result->timed_out = 1;
                result->first_failure = i;
                result->completion_status = NIXL_IN_PROG;
                return;
            }
        } while (completed == NIXL_IN_PROG);

        if (completed != NIXL_SUCCESS) {
            result->first_failure = i;
            result->completion_status = completed;
            return;
        }
        ++result->completed;
        result->completion_status = completed;

        if (!have_old_status) {
            old_status = status;
            have_old_status = true;
        }
    }

    if (have_old_status) {
        result->old_status = nixlGpuGetXferStatus<level>(old_status);
    }
}

__device__ bool
sameStatus(const nixlGpuXferStatusH &lhs, const nixlGpuXferStatusH &rhs) {
    const auto *a = reinterpret_cast<const unsigned char *>(&lhs);
    const auto *b = reinterpret_cast<const unsigned char *>(&rhs);
    for (size_t i = 0; i < sizeof(nixlGpuXferStatusH); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

__device__ void
initRejectedStatus(nixlGpuXferStatusH &status) {
    auto *bytes = reinterpret_cast<unsigned char *>(&status);
    for (size_t i = 0; i < sizeof(nixlGpuXferStatusH); ++i) {
        bytes[i] = 0xa5;
    }
    for (size_t i = kStatusModeTagOffset; i < sizeof(nixlGpuXferStatusH); ++i) {
        bytes[i] = 0;
    }
}

__global__ void
busyKernel(PutSpec spec, DeviceResult *result, uint64_t timeout_cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    nixlGpuXferStatusH first{};
    nixlGpuXferStatusH second_before{};
    nixlGpuXferStatusH second{};
    initRejectedStatus(second_before);
    second = second_before;
    const auto first_submit =
        nixlPut<nixl_gpu_level_t::THREAD>(spec.src, spec.dst, spec.bytes, 0, 0, &first);
    const auto second_submit =
        nixlPut<nixl_gpu_level_t::THREAD>(spec.src, spec.dst, spec.bytes, 0, 0, &second);
    result->submission_status = first_submit;
    result->completion_status = second_submit;
    result->status_untouched = sameStatus(second_before, second) ? 1 : 0;
    if (first_submit == NIXL_IN_PROG) {
        ++result->accepted;
    }

    const uint64_t deadline = deviceNow() + timeout_cycles;
    nixl_status_t status = NIXL_IN_PROG;
    while (status == NIXL_IN_PROG) {
        status = nixlGpuGetXferStatus<nixl_gpu_level_t::THREAD>(first);
        ++result->poll_count;
        if (status == NIXL_IN_PROG && deviceNow() >= deadline) {
            result->timed_out = 1;
            return;
        }
    }
    result->old_status = status;
    result->completed = status == NIXL_SUCCESS ? 1 : 0;
}

__global__ void
invalidKernel(InvalidSpec spec, DeviceResult *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    nixlGpuXferStatusH before{};
    nixlGpuXferStatusH after{};
    initRejectedStatus(before);
    after = before;
    const auto returned = nixlPut<nixl_gpu_level_t::THREAD>(
        spec.src, spec.dst, spec.bytes, spec.channel, spec.flags, &after);
    result->submission_status = returned;
    result->status_untouched = sameStatus(before, after) ? 1 : 0;
    result->expected = static_cast<int32_t>(spec.expected_status);
}

__global__ void
nullStatusKernel(PutSpec spec, DeviceResult *result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    result->submission_status =
        nixlPut<nixl_gpu_level_t::THREAD>(spec.src, spec.dst, spec.bytes, 0, 0, nullptr);
    result->status_untouched = 1;
}

std::optional<std::string>
env(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::string(value);
}

uint64_t
envUint(const char *name, uint64_t fallback) {
    const auto value = env(name);
    if (!value) {
        return fallback;
    }
    size_t consumed = 0;
    const uint64_t parsed = std::stoull(*value, &consumed, 0);
    if (consumed != value->size()) {
        throw std::invalid_argument(std::string(name) + " is not an integer");
    }
    return parsed;
}

class Control {
public:
    Control(const std::string &role, const std::string &target_ipv4, uint16_t port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) {
            throw std::runtime_error("control socket failed");
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(port);
        if (role == "receiver") {
            int reuse = 1;
            ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
            address.sin_addr.s_addr = htonl(INADDR_ANY);
            if (::bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
                ::listen(fd_, 1) != 0) {
                throw std::runtime_error("control bind/listen failed");
            }
            pollfd listener{fd_, POLLIN, 0};
            const int timeout_ms = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(kWaitTimeout).count());
            if (::poll(&listener, 1, timeout_ms) <= 0) {
                throw std::runtime_error("control accept timeout");
            }
            const int accepted = ::accept(fd_, nullptr, nullptr);
            ::close(fd_);
            fd_ = accepted;
            if (fd_ < 0) {
                throw std::runtime_error("control accept failed");
            }
        } else {
            if (::inet_pton(AF_INET, target_ipv4.c_str(), &address.sin_addr) != 1) {
                throw std::runtime_error("GPUNETIO_DEVICE_API_TARGET_IPV4 must be numeric IPv4");
            }
            const auto deadline = std::chrono::steady_clock::now() + kWaitTimeout;
            while (::connect(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("control connect timeout");
                }
                ::close(fd_);
                fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
                if (fd_ < 0) {
                    throw std::runtime_error("control retry socket failed");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        timeval timeout{static_cast<long>(kWaitTimeout.count()), 0};
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    }

    ~Control() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    Control(const Control &) = delete;
    Control &
    operator=(const Control &) = delete;

    void
    send(char token) const {
        if (::send(fd_, &token, 1, MSG_NOSIGNAL) != 1) {
            throw std::runtime_error("control send failed");
        }
    }

    void
    expect(char expected) const {
        char actual = 0;
        if (::recv(fd_, &actual, 1, MSG_WAITALL) != 1 || actual != expected) {
            throw std::runtime_error("control token/order mismatch");
        }
    }

private:
    int fd_ = -1;
};

void
writeAll(int fd, const char *data, size_t size) {
    while (size != 0) {
        const ssize_t written = ::write(fd, data, size);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            throw std::runtime_error("write failed: " + std::string(std::strerror(errno)));
        }
        data += written;
        size -= static_cast<size_t>(written);
    }
}

void
publishFile(const std::filesystem::path &path, const std::string &contents) {
    const auto tmp = path.string() + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        throw std::runtime_error("open temporary file failed: " + tmp + ": " +
                                 std::strerror(errno));
    }
    try {
        writeAll(fd, contents.data(), contents.size());
        if (::fsync(fd) != 0 || ::close(fd) != 0) {
            throw std::runtime_error("fsync/close failed for " + tmp);
        }
        if (::rename(tmp.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("atomic rename failed for " + path.string());
        }
        const int dir_fd = ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
        if (dir_fd < 0 || ::fsync(dir_fd) != 0 || ::close(dir_fd) != 0) {
            throw std::runtime_error("directory fsync failed for " + path.parent_path().string());
        }
    }
    catch (...) {
        ::close(fd);
        ::unlink(tmp.c_str());
        throw;
    }
}

std::string
readFile(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

uint8_t
patternByte(uint64_t seed, uint64_t offset) {
    uint64_t value = seed ^ (offset + 0x9e3779b97f4a7c15ull);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return static_cast<uint8_t>(value);
}

void
fillPattern(std::vector<uint8_t> &buffer, size_t offset, size_t size, uint64_t seed) {
    for (size_t i = 0; i < size; ++i) {
        buffer[offset + i] = patternByte(seed, i);
    }
}

DeviceResult
copyResult(DeviceBuffer &device_result) {
    DeviceResult result{};
    const auto copied =
        cudaMemcpy(&result, device_result.ptr, sizeof(result), cudaMemcpyDeviceToHost);
    if (copied != cudaSuccess || result.timed_out || result.accepted > result.completed) {
        // Do not run normal MR/buffer teardown after an unretired operation.
        // The supervising process watchdog treats this nonzero exit as failure.
        std::fputs("FAIL: native operation not safely retired; terminating endpoint\n", stderr);
        std::fflush(stderr);
        std::_Exit(2);
    }
    return result;
}

uint64_t
pollCycles() {
    cudaDeviceProp prop{};
    int device = 0;
    EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
    EXPECT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
    return static_cast<uint64_t>(prop.clockRate) *
        static_cast<uint64_t>(
               std::chrono::duration_cast<std::chrono::milliseconds>(kPollTimeout).count());
}

bool
flushGpuDirectWrites(std::string &error) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11030
    CUcontext context = nullptr;
    if (cuInit(0) != CUDA_SUCCESS || cuCtxGetCurrent(&context) != CUDA_SUCCESS ||
        context == nullptr) {
        error = "CUDA driver context unavailable for GPUDirect RDMA flush";
        return false;
    }
    const CUresult result =
        cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                   CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
    if (result != CUDA_SUCCESS) {
        const char *name = nullptr;
        cuGetErrorName(result, &name);
        error = "cuFlushGPUDirectRDMAWrites failed";
        if (name != nullptr) {
            error += ": ";
            error += name;
        }
        return false;
    }
    return true;
#else
    error = "CUDA headers do not expose cuFlushGPUDirectRDMAWrites";
    return false;
#endif
}

class GpunetioDeviceApiTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        role_ = env("GPUNETIO_DEVICE_API_ROLE").value_or("");
        coord_dir_ = env("GPUNETIO_DEVICE_API_COORD_DIR").value_or("");
        run_id_ = env("GPUNETIO_DEVICE_API_RUN_ID").value_or("");
        test_id_ = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        if (role_ != "sender" && role_ != "receiver") {
            GTEST_SKIP() << "Set GPUNETIO_DEVICE_API_ROLE=sender or receiver";
        }
        if (coord_dir_.empty() || run_id_.empty()) {
            GTEST_SKIP() << "Set GPUNETIO_DEVICE_API_COORD_DIR and GPUNETIO_DEVICE_API_RUN_ID";
        }
        control_target_ipv4_ = env("GPUNETIO_DEVICE_API_TARGET_IPV4").value_or("");
        control_port_ = envUint("GPUNETIO_DEVICE_API_CONTROL_PORT", 0);
        if (control_port_ == 0 || control_port_ > 65535) {
            GTEST_SKIP() << "Set GPUNETIO_DEVICE_API_CONTROL_PORT to 1..65535";
        }
        if (role_ == "sender" && control_target_ipv4_.empty()) {
            GTEST_SKIP() << "Set GPUNETIO_DEVICE_API_TARGET_IPV4 for sender";
        }

        const auto directory = std::filesystem::path(coord_dir_);
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        ASSERT_FALSE(error) << "cannot create coordination directory: " << error.message();
        ASSERT_EQ(cudaSetDevice(static_cast<int>(envUint("GPUNETIO_DEVICE_API_GPU", 0))),
                  cudaSuccess);

        buffer_ = std::make_unique<DeviceBuffer>(kArenaBytes);
        ASSERT_NE(buffer_->ptr, nullptr) << "cudaMalloc failed";
        ASSERT_EQ(cudaMemset(buffer_->ptr, kDestinationGuard, kArenaBytes), cudaSuccess);

        nixlAgentConfig config;
        config.useProgThread = true;
        config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        config.pthrDelay = 100000;
        agent_ = std::make_unique<nixlAgent>(agentName(), config);

        nixl_b_params_t params;
        params["native_device_api"] = "true";
        params["gpu_devices"] = std::to_string(envUint("GPUNETIO_DEVICE_API_GPU", 0));
        setParam(params, "network_devices", "GPUNETIO_DEVICE_API_RDMA");
        setParam(params, "oob_interface", "GPUNETIO_DEVICE_API_OOB_INTERFACE");
        setParam(params, "gid_index", "GPUNETIO_DEVICE_API_GID_INDEX");
        setParam(params, "oob_port", "GPUNETIO_DEVICE_API_OOB_PORT");
        ASSERT_EQ(agent_->createBackend("GPUNETIO", params, backend_), NIXL_SUCCESS);
        ASSERT_NE(backend_, nullptr);

        nixl_opt_args_t backend_hint;
        backend_hint.backends.push_back(backend_);
        registration_address_ = reinterpret_cast<uintptr_t>(buffer_->ptr);
        registration_length_ = kArenaBytes;
        local_address_ = registration_address_ + kDescriptorPrefix;
        local_length_ = kArenaBytes - kDescriptorPrefix - kDescriptorSuffix;
        nixl_reg_dlist_t registration(VRAM_SEG);
        registration.addDesc({registration_address_, registration_length_, 0, {}});
        ASSERT_EQ(agent_->registerMem(registration, &backend_hint), NIXL_SUCCESS);

        nixl_blob_t metadata;
        ASSERT_EQ(agent_->getLocalMD(metadata), NIXL_SUCCESS);
        publishFile(path(".md"), metadata);
        publishFile(path(".desc"),
                    std::to_string(local_address_) + " " + std::to_string(local_length_));

        control_ = std::make_unique<Control>(
            role_, control_target_ipv4_, static_cast<uint16_t>(control_port_));
        control_->send('M');
        control_->expect('M');

        // The receiver is intentionally metadata-only.  Only the sender imports
        // the receiver metadata, creates the OOB data connection, and prepares
        // the public Device API views used by the PUT path.
        if (role_ == "receiver") {
            control_->expect('S');
            return;
        }

        std::string peer_name;
        ASSERT_EQ(agent_->loadRemoteMD(readFile(peerPath(".md")), peer_name), NIXL_SUCCESS);
        ASSERT_EQ(peer_name, peerAgentName());
        ASSERT_EQ(agent_->makeConnection(peer_name, &backend_hint), NIXL_SUCCESS);

        const auto peer_desc = parseDescriptor(readFile(peerPath(".desc")));
        ASSERT_TRUE(peer_desc.has_value()) << "invalid peer descriptor record";
        nixl_local_dlist_t local(VRAM_SEG);
        local.addDesc({local_address_, local_length_, 0});
        nixl_remote_dlist_t remote(VRAM_SEG);
        remote.addDesc({peer_desc->first, peer_desc->second, 0, peer_name});
        local_view_ = nullptr;
        remote_view_ = nullptr;
        ASSERT_EQ(agent_->prepMemView(local, local_view_, &backend_hint), NIXL_SUCCESS);
        ASSERT_EQ(agent_->prepMemView(remote, remote_view_, &backend_hint), NIXL_SUCCESS);
        ASSERT_NE(local_view_, nullptr);
        ASSERT_NE(remote_view_, nullptr);
        control_->send('S');
    }

    void
    TearDown() override {
        if (agent_ != nullptr) {
            if (local_view_ != nullptr) {
                agent_->releaseMemView(local_view_);
            }
            if (remote_view_ != nullptr) {
                agent_->releaseMemView(remote_view_);
            }
            local_view_ = nullptr;
            remote_view_ = nullptr;
            agent_.reset();
        }
        control_.reset();
        buffer_.reset();
    }

    std::filesystem::path
    path(const char *suffix) const {
        return std::filesystem::path(coord_dir_) /
            (run_id_ + "." + test_id_ + "." + role_ + suffix);
    }

    std::filesystem::path
    peerPath(const char *suffix) const {
        const std::string peer = role_ == "sender" ? "receiver" : "sender";
        return std::filesystem::path(coord_dir_) / (run_id_ + "." + test_id_ + "." + peer + suffix);
    }

    std::string
    agentName() const {
        return "gpunetio_native_" + role_;
    }

    std::string
    peerAgentName() const {
        return std::string("gpunetio_native_") + (role_ == "sender" ? "receiver" : "sender");
    }

    static void
    setParam(nixl_b_params_t &params, const char *param, const char *variable) {
        if (const auto value = env(variable)) {
            params[param] = *value;
        }
    }

    static std::optional<std::pair<uintptr_t, size_t>>
    parseDescriptor(const std::string &contents) {
        std::istringstream input(contents);
        uint64_t address = 0;
        size_t length = 0;
        if (!(input >> address >> length)) {
            return std::nullopt;
        }
        return std::make_pair(static_cast<uintptr_t>(address), length);
    }

    DeviceResult
    launchPut(const PutSpec &spec, uint64_t count) {
        DeviceBuffer device_result(sizeof(DeviceResult));
        EXPECT_NE(device_result.ptr, nullptr);
        resetResult(device_result);
        PutSpec kernel_spec = spec;
        kernel_spec.count = count;
        putKernel<nixl_gpu_level_t::THREAD>
            <<<1, 1>>>(kernel_spec, static_cast<DeviceResult *>(device_result.ptr), pollCycles());
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        return copyResult(device_result);
    }

    DeviceResult
    launchBusy(const PutSpec &spec) {
        DeviceBuffer device_result(sizeof(DeviceResult));
        EXPECT_NE(device_result.ptr, nullptr);
        resetResult(device_result);
        busyKernel<<<1, 1>>>(spec, static_cast<DeviceResult *>(device_result.ptr), pollCycles());
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        return copyResult(device_result);
    }

    DeviceResult
    launchInvalid(const InvalidSpec &spec) {
        DeviceBuffer device_result(sizeof(DeviceResult));
        EXPECT_NE(device_result.ptr, nullptr);
        resetResult(device_result);
        invalidKernel<<<1, 1>>>(spec, static_cast<DeviceResult *>(device_result.ptr));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        return copyResult(device_result);
    }

    DeviceResult
    launchNullStatus(const PutSpec &spec) {
        DeviceBuffer device_result(sizeof(DeviceResult));
        EXPECT_NE(device_result.ptr, nullptr);
        resetResult(device_result);
        nullStatusKernel<<<1, 1>>>(spec, static_cast<DeviceResult *>(device_result.ptr));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        return copyResult(device_result);
    }

    void
    prepareSource(const RunCase &test_case, std::vector<uint8_t> &host) {
        host.assign(kArenaBytes, kSourceGuard);
        fillPattern(
            host, kDescriptorPrefix + test_case.src_offset, test_case.bytes, test_case.seed);
        ASSERT_EQ(cudaMemcpy(buffer_->ptr, host.data(), host.size(), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    void
    validateDestination(const RunCase &test_case) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint8_t> received(kArenaBytes);
        ASSERT_EQ(
            cudaMemcpy(received.data(), buffer_->ptr, received.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
        const uint64_t count = std::string(test_case.name) == "wrap" ? kWrapRequests : 1;
        for (uint64_t slot = 0; slot < count; ++slot) {
            const size_t offset = kDescriptorPrefix + test_case.dst_offset + slot * kWrapSlotBytes;
            for (size_t i = 0; i < kGuardBytes; ++i) {
                ASSERT_EQ(received[offset - kGuardBytes + i], kDestinationGuard);
            }
            const size_t trailing_guard =
                count > 1 ? kWrapSlotBytes - test_case.bytes : kGuardBytes;
            for (size_t i = 0; i < trailing_guard; ++i) {
                ASSERT_EQ(received[offset + test_case.bytes + i], kDestinationGuard);
            }
            const uint64_t seed = test_case.seed + slot;
            for (size_t i = 0; i < test_case.bytes; ++i) {
                ASSERT_EQ(received[offset + i], patternByte(seed, i))
                    << "slot " << slot << ", payload byte " << i;
            }
        }
    }

    void
    resetDestinationAndSendReady() {
        ASSERT_EQ(cudaMemset(buffer_->ptr, kDestinationGuard, kArenaBytes), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        control_->send('R');
    }

    void
    waitReady() {
        control_->expect('R');
    }

    void
    sendDataDone() {
        control_->send('D');
    }

    void
    waitDataDoneAndValidate(const RunCase &test_case) {
        control_->expect('D');
        std::string error;
        ASSERT_TRUE(flushGpuDirectWrites(error)) << error;
        validateDestination(test_case);
        control_->send('V');
    }

    void
    waitValidated() {
        control_->expect('V');
    }

    std::string role_;
    std::string coord_dir_;
    std::string run_id_;
    std::string test_id_;
    std::string control_target_ipv4_;
    uint64_t control_port_ = 0;
    std::unique_ptr<DeviceBuffer> buffer_;
    std::unique_ptr<nixlAgent> agent_;
    std::unique_ptr<Control> control_;
    nixlBackendH *backend_ = nullptr;
    nixlMemViewH local_view_ = nullptr;
    nixlMemViewH remote_view_ = nullptr;
    uintptr_t local_address_ = 0;
    size_t local_length_ = 0;
    uintptr_t registration_address_ = 0;
    size_t registration_length_ = 0;

private:
    static void
    resetResult(DeviceBuffer &device_result) {
        DeviceResult initial{};
        ASSERT_EQ(cudaMemcpy(device_result.ptr, &initial, sizeof(initial), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
};

TEST_F(GpunetioDeviceApiTest, BoundedNormalPutAndValidation) {
    const std::array<RunCase, 4> cases{{
        {"4k", 4 * 1024, 1024, 2048, 0x4100},
        {"64k", 64 * 1024, 8192, 16384, 0x6400},
        {"1m", 1024 * 1024, 131072, 262144, 0x100000},
        {"wrap", kWrapPayloadBytes, 512, 1024, 0x8193},
    }};
    std::vector<uint8_t> source(kArenaBytes, kSourceGuard);

    if (role_ == "sender") {
        for (const auto &test_case : cases) {
            waitReady();
            if (std::string(test_case.name) == "wrap") {
                source.assign(kArenaBytes, kSourceGuard);
                for (uint64_t i = 0; i < kWrapRequests; ++i) {
                    fillPattern(source,
                                kDescriptorPrefix + test_case.src_offset + i * kWrapSlotBytes,
                                test_case.bytes,
                                test_case.seed + i);
                }
                ASSERT_EQ(
                    cudaMemcpy(buffer_->ptr, source.data(), source.size(), cudaMemcpyHostToDevice),
                    cudaSuccess);
                PutSpec spec{{local_view_, 0, test_case.src_offset},
                             {remote_view_, 0, test_case.dst_offset},
                             test_case.bytes,
                             test_case.seed,
                             kWrapRequests};
                const DeviceResult result = launchPut(spec, kWrapRequests);
                ASSERT_EQ(result.submission_status, NIXL_IN_PROG);
                ASSERT_EQ(result.completion_status, NIXL_SUCCESS);
                ASSERT_EQ(result.old_status, NIXL_SUCCESS);
                ASSERT_EQ(result.expected, kWrapRequests);
                ASSERT_EQ(result.accepted, kWrapRequests);
                ASSERT_EQ(result.completed, kWrapRequests);
                ASSERT_EQ(result.first_failure, std::numeric_limits<uint64_t>::max());
                ASSERT_EQ(result.timed_out, 0u);
            } else {
                prepareSource(test_case, source);
                PutSpec spec{{local_view_, 0, test_case.src_offset},
                             {remote_view_, 0, test_case.dst_offset},
                             test_case.bytes,
                             test_case.seed,
                             1};
                const DeviceResult result = launchPut(spec, 1);
                ASSERT_EQ(result.submission_status, NIXL_IN_PROG);
                ASSERT_EQ(result.completion_status, NIXL_SUCCESS);
                ASSERT_EQ(result.old_status, NIXL_SUCCESS);
                ASSERT_EQ(result.expected, 1u);
                ASSERT_EQ(result.accepted, 1u);
                ASSERT_EQ(result.completed, 1u);
                ASSERT_EQ(result.first_failure, std::numeric_limits<uint64_t>::max());
                ASSERT_EQ(result.timed_out, 0u);
            }
            sendDataDone();
            waitValidated();
        }
    } else {
        for (const auto &test_case : cases) {
            resetDestinationAndSendReady();
            waitDataDoneAndValidate(test_case);
        }
    }
}

TEST_F(GpunetioDeviceApiTest, BusyAndRejectedArgumentsDoNotMutateStatus) {
    const RunCase busy_case{"busy", 4096, 4 * 1024 * 1024, 4 * 1024 * 1024, 0xbeef};
    if (role_ == "receiver") {
        resetDestinationAndSendReady();
        waitDataDoneAndValidate(busy_case);
        return;
    }

    std::vector<uint8_t> source;
    waitReady();
    prepareSource(busy_case, source);
    const PutSpec valid{{local_view_, 0, busy_case.src_offset},
                        {remote_view_, 0, busy_case.dst_offset},
                        busy_case.bytes,
                        busy_case.seed,
                        1};
    const DeviceResult busy = launchBusy(valid);
    ASSERT_EQ(busy.submission_status, NIXL_IN_PROG);
    ASSERT_NE(busy.completion_status, NIXL_IN_PROG);
    ASSERT_NE(busy.completion_status, NIXL_SUCCESS);
    ASSERT_EQ(busy.status_untouched, 1u);
    ASSERT_EQ(busy.accepted, 1u);
    ASSERT_EQ(busy.completed, 1u);
    ASSERT_EQ(busy.timed_out, 0u);

    const std::array<InvalidSpec, 6> invalid{{
        {{local_view_, 1, 0}, {remote_view_, 0, 0}, 1, 0, 0, NIXL_ERR_INVALID_PARAM, 1},
        {{local_view_, 0, local_length_}, {remote_view_, 0, 0}, 1, 0, 0, NIXL_ERR_INVALID_PARAM, 2},
        {{local_view_, 0, 0}, {remote_view_, 0, 0}, 0, 0, 0, NIXL_ERR_INVALID_PARAM, 3},
        {{local_view_, 0, 0}, {remote_view_, 0, 0}, 4096, 1, 0, NIXL_ERR_NOT_SUPPORTED, 4},
        {{local_view_, 0, 0}, {remote_view_, 0, 0}, 4096, 0, 1, NIXL_ERR_NOT_SUPPORTED, 5},
        {{local_view_, 0, 0},
         {remote_view_, 0, 0},
         std::numeric_limits<uint64_t>::max(),
         0,
         0,
         NIXL_ERR_INVALID_PARAM,
         6},
    }};
    for (const auto &spec : invalid) {
        const DeviceResult result = launchInvalid(spec);
        ASSERT_EQ(static_cast<nixl_status_t>(result.submission_status), spec.expected_status)
            << spec.kind;
        ASSERT_EQ(static_cast<nixl_status_t>(result.expected), spec.expected_status) << spec.kind;
        ASSERT_EQ(result.status_untouched, 1u) << spec.kind;
    }
    const DeviceResult null_status = launchNullStatus(valid);
    ASSERT_EQ(static_cast<nixl_status_t>(null_status.submission_status), NIXL_ERR_INVALID_PARAM);
    sendDataDone();
    waitValidated();
}

} // namespace
