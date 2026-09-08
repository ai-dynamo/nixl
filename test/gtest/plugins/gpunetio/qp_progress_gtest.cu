/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Paired GPUNETIO integration harness.  It observes only public NIXL/CUDA results.
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nixl.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr size_t kLargeBytes = 2ULL << 20;
constexpr size_t kSmallBytes = 4ULL << 10;
constexpr size_t kStressDescriptors = 512;
constexpr size_t kStressRequests = 7;
constexpr size_t kFastChurnRequests = 129;
constexpr int kWarmup = 20;
constexpr int kMeasured = 100;
constexpr int kRepeats = 1;
constexpr auto kTimeout = std::chrono::seconds(30);

struct Config {
    std::string role;
    fs::path coord;
    std::string target_ipv4;
    std::string network_device;
    std::string oob_interface;
    std::string gid_index;
    int control_port = 0;
    int source_port = 0;
    int target_a_port = 0;
    int target_b_port = 0;
    size_t control_bytes = kSmallBytes;
};

[[noreturn]] void
Fail(const std::string &message) {
    throw std::runtime_error(message);
}

void
CheckCuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) {
        Fail(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void
CheckNixl(nixl_status_t status, const char *what) {
    if (status != NIXL_SUCCESS) {
        Fail(std::string(what) + ": status=" + std::to_string(static_cast<int>(status)));
    }
}

uint64_t
NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
        .count();
}

std::optional<std::string>
Env(const char *key) {
    const char *value = std::getenv(key);
    return value == nullptr || *value == '\0' ? std::nullopt : std::optional<std::string>(value);
}

int
EnvPort(const char *key) {
    const auto value = Env(key);
    if (!value) {
        return 0;
    }
    try {
        const int port = std::stoi(*value);
        return port > 0 && port <= 65535 ? port : 0;
    }
    catch (...) {
        return 0;
    }
}

size_t
EnvBytes(const char *key, size_t fallback) {
    const auto value = Env(key);
    if (!value) {
        return fallback;
    }
    try {
        const size_t bytes = std::stoull(*value);
        return bytes == 0 ? fallback : bytes;
    }
    catch (...) {
        return fallback;
    }
}

Config
GetConfig() {
    Config config;
    config.role = Env("NIXL_QP_PROGRESS_ROLE").value_or("");
    config.coord = Env("NIXL_QP_PROGRESS_COORD_DIR").value_or("");
    config.target_ipv4 = Env("NIXL_QP_PROGRESS_TARGET_IPV4").value_or("");
    config.network_device = Env("NIXL_QP_PROGRESS_NETWORK_DEVICE").value_or("");
    config.oob_interface = Env("NIXL_QP_PROGRESS_OOB_INTERFACE").value_or("");
    config.gid_index = Env("NIXL_QP_PROGRESS_GID_INDEX").value_or("");
    config.control_port = EnvPort("NIXL_QP_PROGRESS_CONTROL_PORT");
    config.source_port = EnvPort("NIXL_QP_PROGRESS_SOURCE_OOB_PORT");
    config.target_a_port = EnvPort("NIXL_QP_PROGRESS_TARGET_A_OOB_PORT");
    config.target_b_port = EnvPort("NIXL_QP_PROGRESS_TARGET_B_OOB_PORT");
    config.control_bytes = EnvBytes("NIXL_QP_PROGRESS_CONTROL_BYTES", kSmallBytes);
    return config;
}

bool
IsTarget(const Config &config) {
    return config.role == "target" || config.role == "target-fault";
}

std::optional<std::string>
ConfigProblem(const Config &config, bool fault) {
    if (config.role.empty() || config.coord.empty() || config.control_port == 0 ||
        config.source_port == 0 || config.target_a_port == 0 || config.target_b_port == 0) {
        return "set NIXL_QP_PROGRESS_ROLE, COORD_DIR, and all paired ports";
    }
    if (config.role != "source" && !IsTarget(config)) {
        return "NIXL_QP_PROGRESS_ROLE must be source, target, or target-fault";
    }
    if (fault != (config.role == "target-fault") && IsTarget(config)) {
        return "role does not match this fault/non-fault case";
    }
    if (config.role == "source" && config.target_ipv4.empty()) {
        return "source requires numeric NIXL_QP_PROGRESS_TARGET_IPV4";
    }
    int devices = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&devices);
    if (cuda_status != cudaSuccess) {
        return "CUDA runtime/device unavailable";
    }
    if ((config.role == "source" && devices < 1) || (IsTarget(config) && devices < 2)) {
        return "paired harness needs one source GPU or two target GPUs";
    }
    return std::nullopt;
}

class Control {
public:
    explicit Control(const Config &config) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) {
            Fail("control socket");
        }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(config.control_port);
        if (IsTarget(config)) {
            int one = 1;
            setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            address.sin_addr.s_addr = INADDR_ANY;
            if (bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
                listen(fd_, 1) != 0) {
                Fail("control bind/listen");
            }
            const int accepted = accept(fd_, nullptr, nullptr);
            close(fd_);
            fd_ = accepted;
            if (fd_ < 0) {
                Fail("control accept");
            }
        } else {
            if (inet_pton(AF_INET, config.target_ipv4.c_str(), &address.sin_addr) != 1) {
                Fail("NIXL_QP_PROGRESS_TARGET_IPV4 must be numeric IPv4");
            }
            const auto deadline = Clock::now() + kTimeout;
            while (connect(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
                if (Clock::now() >= deadline) {
                    Fail("control connect timeout");
                }
                close(fd_);
                fd_ = socket(AF_INET, SOCK_STREAM, 0);
                if (fd_ < 0) {
                    Fail("control retry socket");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        timeval timeout{static_cast<long>(kTimeout.count()), 0};
        setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    }

    ~Control() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    void
    Send(char message) const {
        if (send(fd_, &message, 1, MSG_NOSIGNAL) != 1) {
            Fail("control send");
        }
    }

    void
    Expect(char expected) const {
        char actual = 0;
        if (recv(fd_, &actual, 1, MSG_WAITALL) != 1 || actual != expected) {
            Fail("control receive/order");
        }
    }

private:
    int fd_ = -1;
};

void
AtomicWrite(const fs::path &path, const std::string &data) {
    const fs::path temporary = path.string() + ".tmp." + std::to_string(getpid());
    std::ofstream output(temporary, std::ios::binary);
    if (!output) {
        Fail("open metadata for write: " + temporary.string());
    }
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
    output.close();
    fs::rename(temporary, path);
}

std::string
ReadFile(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        Fail("open metadata for read: " + path.string());
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

__device__ uint8_t
Pattern(uint64_t seed, size_t offset) {
    return static_cast<uint8_t>((seed + offset * 1315423911ULL + (offset >> 7U) * 17ULL) & 0xffU);
}

__global__ void
FillKernel(uint8_t *data, size_t bytes, uint64_t seed) {
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < bytes;
         offset += blockDim.x * gridDim.x) {
        data[offset] = Pattern(seed, offset);
    }
}

__global__ void
EpochKernel(uint64_t *epoch, uint64_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *epoch = value;
    }
}

__global__ void
DelayKernel(uint64_t cycles) {
    const uint64_t begin = clock64();
    while (clock64() - begin < cycles) {}
}

__global__ void
VerifyKernel(const uint8_t *data,
             size_t bytes,
             uint64_t seed,
             const uint64_t *epoch,
             uint64_t expected_epoch,
             unsigned long long *mismatches) {
    __shared__ int epoch_matches;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_system> marker(
            *const_cast<unsigned long long *>(reinterpret_cast<const unsigned long long *>(epoch)));
        epoch_matches = marker.load(cuda::memory_order_acquire) == expected_epoch;
        if (!epoch_matches) {
            atomicAdd(mismatches, 1ULL);
        }
    }
    __syncthreads();
    if (!epoch_matches) {
        return;
    }
    unsigned long long local = 0;
    for (size_t offset = blockIdx.x * blockDim.x + threadIdx.x; offset < bytes;
         offset += blockDim.x * gridDim.x) {
        local += data[offset] != Pattern(seed, offset);
    }
    if (local != 0) {
        atomicAdd(mismatches, local);
    }
}

class DeviceMemory {
public:
    DeviceMemory(size_t bytes, int device) : bytes_(bytes), device_(device) {
        CheckCuda(cudaSetDevice(device_), "cudaSetDevice allocation");
        CheckCuda(cudaMalloc(&data_, bytes_), "cudaMalloc data");
        CheckCuda(cudaMalloc(&epoch_, sizeof(uint64_t)), "cudaMalloc epoch");
    }

    ~DeviceMemory() {
        cudaSetDevice(device_);
        if (epoch_ != nullptr) {
            cudaFree(epoch_);
        }
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceMemory(const DeviceMemory &) = delete;
    DeviceMemory &
    operator=(const DeviceMemory &) = delete;

    uint8_t *
    data() const {
        return data_;
    }

    uint64_t *
    epoch() const {
        return epoch_;
    }

    size_t
    bytes() const {
        return bytes_;
    }

    int
    device() const {
        return device_;
    }

private:
    size_t bytes_;
    int device_;
    uint8_t *data_ = nullptr;
    uint64_t *epoch_ = nullptr;
};

class VerifyCounter {
public:
    explicit VerifyCounter(int device) : device_(device) {
        CheckCuda(cudaSetDevice(device_), "cudaSetDevice counter");
        CheckCuda(cudaMalloc(&device_counter_, sizeof(*device_counter_)),
                  "cudaMalloc verify counter");
        CheckCuda(cudaMallocHost(&host_counter_, sizeof(*host_counter_)),
                  "cudaMallocHost verify counter");
    }

    ~VerifyCounter() {
        cudaSetDevice(device_);
        if (host_counter_ != nullptr) {
            cudaFreeHost(host_counter_);
        }
        if (device_counter_ != nullptr) {
            cudaFree(device_counter_);
        }
    }

    uint64_t
    Verify(const DeviceMemory &memory, uint64_t seed, uint64_t epoch, cudaStream_t stream = 0) {
        CheckCuda(cudaSetDevice(device_), "cudaSetDevice verify");
        CheckCuda(cudaMemsetAsync(device_counter_, 0, sizeof(*device_counter_), stream),
                  "clear verify counter");
        VerifyKernel<<<std::min<size_t>(65535, (memory.bytes() + 255) / 256), 256, 0, stream>>>(
            memory.data(), memory.bytes(), seed, memory.epoch(), epoch, device_counter_);
        CheckCuda(cudaGetLastError(), "launch verify kernel");
        CheckCuda(cudaMemcpyAsync(host_counter_,
                                  device_counter_,
                                  sizeof(*host_counter_),
                                  cudaMemcpyDeviceToHost,
                                  stream),
                  "copy verify counter");
        CheckCuda(cudaStreamSynchronize(stream), "verify stream synchronization");
        return *host_counter_;
    }

private:
    int device_;
    unsigned long long *device_counter_ = nullptr;
    unsigned long long *host_counter_ = nullptr;
};

void
Fill(const DeviceMemory &memory, uint64_t seed, uint64_t epoch, cudaStream_t stream = 0) {
    CheckCuda(cudaSetDevice(memory.device()), "cudaSetDevice fill");
    FillKernel<<<std::min<size_t>(65535, (memory.bytes() + 255) / 256), 256, 0, stream>>>(
        memory.data(), memory.bytes(), seed);
    CheckCuda(cudaGetLastError(), "launch fill kernel");
    EpochKernel<<<1, 1, 0, stream>>>(memory.epoch(), epoch);
    CheckCuda(cudaGetLastError(), "launch epoch kernel");
}

nixl_b_params_t
BackendParams(const Config &config, int port, int device) {
    nixl_b_params_t params{{"gpu_devices", std::to_string(device)},
                           {"cuda_streams", "4"},
                           {"data_qp_count", "1"},
                           {"oob_port", std::to_string(port)}};
    if (!config.network_device.empty()) {
        params["network_devices"] = config.network_device;
    }
    if (!config.oob_interface.empty()) {
        params["oob_interface"] = config.oob_interface;
    }
    if (!config.gid_index.empty()) {
        params["gid_index"] = config.gid_index;
    }
    return params;
}

nixl_opt_args_t
BackendOnly(nixlBackendH *backend) {
    nixl_opt_args_t options;
    options.backends.push_back(backend);
    return options;
}

nixl_reg_dlist_t
Registration(const DeviceMemory &memory) {
    nixl_reg_dlist_t registration(VRAM_SEG);
    registration.addDesc(
        nixlBlobDesc(reinterpret_cast<uintptr_t>(memory.data()), memory.bytes(), memory.device()));
    registration.addDesc(nixlBlobDesc(
        reinterpret_cast<uintptr_t>(memory.epoch()), sizeof(uint64_t), memory.device()));
    return registration;
}

nixl_xfer_dlist_t
TransferList(const DeviceMemory &memory) {
    nixl_xfer_dlist_t list(VRAM_SEG);
    list.addDesc(
        nixlBasicDesc(reinterpret_cast<uintptr_t>(memory.data()), memory.bytes(), memory.device()));
    list.addDesc(nixlBasicDesc(
        reinterpret_cast<uintptr_t>(memory.epoch()), sizeof(uint64_t), memory.device()));
    return list;
}

nixl_xfer_dlist_t
RemoteList(uintptr_t data, size_t bytes, uintptr_t epoch, int device) {
    nixl_xfer_dlist_t list(VRAM_SEG);
    list.addDesc(nixlBasicDesc(data, bytes, device));
    list.addDesc(nixlBasicDesc(epoch, sizeof(uint64_t), device));
    return list;
}

struct Endpoint {
    std::string name;
    nixlAgent agent;
    nixlBackendH *backend = nullptr;
    nixl_opt_args_t options;
    nixl_reg_dlist_t registration;
    bool registration_active = false;

    Endpoint(const Config &config, std::string endpoint_name, int port, const DeviceMemory &memory)
        : name(std::move(endpoint_name)),
          agent(name, nixlAgentConfig(true)),
          registration(Registration(memory)) {
        CheckCuda(cudaSetDevice(memory.device()), "cudaSetDevice endpoint");
        CheckNixl(
            agent.createBackend("GPUNETIO", BackendParams(config, port, memory.device()), backend),
            "create GPUNETIO backend");
        options = BackendOnly(backend);
        CheckNixl(agent.registerMem(registration, &options), "register GPUNETIO memory");
        registration_active = true;
    }

    void
    Deregister() {
        if (registration_active) {
            CheckNixl(agent.deregisterMem(registration, &options), "deregister GPUNETIO memory");
            registration_active = false;
        }
    }

    ~Endpoint() {
        if (registration_active) {
            agent.deregisterMem(registration, &options);
        }
    }
};

struct RemoteAddresses {
    uintptr_t a_data = 0;
    uintptr_t a_epoch = 0;
    uintptr_t b_data = 0;
    uintptr_t b_epoch = 0;
};

void
PublishMetadata(const Config &config,
                const Endpoint &a,
                const DeviceMemory &a_memory,
                const Endpoint &b,
                const DeviceMemory &b_memory) {
    fs::create_directories(config.coord);
    std::string metadata;
    CheckNixl(a.agent.getLocalMD(metadata), "get target A metadata");
    AtomicWrite(config.coord / "target-a.md", metadata);
    CheckNixl(b.agent.getLocalMD(metadata), "get target B metadata");
    AtomicWrite(config.coord / "target-b.md", metadata);
    AtomicWrite(config.coord / "target-addresses.txt",
                std::to_string(reinterpret_cast<uintptr_t>(a_memory.data())) + " " +
                    std::to_string(reinterpret_cast<uintptr_t>(a_memory.epoch())) + " " +
                    std::to_string(reinterpret_cast<uintptr_t>(b_memory.data())) + " " +
                    std::to_string(reinterpret_cast<uintptr_t>(b_memory.epoch())) + "\n");
}

RemoteAddresses
LoadMetadata(const Config &config, nixlAgent &source) {
    std::string name;
    CheckNixl(source.loadRemoteMD(ReadFile(config.coord / "target-a.md"), name),
              "load target A metadata");
    if (name != "qp-progress-a") {
        Fail("target A metadata name mismatch");
    }
    CheckNixl(source.loadRemoteMD(ReadFile(config.coord / "target-b.md"), name),
              "load target B metadata");
    if (name != "qp-progress-b") {
        Fail("target B metadata name mismatch");
    }
    RemoteAddresses addresses;
    std::istringstream input(ReadFile(config.coord / "target-addresses.txt"));
    input >> addresses.a_data >> addresses.a_epoch >> addresses.b_data >> addresses.b_epoch;
    if (!input) {
        Fail("invalid target address metadata");
    }
    return addresses;
}

nixl_opt_args_t
Attached(nixlBackendH *backend, cudaStream_t stream, std::optional<std::string> notification) {
    nixl_opt_args_t options = BackendOnly(backend);
    options.customParam.resize(sizeof(stream));
    std::memcpy(options.customParam.data(), &stream, sizeof(stream));
    options.skipDescMerge = true;
    if (notification) {
        options.notif = *notification;
    }
    return options;
}

nixl_status_t
Wait(nixlAgent &agent, nixlXferReqH *request, std::chrono::milliseconds timeout) {
    const auto deadline = Clock::now() + timeout;
    nixl_status_t status = NIXL_IN_PROG;
    while (status == NIXL_IN_PROG && Clock::now() < deadline) {
        status = agent.getXferStatus(request);
    }
    return status;
}

void
Release(nixlAgent &agent, nixlXferReqH *request) {
    CheckNixl(agent.releaseXferReq(request), "release transfer request");
}

uint64_t
SeedA(uint64_t round) {
    return 0xa500000000000031ULL + (round << 8U);
}

uint64_t
SeedB(uint64_t round) {
    return 0xb400000000000073ULL + (round << 8U);
}

uint64_t
EpochA(uint64_t round) {
    return 0xe100000000000000ULL + round;
}

uint64_t
EpochB(uint64_t round) {
    return 0xe200000000000000ULL + round;
}

struct TransferTimes {
    uint64_t start_ns = 0;
    uint64_t done_ns = 0;
};

TransferTimes
PostAndWait(nixlAgent &source, nixlXferReqH *request) {
    TransferTimes times{NowNs(), 0};
    const nixl_status_t post = source.postXferReq(request);
    if (post != NIXL_SUCCESS && post != NIXL_IN_PROG) {
        Fail("post transfer: status=" + std::to_string(static_cast<int>(post)));
    }
    const nixl_status_t status =
        post == NIXL_SUCCESS ? post : Wait(source, request, std::chrono::seconds(20));
    if (status != NIXL_SUCCESS) {
        Fail("transfer completion: status=" + std::to_string(static_cast<int>(status)));
    }
    times.done_ns = NowNs();
    return times;
}

uint64_t
Percentile(std::vector<uint64_t> values, double percentile) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>((values.size() - 1) * percentile);
    return values[index];
}

void
WritePerformanceJson(const Config &config,
                     const std::vector<uint64_t> &b_windows,
                     const std::vector<uint64_t> &a_windows,
                     const std::vector<uint64_t> &max_windows,
                     uint64_t payload_bytes,
                     uint64_t marker_bytes,
                     uint64_t wall_ns) {
    const uint64_t transfer_ns =
        std::accumulate(max_windows.begin(), max_windows.end(), uint64_t{0});
    const double transfer_gbps = double(payload_bytes + marker_bytes) / transfer_ns;
    const double wall_gbps = double(payload_bytes + marker_bytes) / wall_ns;
    AtomicWrite(
        config.coord / "qp_progress_performance.json",
        "{\"result\":\"PASS\",\"warmup\":" + std::to_string(kWarmup) +
            ",\"measured\":" + std::to_string(kMeasured) +
            ",\"external_pair_repeats_required\":3"
            ",\"b_post_to_done_p50_ns\":" +
            std::to_string(Percentile(b_windows, 0.50)) +
            ",\"b_post_to_done_p99_ns\":" + std::to_string(Percentile(b_windows, 0.99)) +
            ",\"a_post_to_done_p50_ns\":" + std::to_string(Percentile(a_windows, 0.50)) +
            ",\"a_post_to_done_p99_ns\":" + std::to_string(Percentile(a_windows, 0.99)) +
            ",\"max_window_p50_ns\":" + std::to_string(Percentile(max_windows, 0.50)) +
            ",\"max_window_p99_ns\":" + std::to_string(Percentile(max_windows, 0.99)) +
            ",\"actual_payload_bytes\":" + std::to_string(payload_bytes) +
            ",\"actual_marker_bytes\":" + std::to_string(marker_bytes) +
            ",\"actual_transfer_window_bytes\":" + std::to_string(payload_bytes + marker_bytes) +
            ",\"transfer_window_ns\":" + std::to_string(transfer_ns) +
            ",\"transfer_window_GBps\":" + std::to_string(transfer_gbps) +
            ",\"wall_GBps\":" + std::to_string(wall_gbps) +
            ",\"completed_requests\":" + std::to_string(2 * (kWarmup + kMeasured)) +
            ",\"measured_requests\":" + std::to_string(2 * kMeasured) +
            ",\"payload_and_marker_mismatches\":0" + ",\"walltime_ns\":" + std::to_string(wall_ns) +
            ",\"walltime_scope\":\"caller only; not serving\"" +
            ",\"window_definition\":\"source post to source API completion; no CQ timestamps\"}\n");
}

void
TargetPerformance(const Config &config) {
    // All target resources exist before the persistent backend is constructed.
    DeviceMemory a_memory(kLargeBytes, 0);
    DeviceMemory b_memory(kSmallBytes, 1);
    VerifyCounter a_counter(0);
    VerifyCounter b_counter(1);
    Endpoint a(config, "qp-progress-a", config.target_a_port, a_memory);
    Endpoint b(config, "qp-progress-b", config.target_b_port, b_memory);
    Control control(config);
    PublishMetadata(config, a, a_memory, b, b_memory);
    control.Send('R');
    for (uint64_t repeat = 0; repeat < kRepeats; ++repeat) {
        for (uint64_t round = 0; round < kWarmup + kMeasured; ++round) {
            control.Expect('D');
            ASSERT_EQ(a_counter.Verify(
                          a_memory, SeedA(repeat * 1000 + round), EpochA(repeat * 1000 + round)),
                      0U);
            ASSERT_EQ(b_counter.Verify(
                          b_memory, SeedB(repeat * 1000 + round), EpochB(repeat * 1000 + round)),
                      0U);
            control.Send('V');
        }
    }
    control.Expect('C');
    control.Send('K');
}

void
SourcePerformance(const Config &config) {
    DeviceMemory a_memory(kLargeBytes, 0);
    DeviceMemory b_memory(kSmallBytes, 0);
    Endpoint source(config, "qp-progress-source", config.source_port, a_memory);
    // Both source buffers are allocated before backend construction; B is registered before posts.
    nixl_reg_dlist_t b_registration = Registration(b_memory);
    CheckNixl(source.agent.registerMem(b_registration, &source.options),
              "register source B memory");
    Control control(config);
    control.Expect('R');
    const RemoteAddresses remote = LoadMetadata(config, source.agent);
    std::vector<uint64_t> b_windows, a_windows, max_windows;
    uint64_t wall_start = 0;
    for (uint64_t repeat = 0; repeat < kRepeats; ++repeat) {
        for (uint64_t round = 0; round < kWarmup + kMeasured; ++round) {
            const uint64_t token = repeat * 1000 + round;
            Fill(a_memory, SeedA(token), EpochA(token));
            Fill(b_memory, SeedB(token), EpochB(token));
            CheckCuda(cudaStreamSynchronize(0), "prepare performance payloads");
            nixlXferReqH *a_request = nullptr;
            nixlXferReqH *b_request = nullptr;
            const auto a_local = TransferList(a_memory);
            const auto b_local = TransferList(b_memory);
            const auto a_remote = RemoteList(remote.a_data, kLargeBytes, remote.a_epoch, 0);
            const auto b_remote = RemoteList(remote.b_data, kSmallBytes, remote.b_epoch, 1);
            CheckNixl(
                source.agent.createXferReq(
                    NIXL_WRITE, a_local, a_remote, "qp-progress-a", a_request, &source.options),
                "create performance A");
            CheckNixl(
                source.agent.createXferReq(
                    NIXL_WRITE, b_local, b_remote, "qp-progress-b", b_request, &source.options),
                "create performance B");
            const uint64_t start = NowNs();
            if (round == kWarmup) {
                wall_start = start;
            }
            const nixl_status_t a_post = source.agent.postXferReq(a_request);
            const uint64_t b_post_ns = NowNs();
            const nixl_status_t b_post = source.agent.postXferReq(b_request);
            ASSERT_TRUE(a_post == NIXL_SUCCESS || a_post == NIXL_IN_PROG);
            ASSERT_TRUE(b_post == NIXL_SUCCESS || b_post == NIXL_IN_PROG);
            nixl_status_t a_status = a_post;
            nixl_status_t b_status = b_post;
            uint64_t a_done = a_status == NIXL_SUCCESS ? NowNs() : 0;
            uint64_t b_done = b_status == NIXL_SUCCESS ? NowNs() : 0;
            const auto deadline = Clock::now() + std::chrono::seconds(20);
            while ((a_status == NIXL_IN_PROG || b_status == NIXL_IN_PROG) &&
                   Clock::now() < deadline) {
                if (b_status == NIXL_IN_PROG) {
                    b_status = source.agent.getXferStatus(b_request);
                    if (b_status == NIXL_SUCCESS) {
                        b_done = NowNs();
                    }
                }
                if (a_status == NIXL_IN_PROG) {
                    a_status = source.agent.getXferStatus(a_request);
                    if (a_status == NIXL_SUCCESS) {
                        a_done = NowNs();
                    }
                }
            }
            ASSERT_EQ(a_status, NIXL_SUCCESS);
            ASSERT_EQ(b_status, NIXL_SUCCESS);
            if (round >= kWarmup) {
                b_windows.push_back(b_done - b_post_ns);
                a_windows.push_back(a_done - start);
                max_windows.push_back(std::max(a_done, b_done) - start);
            }
            Release(source.agent, b_request);
            Release(source.agent, a_request);
            control.Send('D');
            control.Expect('V');
        }
    }
    const uint64_t wall_ns = NowNs() - wall_start;
    WritePerformanceJson(config,
                         b_windows,
                         a_windows,
                         max_windows,
                         kRepeats * kMeasured * (kLargeBytes + kSmallBytes),
                         kRepeats * kMeasured * 2 * sizeof(uint64_t),
                         wall_ns);
    CheckNixl(source.agent.invalidateRemoteMD("qp-progress-a"), "invalidate target A metadata");
    CheckNixl(source.agent.invalidateRemoteMD("qp-progress-b"), "invalidate target B metadata");
    CheckNixl(source.agent.deregisterMem(b_registration, &source.options),
              "deregister source B memory");
    control.Send('C');
    control.Expect('K');
}

// The stress layout creates 512 payload descriptors plus one exact epoch descriptor per request.
struct StressMemory {
    std::vector<std::unique_ptr<DeviceMemory>> slots;

    explicit StressMemory(int device) {
        slots.reserve(kStressRequests);
        for (size_t slot = 0; slot < kStressRequests; ++slot) {
            slots.emplace_back(
                std::make_unique<DeviceMemory>(kStressDescriptors * kSmallBytes, device));
        }
    }
};

nixl_reg_dlist_t
StressRegistration(const StressMemory &memory) {
    nixl_reg_dlist_t list(VRAM_SEG);
    for (const auto &slot : memory.slots) {
        list.addDesc(
            nixlBlobDesc(reinterpret_cast<uintptr_t>(slot->data()), slot->bytes(), slot->device()));
        list.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(slot->epoch()), sizeof(uint64_t), slot->device()));
    }
    return list;
}

nixl_xfer_dlist_t
StressList(const DeviceMemory &memory) {
    nixl_xfer_dlist_t list(VRAM_SEG);
    for (size_t descriptor = 0; descriptor < kStressDescriptors; ++descriptor) {
        list.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(memory.data() + descriptor * kSmallBytes),
                          kSmallBytes,
                          memory.device()));
    }
    list.addDesc(nixlBasicDesc(
        reinterpret_cast<uintptr_t>(memory.epoch()), sizeof(uint64_t), memory.device()));
    return list;
}

nixl_xfer_dlist_t
StressRemoteList(uintptr_t data, size_t bytes, uintptr_t epoch, int device) {
    nixl_xfer_dlist_t list(VRAM_SEG);
    for (size_t descriptor = 0; descriptor < kStressDescriptors; ++descriptor) {
        list.addDesc(nixlBasicDesc(data + descriptor * kSmallBytes, kSmallBytes, device));
    }
    list.addDesc(nixlBasicDesc(epoch, sizeof(uint64_t), device));
    return list;
}

void
TargetStress(const Config &config) {
    StressMemory a_memory(0);
    StressMemory b_memory(1);
    std::vector<std::unique_ptr<VerifyCounter>> a_counters;
    std::vector<std::unique_ptr<VerifyCounter>> b_counters;
    for (size_t index = 0; index < kStressRequests; ++index) {
        a_counters.emplace_back(std::make_unique<VerifyCounter>(0));
        b_counters.emplace_back(std::make_unique<VerifyCounter>(1));
    }
    // Register all slots before persistent engines; source uses matching full registrations.
    DeviceMemory a_anchor(1, 0);
    DeviceMemory b_anchor(1, 1);
    Endpoint a(config, "qp-progress-a", config.target_a_port, a_anchor);
    Endpoint b(config, "qp-progress-b", config.target_b_port, b_anchor);
    const nixl_reg_dlist_t a_registration = StressRegistration(a_memory);
    const nixl_reg_dlist_t b_registration = StressRegistration(b_memory);
    CheckNixl(a.agent.registerMem(a_registration, &a.options), "register target A stress slots");
    CheckNixl(b.agent.registerMem(b_registration, &b.options), "register target B stress slots");
    Control control(config);
    PublishMetadata(config, a, a_anchor, b, b_anchor);
    // Slot addresses are separate metadata because serialized NIXL metadata carries registration
    // information.
    std::ofstream addresses(config.coord / "stress-addresses.txt");
    for (size_t index = 0; index < kStressRequests; ++index) {
        addresses << reinterpret_cast<uintptr_t>(a_memory.slots[index]->data()) << ' '
                  << reinterpret_cast<uintptr_t>(a_memory.slots[index]->epoch()) << ' '
                  << reinterpret_cast<uintptr_t>(b_memory.slots[index]->data()) << ' '
                  << reinterpret_cast<uintptr_t>(b_memory.slots[index]->epoch()) << '\n';
    }
    addresses.close();
    control.Send('R');
    nixl_notifs_t a_notifications;
    nixl_notifs_t b_notifications;
    auto poll_notifications = [&] {
        ASSERT_EQ(a.agent.getNotifs(a_notifications), NIXL_SUCCESS);
        ASSERT_EQ(b.agent.getNotifs(b_notifications), NIXL_SUCCESS);
    };
    for (size_t index = 0; index < kStressRequests; ++index) {
        control.Expect('B');
        ASSERT_EQ(b_counters[index]->Verify(*b_memory.slots[index], SeedB(index), EpochB(index)),
                  0U);
        poll_notifications();
        control.Send('V');
    }
    for (size_t index = 1; index < kStressRequests; index += 2) {
        control.Expect('A');
        ASSERT_EQ(a_counters[index]->Verify(*a_memory.slots[index], SeedA(index), EpochA(index)),
                  0U);
        poll_notifications();
        control.Send('V');
    }
    for (size_t index = 0; index < kStressRequests; index += 2) {
        control.Expect('A');
        ASSERT_EQ(a_counters[index]->Verify(*a_memory.slots[index], SeedA(index), EpochA(index)),
                  0U);
        poll_notifications();
        control.Send('V');
    }
    for (size_t index = 0; index < kFastChurnRequests; ++index) {
        control.Expect('B');
        ASSERT_EQ(b_counters[0]->Verify(*b_memory.slots[0],
                                        SeedB(kStressRequests + index),
                                        EpochB(kStressRequests + index)),
                  0U);
        poll_notifications();
        control.Send('V');
    }
    control.Expect('R');
    const size_t read_token = kStressRequests + kFastChurnRequests;
    Fill(*b_memory.slots[0], SeedB(read_token), EpochB(read_token));
    CheckCuda(cudaStreamSynchronize(0), "prepare multi-chunk READ payload");
    control.Send('P');
    control.Expect('Q');
    control.Expect('D');
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (
        (a_notifications["qp-progress-source"].size() < kStressRequests + 1 ||
         b_notifications["qp-progress-source"].size() < kStressRequests + kFastChurnRequests + 1) &&
        Clock::now() < deadline) {
        poll_notifications();
    }
    const auto &a_messages = a_notifications["qp-progress-source"];
    const auto &b_messages = b_notifications["qp-progress-source"];
    ASSERT_EQ(a_messages.size(), kStressRequests + 1);
    ASSERT_EQ(b_messages.size(), kStressRequests + kFastChurnRequests + 1);
    for (size_t index = 0; index < kStressRequests; ++index) {
        EXPECT_NE(
            std::find(a_messages.begin(), a_messages.end(), "data-a-" + std::to_string(index)),
            a_messages.end());
        EXPECT_NE(
            std::find(b_messages.begin(), b_messages.end(), "data-b-" + std::to_string(index)),
            b_messages.end());
    }
    for (size_t index = 0; index < kFastChurnRequests; ++index) {
        EXPECT_NE(std::find(b_messages.begin(),
                            b_messages.end(),
                            "data-b-" + std::to_string(kStressRequests + index)),
                  b_messages.end());
    }
    EXPECT_NE(std::find(a_messages.begin(), a_messages.end(), "standalone-a"), a_messages.end());
    EXPECT_NE(std::find(b_messages.begin(), b_messages.end(), "standalone-b"), b_messages.end());
    control.Send('V');
    control.Expect('C');
    CheckNixl(a.agent.deregisterMem(a_registration, &a.options),
              "deregister target A stress slots");
    CheckNixl(b.agent.deregisterMem(b_registration, &b.options),
              "deregister target B stress slots");
    control.Send('K');
}

std::array<RemoteAddresses, kStressRequests>
ReadStressAddresses(const Config &config) {
    std::array<RemoteAddresses, kStressRequests> addresses{};
    std::ifstream input(config.coord / "stress-addresses.txt");
    for (auto &address : addresses) {
        input >> address.a_data >> address.a_epoch >> address.b_data >> address.b_epoch;
        if (!input) {
            Fail("invalid stress address metadata");
        }
    }
    return addresses;
}

void
SourceStress(const Config &config) {
    StressMemory a_memory(0);
    StressMemory b_memory(0);
    VerifyCounter read_counter(0);
    DeviceMemory anchor(1, 0);
    Endpoint source(config, "qp-progress-source", config.source_port, anchor);
    const nixl_reg_dlist_t a_registration = StressRegistration(a_memory);
    const nixl_reg_dlist_t b_registration = StressRegistration(b_memory);
    CheckNixl(source.agent.registerMem(a_registration, &source.options),
              "register source A stress slots");
    CheckNixl(source.agent.registerMem(b_registration, &source.options),
              "register source B stress slots");
    cudaStream_t delayed_stream = nullptr;
    cudaStream_t fast_stream = nullptr;
    CheckCuda(cudaStreamCreateWithFlags(&delayed_stream, cudaStreamNonBlocking),
              "create delayed stream");
    CheckCuda(cudaStreamCreateWithFlags(&fast_stream, cudaStreamNonBlocking), "create fast stream");
    Control control(config);
    control.Expect('R');
    LoadMetadata(config, source.agent);
    const auto remote = ReadStressAddresses(config);
    std::array<nixlXferReqH *, kStressRequests> a_requests{};
    std::array<nixlXferReqH *, kStressRequests> b_requests{};
    // Alternating attached streams on the same A peer reverse CPU post order versus SQ readiness.
    DelayKernel<<<1, 1, 0, delayed_stream>>>(50000000ULL);
    CheckCuda(cudaGetLastError(), "launch one test-only delayed stream kernel");
    for (size_t index = 0; index < kStressRequests; ++index) {
        cudaStream_t stream = index % 2 == 0 ? delayed_stream : fast_stream;
        Fill(*a_memory.slots[index], SeedA(index), EpochA(index), stream);
        auto options = Attached(source.backend, stream, "data-a-" + std::to_string(index));
        CheckNixl(source.agent.createXferReq(NIXL_WRITE,
                                             StressList(*a_memory.slots[index]),
                                             StressRemoteList(remote[index].a_data,
                                                              a_memory.slots[index]->bytes(),
                                                              remote[index].a_epoch,
                                                              0),
                                             "qp-progress-a",
                                             a_requests[index],
                                             &options),
                  "create alternating A stress request");
        ASSERT_TRUE(source.agent.postXferReq(a_requests[index]) >= NIXL_SUCCESS);
    }
    for (size_t index = 0; index < kStressRequests; ++index) {
        Fill(*b_memory.slots[index], SeedB(index), EpochB(index), fast_stream);
        auto options = Attached(source.backend, fast_stream, "data-b-" + std::to_string(index));
        CheckNixl(source.agent.createXferReq(NIXL_WRITE,
                                             StressList(*b_memory.slots[index]),
                                             StressRemoteList(remote[index].b_data,
                                                              b_memory.slots[index]->bytes(),
                                                              remote[index].b_epoch,
                                                              1),
                                             "qp-progress-b",
                                             b_requests[index],
                                             &options),
                  "create initial B stress request");
        ASSERT_TRUE(source.agent.postXferReq(b_requests[index]) >= NIXL_SUCCESS);
    }
    for (size_t index = 0; index < kStressRequests; ++index) {
        ASSERT_EQ(Wait(source.agent, b_requests[index], std::chrono::seconds(10)), NIXL_SUCCESS);
        Release(source.agent, b_requests[index]);
        control.Send('B');
        control.Expect('V');
    }
    for (size_t index = 1; index < kStressRequests; index += 2) {
        ASSERT_EQ(Wait(source.agent, a_requests[index], std::chrono::seconds(10)), NIXL_SUCCESS);
        Release(source.agent, a_requests[index]);
        control.Send('A');
        control.Expect('V');
    }
    for (size_t index = 0; index < kStressRequests; index += 2) {
        ASSERT_EQ(Wait(source.agent, a_requests[index], std::chrono::seconds(10)), NIXL_SUCCESS);
        Release(source.agent, a_requests[index]);
        control.Send('A');
        control.Expect('V');
    }
    for (size_t index = 0; index < kFastChurnRequests; ++index) {
        const size_t token = kStressRequests + index;
        Fill(*b_memory.slots[0], SeedB(token), EpochB(token), fast_stream);
        nixlXferReqH *b_request = nullptr;
        auto b_options = Attached(source.backend, fast_stream, "data-b-" + std::to_string(token));
        CheckNixl(source.agent.createXferReq(
                      NIXL_WRITE,
                      StressList(*b_memory.slots[0]),
                      StressRemoteList(
                          remote[0].b_data, b_memory.slots[0]->bytes(), remote[0].b_epoch, 1),
                      "qp-progress-b",
                      b_request,
                      &b_options),
                  "create fast stress request");
        ASSERT_TRUE(source.agent.postXferReq(b_request) >= NIXL_SUCCESS);
        ASSERT_EQ(Wait(source.agent, b_request, std::chrono::seconds(10)), NIXL_SUCCESS);
        Release(source.agent, b_request);
        control.Send('B');
        control.Expect('V');
    }
    // Every B request has completed, released, and been target-verified, so slot 0 is safe to
    // reuse as the local destination of one matched 513-descriptor READ.
    control.Send('R');
    control.Expect('P');
    nixlXferReqH *read_request = nullptr;
    const auto read_options = Attached(source.backend, fast_stream, std::nullopt);
    CheckNixl(
        source.agent.createXferReq(
            NIXL_READ,
            StressList(*b_memory.slots[0]),
            StressRemoteList(remote[0].b_data, b_memory.slots[0]->bytes(), remote[0].b_epoch, 1),
            "qp-progress-b",
            read_request,
            &read_options),
        "create multi-chunk stress READ");
    const nixl_status_t read_post = source.agent.postXferReq(read_request);
    ASSERT_TRUE(read_post == NIXL_SUCCESS || read_post == NIXL_IN_PROG);
    ASSERT_EQ(read_post == NIXL_SUCCESS ?
                  read_post :
                  Wait(source.agent, read_request, std::chrono::seconds(10)),
              NIXL_SUCCESS);
    Release(source.agent, read_request);
    const size_t read_token = kStressRequests + kFastChurnRequests;
    ASSERT_EQ(read_counter.Verify(*b_memory.slots[0], SeedB(read_token), EpochB(read_token)), 0U);
    control.Send('Q');
    CheckNixl(source.agent.genNotif("qp-progress-a", "standalone-a", &source.options),
              "generate standalone notification for target A");
    CheckNixl(source.agent.genNotif("qp-progress-b", "standalone-b", &source.options),
              "generate standalone notification for target B");
    // Attached transfer completion is the progress/reuse boundary.
    control.Send('D');
    control.Expect('V');
    CheckNixl(source.agent.deregisterMem(a_registration, &source.options),
              "deregister source A stress slots");
    CheckNixl(source.agent.deregisterMem(b_registration, &source.options),
              "deregister source B stress slots");
    CheckCuda(cudaStreamDestroy(fast_stream), "destroy fast stream");
    CheckCuda(cudaStreamDestroy(delayed_stream), "destroy delayed stream");
    control.Send('C');
    control.Expect('K');
}

void
TargetReadWriteControl(const Config &config) {
    DeviceMemory write_memory(config.control_bytes, 0);
    DeviceMemory read_memory(config.control_bytes, 1);
    VerifyCounter write_counter(0);
    Endpoint write_target(config, "qp-progress-a", config.target_a_port, write_memory);
    Endpoint read_target(config, "qp-progress-b", config.target_b_port, read_memory);
    Control control(config);
    PublishMetadata(config, write_target, write_memory, read_target, read_memory);
    control.Send('R');
    for (uint64_t round = 0; round < kWarmup + kMeasured; ++round) {
        control.Expect('W');
        ASSERT_EQ(write_counter.Verify(write_memory, SeedA(round), EpochA(round)), 0U);
        control.Send('V');
        Fill(read_memory, SeedB(round), EpochB(round));
        CheckCuda(cudaStreamSynchronize(0), "prepare control READ payload");
        control.Send('P');
        control.Expect('D');
    }
    control.Expect('C');
    control.Send('K');
}

void
SourceReadWriteControl(const Config &config) {
    DeviceMemory write_memory(config.control_bytes, 0);
    DeviceMemory read_memory(config.control_bytes, 0);
    VerifyCounter read_counter(0);
    Endpoint source(config, "qp-progress-source", config.source_port, write_memory);
    const nixl_reg_dlist_t read_registration = Registration(read_memory);
    CheckNixl(source.agent.registerMem(read_registration, &source.options),
              "register control READ memory");
    Control control(config);
    control.Expect('R');
    const RemoteAddresses remote = LoadMetadata(config, source.agent);
    nixlXferReqH *write_request = nullptr;
    nixlXferReqH *read_request = nullptr;
    CheckNixl(source.agent.createXferReq(
                  NIXL_WRITE,
                  TransferList(write_memory),
                  RemoteList(remote.a_data, config.control_bytes, remote.a_epoch, 0),
                  "qp-progress-a",
                  write_request,
                  &source.options),
              "create repostable control WRITE");
    CheckNixl(source.agent.createXferReq(
                  NIXL_READ,
                  TransferList(read_memory),
                  RemoteList(remote.b_data, config.control_bytes, remote.b_epoch, 1),
                  "qp-progress-b",
                  read_request,
                  &source.options),
              "create repostable control READ");
    std::vector<uint64_t> write_windows;
    std::vector<uint64_t> read_windows;
    for (uint64_t round = 0; round < kWarmup + kMeasured; ++round) {
        Fill(write_memory, SeedA(round), EpochA(round));
        CheckCuda(cudaStreamSynchronize(0), "prepare control WRITE payload");
        const TransferTimes write_times = PostAndWait(source.agent, write_request);
        control.Send('W');
        control.Expect('V');
        control.Expect('P');
        const TransferTimes read_times = PostAndWait(source.agent, read_request);
        ASSERT_EQ(read_counter.Verify(read_memory, SeedB(round), EpochB(round)), 0U);
        if (round >= kWarmup) {
            write_windows.push_back(write_times.done_ns - write_times.start_ns);
            read_windows.push_back(read_times.done_ns - read_times.start_ns);
        }
        control.Send('D');
    }
    AtomicWrite(config.coord / "qp_progress_single_qp_control.json",
                "{\"result\":\"PASS\",\"warmup\":" + std::to_string(kWarmup) +
                    ",\"measured\":" + std::to_string(kMeasured) +
                    ",\"bytes_per_op\":" + std::to_string(config.control_bytes + sizeof(uint64_t)) +
                    ",\"write_p50_ns\":" + std::to_string(Percentile(write_windows, 0.50)) +
                    ",\"write_p99_ns\":" + std::to_string(Percentile(write_windows, 0.99)) +
                    ",\"read_p50_ns\":" + std::to_string(Percentile(read_windows, 0.50)) +
                    ",\"read_p99_ns\":" + std::to_string(Percentile(read_windows, 0.99)) +
                    ",\"window_definition\":\"source post to source API completion\"}\n");
    Release(source.agent, read_request);
    Release(source.agent, write_request);
    CheckNixl(source.agent.invalidateRemoteMD("qp-progress-a"), "invalidate control A metadata");
    CheckNixl(source.agent.invalidateRemoteMD("qp-progress-b"), "invalidate control B metadata");
    CheckNixl(source.agent.deregisterMem(read_registration, &source.options),
              "deregister control READ memory");
    control.Send('C');
    control.Expect('K');
}

void
TargetFault(const Config &config) {
    DeviceMemory a_memory(kLargeBytes, 0);
    DeviceMemory b_memory(kSmallBytes, 1);
    Endpoint a(config, "qp-progress-a", config.target_a_port, a_memory);
    Endpoint b(config, "qp-progress-b", config.target_b_port, b_memory);
    Control control(config);
    PublishMetadata(config, a, a_memory, b, b_memory);
    control.Send('R');
    control.Expect('M');
    b.Deregister();
    control.Send('F');
    control.Expect('E');
}

void
SourceFault(const Config &config) {
    DeviceMemory b_memory(kSmallBytes, 0);
    Endpoint source(config, "qp-progress-source", config.source_port, b_memory);
    Control control(config);
    control.Expect('R');
    const RemoteAddresses remote = LoadMetadata(config, source.agent);
    control.Send('M');
    control.Expect('F');
    Fill(b_memory, SeedB(0), EpochB(0));
    CheckCuda(cudaStreamSynchronize(0), "prepare fault payload");
    nixlXferReqH *request = nullptr;
    CheckNixl(source.agent.createXferReq(NIXL_WRITE,
                                         TransferList(b_memory),
                                         RemoteList(remote.b_data, kSmallBytes, remote.b_epoch, 1),
                                         "qp-progress-b",
                                         request,
                                         &source.options),
              "create fault request");
    const nixl_status_t post = source.agent.postXferReq(request);
    const nixl_status_t status =
        post == NIXL_IN_PROG ? Wait(source.agent, request, std::chrono::seconds(10)) : post;
    ASSERT_EQ(status, NIXL_ERR_BACKEND);
    ASSERT_EQ(source.agent.releaseXferReq(request), NIXL_SUCCESS);
    control.Send('E');
}

enum class PreErrorAction { PostTransfer, GenerateNotification };

void
TargetPreError(const Config &config) {
    constexpr uint64_t kSentinelRound = 0x5100;
    DeviceMemory a_memory(kSmallBytes, 0);
    DeviceMemory b_memory(kSmallBytes, 1);
    cudaStream_t a_verify_stream = nullptr;
    cudaStream_t b_verify_stream = nullptr;
    CheckCuda(cudaSetDevice(0), "select pre-error target A device");
    CheckCuda(cudaStreamCreateWithFlags(&a_verify_stream, cudaStreamNonBlocking),
              "create pre-error target A verify stream");
    CheckCuda(cudaSetDevice(1), "select pre-error target B device");
    CheckCuda(cudaStreamCreateWithFlags(&b_verify_stream, cudaStreamNonBlocking),
              "create pre-error target B verify stream");
    VerifyCounter a_counter(0);
    VerifyCounter b_counter(1);
    Fill(a_memory, SeedA(kSentinelRound), EpochA(kSentinelRound), a_verify_stream);
    Fill(b_memory, SeedB(kSentinelRound), EpochB(kSentinelRound), b_verify_stream);
    CheckCuda(cudaSetDevice(0), "select pre-error target A prepare device");
    CheckCuda(cudaStreamSynchronize(a_verify_stream), "prepare pre-error target A sentinel");
    CheckCuda(cudaSetDevice(1), "select pre-error target B prepare device");
    CheckCuda(cudaStreamSynchronize(b_verify_stream), "prepare pre-error target B sentinel");

    // All target GPU allocations, verification state, and streams precede the persistent engines.
    Endpoint a(config, "qp-progress-a", config.target_a_port, a_memory);
    Endpoint b(config, "qp-progress-b", config.target_b_port, b_memory);
    Control control(config);
    PublishMetadata(config, a, a_memory, b, b_memory);
    control.Send('R');
    control.Expect('D');
    ASSERT_EQ(
        a_counter.Verify(a_memory, SeedA(kSentinelRound), EpochA(kSentinelRound), a_verify_stream),
        0U);
    ASSERT_EQ(
        b_counter.Verify(b_memory, SeedB(kSentinelRound), EpochB(kSentinelRound), b_verify_stream),
        0U);
    control.Send('V');
    control.Expect('C');
    control.Send('K');
    CheckCuda(cudaSetDevice(0), "select pre-error target A destroy device");
    CheckCuda(cudaStreamSynchronize(a_verify_stream), "drain pre-error target A verify stream");
    CheckCuda(cudaStreamDestroy(a_verify_stream), "destroy pre-error target A verify stream");
    CheckCuda(cudaSetDevice(1), "select pre-error target B destroy device");
    CheckCuda(cudaStreamSynchronize(b_verify_stream), "drain pre-error target B verify stream");
    CheckCuda(cudaStreamDestroy(b_verify_stream), "destroy pre-error target B verify stream");
}

void
SourcePreError(const Config &config, PreErrorAction action) {
    constexpr uint64_t kSourceRound = 0x5200;
    DeviceMemory a_memory(kSmallBytes, 0);
    DeviceMemory b_memory(kSmallBytes, 0);
    cudaStream_t delayed_stream = nullptr;
    cudaStream_t fast_stream = nullptr;
    CheckCuda(cudaStreamCreateWithFlags(&delayed_stream, cudaStreamNonBlocking),
              "create pre-error delayed stream");
    CheckCuda(cudaStreamCreateWithFlags(&fast_stream, cudaStreamNonBlocking),
              "create pre-error fast stream");
    nixl_reg_dlist_t b_registration = Registration(b_memory);

    // Both attached streams and all source GPU memory precede the persistent engine.
    Endpoint source(config, "qp-progress-source", config.source_port, a_memory);
    CheckNixl(source.agent.registerMem(b_registration, &source.options),
              "register pre-error source B memory");
    Control control(config);
    control.Expect('R');
    const RemoteAddresses remote = LoadMetadata(config, source.agent);

    CheckCuda(cudaSetDevice(0), "select pre-error source device");
    Fill(b_memory, SeedB(kSourceRound), EpochB(kSourceRound), fast_stream);
    CheckCuda(cudaStreamSynchronize(fast_stream), "prepare pre-error B payload");

    // A's transfer kernel is queued behind a test-only delay. The transfer variant prepares B
    // before the deliberate CUDA launch-configuration error is left uncleared for
    // doca_kernel_write.
    DelayKernel<<<1, 1, 0, delayed_stream>>>(500000000ULL);
    CheckCuda(cudaGetLastError(), "launch pre-error delay kernel");
    Fill(a_memory, SeedA(kSourceRound), EpochA(kSourceRound), delayed_stream);

    nixlXferReqH *a_request = nullptr;
    nixlXferReqH *b_request = nullptr;
    const auto a_options = Attached(source.backend, delayed_stream, std::nullopt);
    const auto b_options = Attached(source.backend, fast_stream, std::nullopt);
    CheckNixl(source.agent.createXferReq(NIXL_WRITE,
                                         TransferList(a_memory),
                                         RemoteList(remote.a_data, kSmallBytes, remote.a_epoch, 0),
                                         "qp-progress-a",
                                         a_request,
                                         &a_options),
              "create pre-error A request");
    if (action == PreErrorAction::PostTransfer) {
        CheckNixl(
            source.agent.createXferReq(NIXL_WRITE,
                                       TransferList(b_memory),
                                       RemoteList(remote.b_data, kSmallBytes, remote.b_epoch, 1),
                                       "qp-progress-b",
                                       b_request,
                                       &b_options),
            "prepare pre-error B request");
    }
    const nixl_status_t a_post = source.agent.postXferReq(a_request);
    ASSERT_TRUE(a_post == NIXL_SUCCESS || a_post == NIXL_IN_PROG);

    CheckCuda(cudaGetLastError(), "clear CUDA error before injection");
    DelayKernel<<<0, 1>>>(1);
    ASSERT_EQ(cudaPeekAtLastError(), cudaErrorInvalidConfiguration);

    if (action == PreErrorAction::PostTransfer) {
        ASSERT_EQ(source.agent.postXferReq(b_request), NIXL_ERR_BACKEND);
        ASSERT_EQ(source.agent.releaseXferReq(b_request), NIXL_SUCCESS);
    } else {
        ASSERT_EQ(source.agent.genNotif("qp-progress-b", "pre-error", &source.options),
                  NIXL_ERR_BACKEND);
    }

    // CPU ownership is retained while A is still behind DelayKernel; only the GPU terminal
    // publication makes its slot releasable.
    ASSERT_EQ(source.agent.releaseXferReq(a_request), NIXL_ERR_REPOST_ACTIVE);
    CheckCuda(cudaStreamSynchronize(delayed_stream), "drain pre-error delayed stream");
    ASSERT_EQ(Wait(source.agent, a_request, std::chrono::seconds(10)), NIXL_ERR_BACKEND);
    ASSERT_EQ(source.agent.releaseXferReq(a_request), NIXL_SUCCESS);
    CheckCuda(cudaStreamSynchronize(fast_stream), "drain pre-error attached stream");

    control.Send('D');
    control.Expect('V');
    CheckNixl(source.agent.invalidateRemoteMD("qp-progress-a"),
              "invalidate pre-error target A metadata");
    CheckNixl(source.agent.invalidateRemoteMD("qp-progress-b"),
              "invalidate pre-error target B metadata");
    CheckNixl(source.agent.deregisterMem(b_registration, &source.options),
              "deregister pre-error source B memory");
    CheckCuda(cudaStreamDestroy(fast_stream), "destroy pre-error fast stream");
    CheckCuda(cudaStreamDestroy(delayed_stream), "destroy pre-error delayed stream");
    control.Send('C');
    control.Expect('K');
}

TEST(QpProgress, PerformanceMixed2MiBAnd4KiB) {
    const Config config = GetConfig();
    if (const auto problem = ConfigProblem(config, false)) {
        GTEST_SKIP() << *problem;
    }
    if (config.role == "source") {
        SourcePerformance(config);
    } else {
        TargetPerformance(config);
    }
}

TEST(QpProgress, OutstandingDescriptorsAndAttachedStreamOrder) {
    const Config config = GetConfig();
    if (const auto problem = ConfigProblem(config, false)) {
        GTEST_SKIP() << *problem;
    }
    if (config.role == "source") {
        SourceStress(config);
    } else {
        TargetStress(config);
    }
}

TEST(QpProgress, SingleQpReadWriteControl) {
    const Config config = GetConfig();
    if (const auto problem = ConfigProblem(config, false)) {
        GTEST_SKIP() << *problem;
    }
    if (config.role == "source") {
        SourceReadWriteControl(config);
    } else {
        TargetReadWriteControl(config);
    }
}

TEST(QpProgress, RemoteDeregisterReturnsBackendError) {
    const Config config = GetConfig();
    if (const auto problem = ConfigProblem(config, true)) {
        GTEST_SKIP() << *problem;
    }
    if (config.role == "source") {
        SourceFault(config);
    } else {
        TargetFault(config);
    }
}

TEST(QpProgress, CpuFatalLatchRejectsQueuedTransfer) {
    const Config config = GetConfig();
    if (const auto problem = ConfigProblem(config, false)) {
        GTEST_SKIP() << *problem;
    }
    if (config.role == "source") {
        SourcePreError(config, PreErrorAction::PostTransfer);
    } else {
        TargetPreError(config);
    }
}

TEST(QpProgress, CpuFatalLatchRejectsQueuedNotification) {
    const Config config = GetConfig();
    if (const auto problem = ConfigProblem(config, false)) {
        GTEST_SKIP() << *problem;
    }
    if (config.role == "source") {
        SourcePreError(config, PreErrorAction::GenerateNotification);
    } else {
        TargetPreError(config);
    }
}

} // namespace

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
