/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint64_t kMagic = 0x4e49584c4c535447ULL; // NIXLLSTG
constexpr uint32_t kStateInit = 0;
constexpr uint32_t kStateTargetReady = 1;
constexpr uint32_t kStateD2HReady = 2;
constexpr uint32_t kStateH2DDone = 3;
constexpr uint32_t kStateDone = 4;
constexpr uint32_t kStateError = 5;

struct SharedHeader {
    uint64_t magic;
    uint64_t bytes;
    uint32_t iters;
    uint32_t pattern;
    uint32_t state;
    uint32_t iter;
    uint32_t error;
    char errorMsg[160];
};

struct Args {
    std::string mode = "single";
    std::string path = "/tmp/nixl-local-vram-staging-probe.bin";
    size_t bytes = 256ULL * 1024ULL * 1024ULL;
    uint32_t iters = 4;
    int sourceGpu = 0;
    int targetGpu = 1;
    uint8_t pattern = 0x5a;
    int timeoutSec = 60;
    bool verify = true;
    bool keepFile = false;
};

[[noreturn]] void
throwError(const std::string &message) {
    throw std::runtime_error(message);
}

void
checkCuda(cudaError_t status, const char *label) {
    if (status != cudaSuccess) {
        throwError(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

uint32_t
loadU32(const uint32_t *value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void
storeU32(uint32_t *value, uint32_t new_value) {
    __atomic_store_n(value, new_value, __ATOMIC_RELEASE);
}

uint64_t
loadU64(const uint64_t *value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void
storeU64(uint64_t *value, uint64_t new_value) {
    __atomic_store_n(value, new_value, __ATOMIC_RELEASE);
}

size_t
pageSize() {
    const long value = sysconf(_SC_PAGESIZE);
    if (value <= 0) {
        throwError("sysconf(_SC_PAGESIZE) failed");
    }
    return static_cast<size_t>(value);
}

size_t
roundUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t
payloadOffset() {
    return roundUp(sizeof(SharedHeader), pageSize());
}

size_t
mappingBytes(size_t bytes) {
    return roundUp(payloadOffset() + bytes, pageSize());
}

size_t
parseSize(std::string_view text) {
    if (text.empty()) {
        throwError("empty size");
    }

    size_t pos = 0;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        ++pos;
    }
    if (pos == 0) {
        throwError("invalid size: " + std::string(text));
    }

    size_t value = std::strtoull(std::string(text.substr(0, pos)).c_str(), nullptr, 10);
    std::string suffix(text.substr(pos));
    for (char &ch : suffix) {
        if (ch >= 'a' && ch <= 'z') {
            ch = static_cast<char>(ch - 'a' + 'A');
        }
    }

    if (suffix.empty() || suffix == "B") {
        return value;
    }
    if (suffix == "K" || suffix == "KB" || suffix == "KIB") {
        return value * 1024ULL;
    }
    if (suffix == "M" || suffix == "MB" || suffix == "MIB") {
        return value * 1024ULL * 1024ULL;
    }
    if (suffix == "G" || suffix == "GB" || suffix == "GIB") {
        return value * 1024ULL * 1024ULL * 1024ULL;
    }

    throwError("unsupported size suffix: " + suffix);
}

bool
parseBool(std::string_view text) {
    return text == "1" || text == "true" || text == "TRUE" || text == "yes" || text == "on";
}

void
usage(const char *prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --mode single --source-gpu 0 --target-gpu 1 [options]\n"
        << "  " << prog << " --mode target --target-gpu 1 [options]\n"
        << "  " << prog << " --mode source --source-gpu 0 --target-gpu 1 [options]\n\n"
        << "Options:\n"
        << "  --path PATH          shared mmap file path, default /tmp/nixl-local-vram-staging-probe.bin\n"
        << "  --bytes SIZE         payload bytes, accepts K/M/G suffixes, default 256M\n"
        << "  --iters N            timed iterations, default 4\n"
        << "  --source-gpu N       source CUDA device ordinal, default 0\n"
        << "  --target-gpu N       target CUDA device ordinal, default 1\n"
        << "  --pattern N          byte pattern, default 90\n"
        << "  --verify 0|1         verify target GPU bytes after transfer, default 1\n"
        << "  --timeout SEC        cross-process wait timeout, default 60\n"
        << "  --keep-file 0|1      keep shared mmap file after owner exits, default 0\n";
}

Args
parseArgs(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throwError("missing value for " + key);
            }
            return argv[++i];
        };

        if (key == "--mode") {
            args.mode = need_value();
        } else if (key == "--path") {
            args.path = need_value();
        } else if (key == "--bytes") {
            args.bytes = parseSize(need_value());
        } else if (key == "--iters") {
            args.iters = static_cast<uint32_t>(std::stoul(need_value()));
        } else if (key == "--source-gpu") {
            args.sourceGpu = std::stoi(need_value());
        } else if (key == "--target-gpu") {
            args.targetGpu = std::stoi(need_value());
        } else if (key == "--pattern") {
            args.pattern = static_cast<uint8_t>(std::stoul(need_value()) & 0xffU);
        } else if (key == "--verify") {
            args.verify = parseBool(need_value());
        } else if (key == "--timeout") {
            args.timeoutSec = std::stoi(need_value());
        } else if (key == "--keep-file") {
            args.keepFile = parseBool(need_value());
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throwError("unknown argument: " + key);
        }
    }

    if (args.mode != "single" && args.mode != "source" && args.mode != "target") {
        throwError("mode must be single, source, or target");
    }
    if (args.bytes == 0 || args.iters == 0) {
        throwError("bytes and iters must be non-zero");
    }
    return args;
}

class SharedMapping {
public:
    SharedMapping(std::string path, size_t bytes, bool create, bool keep_file)
        : path_(std::move(path)),
          bytes_(bytes),
          totalBytes_(mappingBytes(bytes)),
          owner_(create),
          keepFile_(keep_file) {
        const int flags = create ? (O_CREAT | O_TRUNC | O_RDWR) : O_RDWR;
        fd_ = open(path_.c_str(), flags, 0600);
        if (fd_ < 0) {
            throwError("open failed for " + path_ + ": " + std::strerror(errno));
        }
        if (create && ftruncate(fd_, static_cast<off_t>(totalBytes_)) != 0) {
            throwError("ftruncate failed for " + path_ + ": " + std::strerror(errno));
        }
        void *mapped = mmap(nullptr, totalBytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            throwError("mmap failed for " + path_ + ": " + std::strerror(errno));
        }
        base_ = mapped;
    }

    ~SharedMapping() {
        if (payloadRegistered_) {
            cudaHostUnregister(payload());
        }
        if (base_) {
            munmap(base_, totalBytes_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
        if (owner_ && !keepFile_) {
            unlink(path_.c_str());
        }
    }

    SharedMapping(const SharedMapping &) = delete;
    SharedMapping &operator=(const SharedMapping &) = delete;

    [[nodiscard]] SharedHeader *
    header() const {
        return reinterpret_cast<SharedHeader *>(base_);
    }

    [[nodiscard]] void *
    payload() const {
        return static_cast<unsigned char *>(base_) + payloadOffset();
    }

    void
    registerPayload() {
        checkCuda(cudaHostRegister(payload(), bytes_, cudaHostRegisterDefault),
                  "cudaHostRegister(shared payload)");
        payloadRegistered_ = true;
    }

private:
    std::string path_;
    size_t bytes_;
    size_t totalBytes_;
    bool owner_;
    bool keepFile_;
    int fd_ = -1;
    void *base_ = nullptr;
    bool payloadRegistered_ = false;
};

void
waitForFileSize(const std::string &path, size_t size, int timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (true) {
        struct stat st {};
        if (stat(path.c_str(), &st) == 0 && static_cast<size_t>(st.st_size) >= size) {
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throwError("timed out waiting for shared mmap file " + path);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void
waitForState(const SharedHeader *header, uint32_t state, int timeout_sec, const char *label) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (true) {
        const uint32_t current = loadU32(&header->state);
        if (current == state) {
            return;
        }
        if (current == kStateError) {
            throwError(std::string("peer reported error while waiting for ") + label + ": " +
                       header->errorMsg);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throwError(std::string("timed out waiting for ") + label);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void
waitForMagic(const SharedHeader *header, int timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (true) {
        if (loadU64(&header->magic) == kMagic) {
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throwError("timed out waiting for shared header initialization");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void
setPeerError(SharedHeader *header, const std::string &message) {
    std::strncpy(header->errorMsg, message.c_str(), sizeof(header->errorMsg) - 1);
    header->errorMsg[sizeof(header->errorMsg) - 1] = '\0';
    storeU32(&header->error, 1);
    storeU32(&header->state, kStateError);
}

class CopyContext {
public:
    explicit CopyContext(int device)
        : device_(device) {
        checkCuda(cudaSetDevice(device_), "cudaSetDevice");
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
        checkCuda(cudaEventCreate(&start_), "cudaEventCreate start");
        checkCuda(cudaEventCreate(&stop_), "cudaEventCreate stop");
    }

    ~CopyContext() {
        cudaSetDevice(device_);
        if (start_) {
            cudaEventDestroy(start_);
        }
        if (stop_) {
            cudaEventDestroy(stop_);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    CopyContext(const CopyContext &) = delete;
    CopyContext &operator=(const CopyContext &) = delete;

    double
    copy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind, const char *label) {
        checkCuda(cudaSetDevice(device_), "cudaSetDevice copy");
        checkCuda(cudaEventRecord(start_, stream_), "cudaEventRecord start");
        checkCuda(cudaMemcpyAsync(dst, src, bytes, kind, stream_), label);
        checkCuda(cudaEventRecord(stop_, stream_), "cudaEventRecord stop");
        checkCuda(cudaEventSynchronize(stop_), "cudaEventSynchronize stop");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
        return static_cast<double>(ms) * 1000.0;
    }

private:
    int device_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

void *
allocDevice(int device, size_t bytes, uint8_t pattern) {
    checkCuda(cudaSetDevice(device), "cudaSetDevice alloc");
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    checkCuda(cudaMemset(ptr, pattern, bytes), "cudaMemset");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after memset");
    return ptr;
}

bool
verifyDevicePattern(int device, void *ptr, size_t bytes, uint8_t pattern) {
    checkCuda(cudaSetDevice(device), "cudaSetDevice verify");
    constexpr size_t kChunk = 64ULL * 1024ULL * 1024ULL;
    std::vector<unsigned char> host(std::min(kChunk, bytes));
    for (size_t offset = 0; offset < bytes; offset += host.size()) {
        const size_t chunk = std::min(host.size(), bytes - offset);
        checkCuda(cudaMemcpy(host.data(),
                             static_cast<unsigned char *>(ptr) + offset,
                             chunk,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy verify D2H");
        for (size_t i = 0; i < chunk; ++i) {
            if (host[i] != pattern) {
                std::cerr << "verify_mismatch_offset=" << (offset + i)
                          << " expected=" << static_cast<unsigned>(pattern)
                          << " actual=" << static_cast<unsigned>(host[i]) << "\n";
                return false;
            }
        }
    }
    return true;
}

double
gibPerSec(size_t bytes, double usec) {
    const double gib = static_cast<double>(bytes) / static_cast<double>(1024ULL * 1024ULL * 1024ULL);
    return gib / (usec / 1000000.0);
}

void
printPeerAccess(const Args &args) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return;
    }
    if (args.sourceGpu >= 0 && args.targetGpu >= 0 && args.sourceGpu < device_count &&
        args.targetGpu < device_count) {
        int can_access = 0;
        checkCuda(cudaDeviceCanAccessPeer(&can_access, args.sourceGpu, args.targetGpu),
                  "cudaDeviceCanAccessPeer");
        std::cout << "cuda_peer_access_source_to_target=" << can_access << "\n";
    }
}

void
runSingle(const Args &args) {
    printPeerAccess(args);

    SharedMapping mapping(args.path, args.bytes, true, args.keepFile);
    mapping.registerPayload();

    void *src = allocDevice(args.sourceGpu, args.bytes, args.pattern);
    void *dst = allocDevice(args.targetGpu, args.bytes, 0);
    CopyContext d2h(args.sourceGpu);
    CopyContext h2d(args.targetGpu);

    double d2h_us = 0.0;
    double h2d_us = 0.0;
    const auto wall_start = std::chrono::steady_clock::now();
    for (uint32_t iter = 0; iter < args.iters; ++iter) {
        d2h_us += d2h.copy(mapping.payload(),
                           src,
                           args.bytes,
                           cudaMemcpyDeviceToHost,
                           "cudaMemcpyAsync D2H");
        h2d_us += h2d.copy(dst,
                           mapping.payload(),
                           args.bytes,
                           cudaMemcpyHostToDevice,
                           "cudaMemcpyAsync H2D");
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_us =
        std::chrono::duration<double, std::micro>(wall_end - wall_start).count();

    bool verified = true;
    if (args.verify) {
        verified = verifyDevicePattern(args.targetGpu, dst, args.bytes, args.pattern);
    }

    std::cout << "mode=single\n"
              << "bytes=" << args.bytes << "\n"
              << "iters=" << args.iters << "\n"
              << "d2h_avg_us=" << (d2h_us / args.iters) << "\n"
              << "h2d_avg_us=" << (h2d_us / args.iters) << "\n"
              << "d2h_gib_per_sec=" << gibPerSec(args.bytes * args.iters, d2h_us) << "\n"
              << "h2d_gib_per_sec=" << gibPerSec(args.bytes * args.iters, h2d_us) << "\n"
              << "serial_copy_gib_per_sec="
              << gibPerSec(args.bytes * args.iters, d2h_us + h2d_us) << "\n"
              << "wall_gib_per_sec=" << gibPerSec(args.bytes * args.iters, wall_us) << "\n"
              << "verification=" << (verified ? "passed" : "failed") << "\n";

    cudaSetDevice(args.sourceGpu);
    cudaFree(src);
    cudaSetDevice(args.targetGpu);
    cudaFree(dst);

    if (!verified) {
        throwError("verification failed");
    }
}

void
runSource(const Args &args) {
    printPeerAccess(args);

    SharedMapping mapping(args.path, args.bytes, true, args.keepFile);
    std::memset(mapping.header(), 0, payloadOffset());
    SharedHeader *header = mapping.header();
    header->bytes = args.bytes;
    header->iters = args.iters;
    header->pattern = args.pattern;
    storeU64(&header->magic, kMagic);
    storeU32(&header->state, kStateInit);

    mapping.registerPayload();

    void *src = allocDevice(args.sourceGpu, args.bytes, args.pattern);
    CopyContext d2h(args.sourceGpu);

    waitForState(header, kStateTargetReady, args.timeoutSec, "target ready");

    double d2h_us = 0.0;
    for (uint32_t iter = 0; iter < args.iters; ++iter) {
        if (iter > 0) {
            waitForState(header, kStateH2DDone, args.timeoutSec, "previous H2D done");
        }
        storeU32(&header->iter, iter);
        d2h_us += d2h.copy(mapping.payload(),
                           src,
                           args.bytes,
                           cudaMemcpyDeviceToHost,
                           "cudaMemcpyAsync source D2H");
        storeU32(&header->state, kStateD2HReady);
    }

    waitForState(header, kStateH2DDone, args.timeoutSec, "final H2D done");
    storeU32(&header->state, kStateDone);

    std::cout << "mode=source\n"
              << "bytes=" << args.bytes << "\n"
              << "iters=" << args.iters << "\n"
              << "d2h_avg_us=" << (d2h_us / args.iters) << "\n"
              << "d2h_gib_per_sec=" << gibPerSec(args.bytes * args.iters, d2h_us) << "\n";

    cudaSetDevice(args.sourceGpu);
    cudaFree(src);
}

void
runTarget(const Args &args) {
    waitForFileSize(args.path, mappingBytes(args.bytes), args.timeoutSec);
    SharedMapping mapping(args.path, args.bytes, false, true);
    SharedHeader *header = mapping.header();
    waitForMagic(header, args.timeoutSec);

    if (header->bytes != args.bytes || header->iters != args.iters) {
        setPeerError(header, "target args do not match source header");
        throwError("target args do not match source header");
    }

    mapping.registerPayload();

    void *dst = allocDevice(args.targetGpu, args.bytes, 0);
    CopyContext h2d(args.targetGpu);

    storeU32(&header->state, kStateTargetReady);

    double h2d_us = 0.0;
    for (uint32_t iter = 0; iter < args.iters; ++iter) {
        waitForState(header, kStateD2HReady, args.timeoutSec, "source D2H ready");
        const uint32_t header_iter = loadU32(&header->iter);
        if (header_iter != iter) {
            setPeerError(header, "unexpected iter in shared header");
            throwError("unexpected iter in shared header");
        }
        h2d_us += h2d.copy(dst,
                           mapping.payload(),
                           args.bytes,
                           cudaMemcpyHostToDevice,
                           "cudaMemcpyAsync target H2D");
        storeU32(&header->state, kStateH2DDone);
    }

    bool verified = true;
    if (args.verify) {
        verified = verifyDevicePattern(args.targetGpu,
                                       dst,
                                       args.bytes,
                                       static_cast<uint8_t>(header->pattern & 0xffU));
    }
    if (!verified) {
        setPeerError(header, "target verification failed");
    }

    std::cout << "mode=target\n"
              << "bytes=" << args.bytes << "\n"
              << "iters=" << args.iters << "\n"
              << "h2d_avg_us=" << (h2d_us / args.iters) << "\n"
              << "h2d_gib_per_sec=" << gibPerSec(args.bytes * args.iters, h2d_us) << "\n"
              << "verification=" << (verified ? "passed" : "failed") << "\n";

    cudaSetDevice(args.targetGpu);
    cudaFree(dst);

    if (!verified) {
        throwError("verification failed");
    }
}

} // namespace

int
main(int argc, char **argv) {
    try {
        const Args args = parseArgs(argc, argv);
        if (args.mode == "single") {
            runSingle(args);
        } else if (args.mode == "source") {
            runSource(args);
        } else {
            runTarget(args);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "local_vram_staging_probe failed: " << e.what() << "\n";
        return 1;
    }
}
