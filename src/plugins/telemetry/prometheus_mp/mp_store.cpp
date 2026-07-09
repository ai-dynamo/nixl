/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mp_store.h"

#include "common/nixl_log.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace nixl::telemetry::mp {

namespace {

    // "NIXLMPS1" as a little-endian tag; changing the layout must change either this
    // or MP_STORE_SCHEMA_VERSION so stale-format files are rejected.
    constexpr uint64_t MP_STORE_MAGIC = 0x3153504d4c58494eULL;

    constexpr std::size_t MP_MAX_AGENT_NAME = 256;
    constexpr std::size_t MP_MAX_HOSTNAME = 128;
    constexpr std::size_t MP_MAX_DP_RANK = 64;

    // Fixed on-disk layout. Plain trivially-copyable POD operated on with __atomic
    // builtins (not std::atomic) so it is safe to memset/reinterpret over an mmap'd
    // region shared between processes. Field order keeps every uint64 8-byte aligned.
    struct mpStoreLayout {
        uint64_t magic;
        uint32_t schemaVersion;
        uint32_t slotCount;
        int64_t pid;
        uint64_t startTime;
        uint64_t lastUpdateNs;
        char agentName[MP_MAX_AGENT_NAME];
        char hostname[MP_MAX_HOSTNAME];
        char dpRank[MP_MAX_DP_RANK];
        uint64_t counters[MP_STORE_SLOT_COUNT];
        uint64_t gauges[MP_STORE_SLOT_COUNT];
    };

    [[nodiscard]] uint64_t
    nowNs() noexcept {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count());
    }

    void
    copyField(char *dst, std::size_t cap, const std::string &src, const char *what) {
        if (src.size() >= cap) {
            NIXL_WARN << "prometheus_mp: " << what << " '" << src << "' exceeds " << (cap - 1)
                      << " chars; truncating in telemetry store";
        }
        const std::size_t n = std::min(src.size(), cap - 1);
        std::memcpy(dst, src.data(), n);
        dst[n] = '\0';
    }

    [[nodiscard]] std::string
    readField(const char *src, std::size_t cap) {
        const std::size_t n = ::strnlen(src, cap);
        return std::string(src, n);
    }

} // namespace

uint64_t
readProcessStartTime(int64_t pid) {
    std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
    if (!stat.is_open()) {
        return 0;
    }
    std::string content((std::istreambuf_iterator<char>(stat)), std::istreambuf_iterator<char>());

    // comm (field 2) is wrapped in parentheses and may itself contain spaces or
    // ')', so split on the LAST ')': everything after it starts at field 3.
    const auto close = content.rfind(')');
    if (close == std::string::npos) {
        return 0;
    }

    std::istringstream rest(content.substr(close + 1));
    std::vector<std::string> tokens{std::istream_iterator<std::string>(rest),
                                    std::istream_iterator<std::string>()};
    // starttime is field 22; tokens[0] is field 3, so index 22 - 3 = 19.
    constexpr std::size_t kStartTimeIndex = 19;
    if (tokens.size() <= kStartTimeIndex) {
        return 0;
    }
    try {
        return static_cast<uint64_t>(std::stoull(tokens[kStartTimeIndex]));
    }
    catch (const std::exception &) {
        return 0;
    }
}

mpStoreWriter::mpStoreWriter(std::filesystem::path path,
                             const std::string &agent_name,
                             const std::string &hostname,
                             const std::string &dp_rank)
    : path_(std::move(path)),
      mappingSize_(sizeof(mpStoreLayout)) {
    const int fd = ::open(path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("prometheus_mp: cannot open telemetry store '" + path_.string() +
                                 "': " + std::strerror(errno));
    }

    if (::ftruncate(fd, static_cast<off_t>(mappingSize_)) != 0) {
        const std::string reason = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error("prometheus_mp: cannot size telemetry store '" + path_.string() +
                                 "': " + reason);
    }

    mapping_ = ::mmap(nullptr, mappingSize_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        throw std::runtime_error("prometheus_mp: cannot map telemetry store '" + path_.string() +
                                 "': " + std::strerror(errno));
    }

    auto *layout = static_cast<mpStoreLayout *>(mapping_);
    std::memset(layout, 0, mappingSize_);
    layout->schemaVersion = MP_STORE_SCHEMA_VERSION;
    layout->slotCount = static_cast<uint32_t>(MP_STORE_SLOT_COUNT);
    layout->pid = static_cast<int64_t>(::getpid());
    layout->startTime = readProcessStartTime(layout->pid);
    copyField(layout->agentName, MP_MAX_AGENT_NAME, agent_name, "agent name");
    copyField(layout->hostname, MP_MAX_HOSTNAME, hostname, "hostname");
    copyField(layout->dpRank, MP_MAX_DP_RANK, dp_rank, "dp_rank");
    __atomic_store_n(&layout->lastUpdateNs, nowNs(), __ATOMIC_RELEASE);
    // Publish the magic last so a concurrent reader never validates a
    // half-initialized header.
    __atomic_store_n(&layout->magic, MP_STORE_MAGIC, __ATOMIC_RELEASE);
}

mpStoreWriter::~mpStoreWriter() {
    if (mapping_ != nullptr) {
        ::munmap(mapping_, mappingSize_);
        mapping_ = nullptr;
    }
}

void
mpStoreWriter::touch() noexcept {
    auto *layout = static_cast<mpStoreLayout *>(mapping_);
    __atomic_store_n(&layout->lastUpdateNs, nowNs(), __ATOMIC_RELEASE);
}

void
mpStoreWriter::addCounter(nixl_telemetry_event_type_t type, uint64_t delta) noexcept {
    const auto idx = static_cast<std::size_t>(type);
    if (idx >= MP_STORE_SLOT_COUNT) {
        return;
    }
    auto *layout = static_cast<mpStoreLayout *>(mapping_);
    __atomic_fetch_add(&layout->counters[idx], delta, __ATOMIC_RELAXED);
    touch();
}

void
mpStoreWriter::setGauge(nixl_telemetry_event_type_t type, uint64_t value) noexcept {
    const auto idx = static_cast<std::size_t>(type);
    if (idx >= MP_STORE_SLOT_COUNT) {
        return;
    }
    auto *layout = static_cast<mpStoreLayout *>(mapping_);
    __atomic_store_n(&layout->gauges[idx], value, __ATOMIC_RELAXED);
    touch();
}

void
mpStoreWriter::refreshHeartbeat() noexcept {
    touch();
}

std::optional<mpStoreSnapshot>
readStoreSnapshot(const std::filesystem::path &path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        // Missing/unreadable file is not an error here (peer may have exited).
        return std::nullopt;
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0 || static_cast<std::size_t>(st.st_size) < sizeof(mpStoreLayout)) {
        // Too small: likely a file mid-creation by a peer. Skip quietly.
        ::close(fd);
        return std::nullopt;
    }

    void *mapping = ::mmap(nullptr, sizeof(mpStoreLayout), PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapping == MAP_FAILED) {
        NIXL_WARN << "prometheus_mp: cannot map telemetry store '" << path.string()
                  << "': " << std::strerror(errno);
        return std::nullopt;
    }

    const auto *layout = static_cast<const mpStoreLayout *>(mapping);

    if (__atomic_load_n(&layout->magic, __ATOMIC_ACQUIRE) != MP_STORE_MAGIC) {
        NIXL_WARN << "prometheus_mp: ignoring telemetry store '" << path.string()
                  << "' with bad magic";
        ::munmap(mapping, sizeof(mpStoreLayout));
        return std::nullopt;
    }
    if (layout->schemaVersion != MP_STORE_SCHEMA_VERSION ||
        layout->slotCount != MP_STORE_SLOT_COUNT) {
        NIXL_WARN << "prometheus_mp: ignoring telemetry store '" << path.string()
                  << "' with incompatible schema (version " << layout->schemaVersion << ", slots "
                  << layout->slotCount << ")";
        ::munmap(mapping, sizeof(mpStoreLayout));
        return std::nullopt;
    }

    mpStoreSnapshot snap;
    snap.pid = layout->pid;
    snap.startTime = layout->startTime;
    snap.lastUpdateNs = __atomic_load_n(&layout->lastUpdateNs, __ATOMIC_ACQUIRE);
    snap.agentName = readField(layout->agentName, MP_MAX_AGENT_NAME);
    snap.hostname = readField(layout->hostname, MP_MAX_HOSTNAME);
    snap.dpRank = readField(layout->dpRank, MP_MAX_DP_RANK);
    for (std::size_t i = 0; i < MP_STORE_SLOT_COUNT; ++i) {
        snap.counters[i] = __atomic_load_n(&layout->counters[i], __ATOMIC_RELAXED);
        snap.gauges[i] = __atomic_load_n(&layout->gauges[i], __ATOMIC_RELAXED);
    }

    ::munmap(mapping, sizeof(mpStoreLayout));
    return snap;
}

} // namespace nixl::telemetry::mp
