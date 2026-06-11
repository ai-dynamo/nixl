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

#include "ucx_backend.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <limits>
#include <future>
#include <memory>
#include <set>
#include <string_view>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include <asio.hpp>

namespace {

[[nodiscard]] std::string
trimAscii(std::string_view value) {
    size_t first = 0;
    while (first < value.size() &&
           std::isspace(static_cast<unsigned char>(value[first])) != 0) {
        ++first;
    }

    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1])) != 0) {
        --last;
    }

    return std::string(value.substr(first, last - first));
}

[[nodiscard]] std::vector<std::string>
splitCommaSeparatedDeviceList(std::string_view value) {
    std::vector<std::string> devices;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const size_t end = comma == std::string_view::npos ? value.size() : comma;
        std::string device = trimAscii(value.substr(start, end - start));
        if (!device.empty()) {
            devices.push_back(std::move(device));
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    return devices;
}

[[nodiscard]] std::vector<std::string>
getConfiguredUcxDevices(const nixl_b_params_t *custom_params) {
    if (!custom_params) {
        return {};
    }

    const auto ucx_devices_it = custom_params->find("ucx_devices");
    if (ucx_devices_it != custom_params->end() && !trimAscii(ucx_devices_it->second).empty()) {
        return splitCommaSeparatedDeviceList(ucx_devices_it->second);
    }

    const auto device_list_it = custom_params->find("device_list");
    if (device_list_it != custom_params->end() && !trimAscii(device_list_it->second).empty()) {
        return splitCommaSeparatedDeviceList(device_list_it->second);
    }

    return {};
}

[[nodiscard]] uint64_t
profileNowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

[[nodiscard]] bool
stagingProfileEnabled() {
    const char *value = std::getenv("NIXL_UCX_STAGING_PROFILE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

[[nodiscard]] bool
localStagingForceAttachFail() {
    static const std::string env_name(nixl_ucx_local_staging_force_attach_fail_env_name);
    const char *value = std::getenv(env_name.c_str());
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

[[nodiscard]] std::string
readFirstLine(const char *path) {
    std::ifstream file(path);
    std::string line;
    if (!file.good() || !std::getline(file, line)) {
        return {};
    }
    return trimAscii(line);
}

[[nodiscard]] size_t
pageSize() {
    const long value = sysconf(_SC_PAGESIZE);
    return value > 0 ? static_cast<size_t>(value) : 4096;
}

[[nodiscard]] size_t
roundUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

[[nodiscard]] std::string
sanitizePathComponent(std::string value) {
    for (char &ch : value) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.';
        if (!ok) {
            ch = '_';
        }
    }
    return value;
}

[[nodiscard]] std::string
newLocalSharedRegionCookie() {
    std::string uuid = readFirstLine("/proc/sys/kernel/random/uuid");
    if (!uuid.empty()) {
        return sanitizePathComponent(std::move(uuid));
    }
    return std::to_string(getpid()) + "-" + std::to_string(profileNowUs());
}

[[nodiscard]] std::string
localHostId() {
    if (const char *env = std::getenv("NIXL_UCX_LOCAL_STAGING_HOST_ID");
        env != nullptr && env[0] != '\0') {
        return env;
    }

    if (std::string boot_id = readFirstLine("/proc/sys/kernel/random/boot_id");
        !boot_id.empty()) {
        return boot_id;
    }

    if (std::string machine_id = readFirstLine("/etc/machine-id");
        !machine_id.empty()) {
        return machine_id;
    }

    char host[256] = {};
    if (gethostname(host, sizeof(host) - 1) == 0 && host[0] != '\0') {
        return host;
    }
    return "unknown";
}

[[nodiscard]] std::optional<std::string>
canonicalPath(const std::string &path) {
    char *resolved = realpath(path.c_str(), nullptr);
    if (resolved == nullptr) {
        return std::nullopt;
    }
    std::unique_ptr<char, decltype(&std::free)> holder(resolved, &std::free);
    return std::string(holder.get());
}

[[nodiscard]] bool
pathIsUnderDirectory(const std::string &path, const std::string &directory) {
    if (directory.empty() || path.empty()) {
        return false;
    }

    const std::string dir = directory.back() == '/' ? directory.substr(0, directory.size() - 1) :
                                                      directory;
    if (dir.empty()) {
        return false;
    }
    return path == dir || (path.size() > dir.size() && path.compare(0, dir.size(), dir) == 0 &&
                           path[dir.size()] == '/');
}

[[nodiscard]] std::string
envString(std::string_view name, const std::string &default_value) {
    const char *value = std::getenv(std::string(name).c_str());
    return value != nullptr && value[0] != '\0' ? std::string(value) : default_value;
}

[[nodiscard]] nixl_status_t
ensureDirectory(const std::string &path) {
    if (path.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (mkdir(path.c_str(), 0700) == 0 || errno == EEXIST) {
        return NIXL_SUCCESS;
    }

    NIXL_ERROR << "Failed to create local staging directory " << path << ": "
               << std::strerror(errno);
    return NIXL_ERR_BACKEND;
}

} // namespace

/****************************************
 * Backend request management
*****************************************/

class nixlUcxBackendReqH : public nixlBackendReqH {
private:
    std::set<ucx_connection_ptr_t> connections_;
    std::vector<nixlUcxReq> requests_;
    nixlUcxWorker *worker_;
    size_t workerId_;

    [[nodiscard]] nixl_status_t
    checkConnection(const nixl_status_t status = NIXL_SUCCESS) const {
        NIXL_ASSERT(!connections_.empty());
        for (const auto &conn : connections_) {
            const nixl_status_t conn_status = conn->getEp(workerId_)->checkTxState();
            if (conn_status != NIXL_SUCCESS) {
                return conn_status;
            }
        }
        return status;
    }

protected:
    void
    setWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(worker_ == nullptr || worker == nullptr);
        worker_ = worker;
        workerId_ = worker_id;
    }

public:
    // Notification to be sent after completion of all requests
    struct Notif {
        const std::string agent;
        const nixl_blob_t payload;

        Notif(const std::string &remote_agent, const nixl_blob_t &msg)
            : agent(remote_agent),
              payload(msg) {}
    };

    std::optional<Notif> notif;

    nixlUcxBackendReqH(nixlUcxWorker *worker, size_t worker_id)
        : worker_(worker),
          workerId_(worker_id) {}

    void
    reserve(size_t size) {
        requests_.reserve(size);
        NIXL_ASSERT(connections_.empty());
    }

    [[nodiscard]] nixl_status_t
    append(nixl_status_t status, nixlUcxReq req, const ucx_connection_ptr_t &conn) {
        switch (status) {
        case NIXL_IN_PROG:
            requests_.push_back(req);
            connections_.insert(conn);
            break;
        case NIXL_SUCCESS:
            connections_.insert(conn);
            break;
        default:
            // Error. Release all previously initiated ops and exit:
            release();
            return status;
        }
        return NIXL_SUCCESS;
    }

    [[nodiscard]] const std::set<ucx_connection_ptr_t> &
    getConnections() const noexcept {
        return connections_;
    }

    [[nodiscard]] virtual bool
    isComposite() const noexcept {
        return false;
    }

    virtual void
    release() {
        // TODO: Error log: uncompleted requests found! Cancelling ...
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (ret == NIXL_IN_PROG) {
                // TODO: Need process this properly.
                // it may not be enough to cancel UCX request
                worker_->reqCancel(req);
            }
            worker_->reqRelease(req);
        }
        requests_.clear();
        connections_.clear();
    }

    [[nodiscard]] virtual nixl_status_t
    status() {
        if (requests_.empty()) {
            /* No pending transmissions */
            connections_.clear();
            return NIXL_SUCCESS;
        }

        worker_->progressLoop();

        /* If last request is incomplete, return NIXL_IN_PROG early without
         * checking other requests */
        nixlUcxReq req = requests_.back();
        const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
        if (ret == NIXL_IN_PROG) {
            return NIXL_IN_PROG;
        } else if (ret != NIXL_SUCCESS) {
            return checkConnection(ret);
        }

        /* Last request completed successfully, all the others must be in the
         * same state. TODO: remove extra checks? */
        size_t incomplete_reqs = 0;
        nixl_status_t out_ret = NIXL_SUCCESS;
        for (nixlUcxReq req : requests_) {
            const nixl_status_t ret = nixl::ucx::ucsToNixlStatus(ucp_request_check_status(req));
            if (__builtin_expect(ret == NIXL_SUCCESS, 0)) {
                worker_->reqRelease(req);
            } else if (ret == NIXL_IN_PROG) {
                if (out_ret == NIXL_SUCCESS) {
                    out_ret = NIXL_IN_PROG;
                }
                requests_[incomplete_reqs++] = req;
            } else {
                // Any other ret value is ERR and will be returned
                out_ret = checkConnection(ret);
            }
        }

        requests_.resize(incomplete_reqs);
        if (requests_.empty()) {
            connections_.clear();
        }
        return out_ret;
    }

    [[nodiscard]] nixlUcxWorker *
    getWorker() const noexcept {
        return worker_;
    }

    [[nodiscard]] size_t
    getWorkerId() const noexcept {
        return workerId_;
    }
};

/****************************************
 * Staged VRAM metadata
*****************************************/

namespace {

constexpr std::string_view kUcxStagedMagic = "NIXL_UCX_STAGED_V1";

struct nixlUcxStagedSlot {
    void *hostAddr = nullptr;
    size_t size = 0;
    nixlUcxMem mem;
    nixl_blob_t rkeyStr;
    bool ucxRegistered = false;
};

enum class nixlUcxStagedSlotState {
    FREE,
    LOCAL_D2H,
    REMOTE_RESERVED,
    REMOTE_H2D,
    ERROR,
};

struct nixlUcxStagedSlotLease {
    nixlUcxStagedSlotState state = nixlUcxStagedSlotState::FREE;
    std::string ownerAgent;
    uint64_t transferId = 0;
    uint64_t chunkId = 0;
    uint64_t leaseId = 0;
    uintptr_t gpuAddr = 0;
    uint64_t gpuDev = 0;
    size_t size = 0;
    uint64_t grantedUs = 0;

    void
    reset() {
        state = nixlUcxStagedSlotState::FREE;
        ownerAgent.clear();
        transferId = 0;
        chunkId = 0;
        leaseId = 0;
        gpuAddr = 0;
        gpuDev = 0;
        size = 0;
        grantedUs = 0;
    }
};

struct nixlUcxStagedSlotGrant {
    uint64_t slotId = 0;
    uint64_t leaseId = 0;
    nixl_status_t status = NIXL_IN_PROG;
};

struct nixlUcxStagedReadyLease {
    void *hostAddr = nullptr;
    uint64_t slotId = 0;
    uint64_t leaseId = 0;
};

struct nixlUcxStagedProfile {
    uint64_t slotReqSent = 0;
    uint64_t slotGrantSuccess = 0;
    uint64_t slotGrantInProg = 0;
    uint64_t localSlotMiss = 0;
    uint64_t remoteWindowMiss = 0;
    uint64_t staleGrantReleases = 0;
    uint64_t localSharedChunks = 0;
    uint64_t localSharedBytes = 0;
    uint64_t localSharedAckErrors = 0;
    uint64_t localSharedFallbacks = 0;
    uint64_t rdmaWritePosted = 0;
    uint64_t flushPosted = 0;
    uint64_t readySent = 0;
    uint64_t ackReceived = 0;
    uint64_t bytes = 0;
    uint64_t startUs = 0;
    uint64_t grantWaitUs = 0;
    uint64_t d2hUs = 0;
    uint64_t rdmaFlushWaitUs = 0;
    uint64_t readyWaitUs = 0;
    uint64_t ackWaitUs = 0;
};

[[nodiscard]] bool
rangeCovers(uintptr_t base, size_t len, uintptr_t addr, size_t size) {
    return addr >= base && size <= len && (addr - base) <= (len - size);
}

class nixlUcxStagedPrivateMetadata : public nixlBackendMD {
public:
    explicit nixlUcxStagedPrivateMetadata(const nixlBlobDesc &mem)
        : nixlBackendMD(true),
          gpuBase(mem.addr),
          gpuLen(mem.len),
          gpuDevId(mem.devId) {}

    uintptr_t gpuBase;
    size_t gpuLen;
    uint64_t gpuDevId;
    size_t slotSize = 0;
    uint64_t leaseTimeoutUs = 0;
    std::vector<nixlUcxStagedSlot> slots;
    std::mutex slotMutex;
    std::vector<nixlUcxStagedSlotLease> slotLeases;
    uint64_t nextLeaseId = 1;
    bool localSharedSlots = false;
    uint64_t localSharedRegionId = 0;
    std::string localSharedRegionCookie;
    std::string hostId;
    std::string sharedPath;
    void *sharedBase = nullptr;
    size_t sharedMappingSize = 0;
    int sharedFd = -1;
    bool sharedHostRegistered = false;
    std::vector<uint64_t> localSlotGenerations;
    bool unlinkSharedPath = false;

    [[nodiscard]] std::optional<size_t>
    acquireSlot() {
        const std::lock_guard lock(slotMutex);
        for (size_t i = 0; i < slotLeases.size(); ++i) {
            if (slotLeases[i].state == nixlUcxStagedSlotState::FREE) {
                slotLeases[i].reset();
                slotLeases[i].state = nixlUcxStagedSlotState::LOCAL_D2H;
                if (i < localSlotGenerations.size()) {
                    ++localSlotGenerations[i];
                    if (localSlotGenerations[i] == 0) {
                        localSlotGenerations[i] = 1;
                    }
                }
                return i;
            }
        }
        return std::nullopt;
    }

    void
    releaseSlot(size_t slot_id) {
        const std::lock_guard lock(slotMutex);
        if (slot_id < slotLeases.size() &&
            slotLeases[slot_id].state == nixlUcxStagedSlotState::LOCAL_D2H) {
            slotLeases[slot_id].reset();
        }
    }

    [[nodiscard]] bool
    hasActiveSlots() {
        const std::lock_guard lock(slotMutex);
        return std::any_of(slotLeases.begin(), slotLeases.end(), [](const auto &lease) {
            return lease.state != nixlUcxStagedSlotState::FREE;
        });
    }

    [[nodiscard]] uint64_t
    slotGeneration(size_t slot_id) {
        const std::lock_guard lock(slotMutex);
        if (slot_id >= localSlotGenerations.size()) {
            return 0;
        }
        return localSlotGenerations[slot_id];
    }

    [[nodiscard]] nixlUcxStagedSlotGrant
    reserveRemoteSlot(const std::string &owner_agent,
                      uint64_t transfer_id,
                      uint64_t chunk_id,
                      uintptr_t gpu_addr,
                      uint64_t gpu_dev,
                      size_t size) {
        const std::lock_guard lock(slotMutex);
        if (size == 0 || size > slotSize || gpu_dev != gpuDevId ||
            !rangeCovers(gpuBase, gpuLen, gpu_addr, size)) {
            return {.status = NIXL_ERR_INVALID_PARAM};
        }

        const auto grant_slot = [&](size_t i) -> nixlUcxStagedSlotGrant {
            auto &lease = slotLeases[i];
            lease.reset();
            lease.state = nixlUcxStagedSlotState::REMOTE_RESERVED;
            lease.ownerAgent = owner_agent;
            lease.transferId = transfer_id;
            lease.chunkId = chunk_id;
            lease.leaseId = nextLeaseId++;
            lease.gpuAddr = gpu_addr;
            lease.gpuDev = gpu_dev;
            lease.size = size;
            lease.grantedUs = profileNowUs();
            return {.slotId = static_cast<uint64_t>(i),
                    .leaseId = lease.leaseId,
                    .status = NIXL_SUCCESS};
        };

        for (size_t i = 0; i < slotLeases.size(); ++i) {
            if (slotLeases[i].state == nixlUcxStagedSlotState::FREE) {
                return grant_slot(i);
            }
        }

        // No free slot: lazily reclaim leases a dead initiator can no longer return.
        // ERROR slots are terminal (the failed H2D already consumed the data) and a
        // REMOTE_RESERVED lease past the timeout belongs to an initiator that never
        // sent READY or RELEASE. REMOTE_H2D is never reclaimed: a local copy is
        // still reading the slot. The timeout must stay much larger than any single
        // RDMA write so a late write from a slow-but-alive initiator cannot land in
        // a re-granted slot.
        if (leaseTimeoutUs != 0) {
            const uint64_t now_us = profileNowUs();
            for (size_t i = 0; i < slotLeases.size(); ++i) {
                const auto &lease = slotLeases[i];
                const bool error_slot = lease.state == nixlUcxStagedSlotState::ERROR;
                const bool expired_reserved =
                    lease.state == nixlUcxStagedSlotState::REMOTE_RESERVED &&
                    lease.grantedUs != 0 && now_us - lease.grantedUs >= leaseTimeoutUs;
                if (!error_slot && !expired_reserved) {
                    continue;
                }
                NIXL_WARN << "Reclaiming UCX staged slot " << i
                          << (error_slot ? " in ERROR state" : " with expired lease")
                          << " owner=" << lease.ownerAgent
                          << " transfer_id=" << lease.transferId
                          << " chunk_id=" << lease.chunkId
                          << " lease_id=" << lease.leaseId;
                return grant_slot(i);
            }
        }

        return {.status = NIXL_IN_PROG};
    }

    [[nodiscard]] nixl_status_t
    beginRemoteH2D(const std::string &owner_agent,
                   uint64_t transfer_id,
                   uint64_t chunk_id,
                   uint64_t slot_id,
                   uint64_t lease_id,
                   uintptr_t gpu_addr,
                   uint64_t gpu_dev,
                   size_t size,
                   nixlUcxStagedReadyLease &ready) {
        const std::lock_guard lock(slotMutex);
        if (slot_id >= slotLeases.size() || slot_id >= slots.size() || size == 0 ||
            size > slots[slot_id].size) {
            return NIXL_ERR_INVALID_PARAM;
        }

        auto &lease = slotLeases[slot_id];
        if (lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED ||
            lease.ownerAgent != owner_agent || lease.transferId != transfer_id ||
            lease.chunkId != chunk_id || lease.leaseId != lease_id || lease.gpuAddr != gpu_addr ||
            lease.gpuDev != gpu_dev || lease.size != size) {
            return NIXL_ERR_MISMATCH;
        }

        lease.state = nixlUcxStagedSlotState::REMOTE_H2D;
        ready.hostAddr = slots[slot_id].hostAddr;
        ready.slotId = slot_id;
        ready.leaseId = lease_id;
        return NIXL_SUCCESS;
    }

    void
    finishRemoteLease(uint64_t slot_id, uint64_t lease_id, nixl_status_t status) {
        const std::lock_guard lock(slotMutex);
        if (slot_id >= slotLeases.size()) {
            return;
        }

        auto &lease = slotLeases[slot_id];
        if (lease.leaseId != lease_id) {
            return;
        }

        if (status == NIXL_SUCCESS) {
            lease.reset();
        } else {
            lease.state = nixlUcxStagedSlotState::ERROR;
        }
    }

    // Releases every lease the given remote agent still holds, except REMOTE_H2D
    // (a local copy is reading the slot; finishRemoteLease will settle it).
    // Used when the peer is disconnected and can no longer return its leases.
    size_t
    releaseRemoteLeasesForOwner(const std::string &owner_agent) {
        const std::lock_guard lock(slotMutex);
        size_t released = 0;
        for (auto &lease : slotLeases) {
            if (lease.ownerAgent != owner_agent) {
                continue;
            }
            if (lease.state == nixlUcxStagedSlotState::REMOTE_RESERVED ||
                lease.state == nixlUcxStagedSlotState::ERROR) {
                lease.reset();
                ++released;
            }
        }
        return released;
    }

    [[nodiscard]] bool
    releaseRemoteLease(const std::string &owner_agent,
                       uint64_t transfer_id,
                       uint64_t chunk_id,
                       uint64_t slot_id,
                       uint64_t lease_id) {
        const std::lock_guard lock(slotMutex);
        if (slot_id >= slotLeases.size()) {
            return false;
        }

        auto &lease = slotLeases[slot_id];
        if (lease.ownerAgent != owner_agent || lease.transferId != transfer_id ||
            lease.chunkId != chunk_id || lease.leaseId != lease_id ||
            lease.state != nixlUcxStagedSlotState::REMOTE_RESERVED) {
            return false;
        }

        lease.reset();
        return true;
    }
};

class nixlUcxStagedPublicMetadata : public nixlBackendMD {
public:
    nixlUcxStagedPublicMetadata(const ucx_connection_ptr_t &conn,
                                std::string agent,
                                uintptr_t gpu_base,
                                size_t gpu_len,
                                uint64_t gpu_dev_id,
                                std::string host_id,
                                size_t slot_size,
                                size_t slot_window_limit,
                                bool local_shared_slots,
                                uint64_t local_shared_region_id,
                                std::string local_shared_region_cookie,
                                std::string local_shared_path,
                                size_t local_shared_mapping_size,
                                std::vector<uintptr_t> &&slot_addrs,
                                std::vector<std::vector<nixl::ucx::rkey>> &&slot_rkeys)
        : nixlBackendMD(false),
          conn(conn),
          agent(std::move(agent)),
          gpuBase(gpu_base),
          gpuLen(gpu_len),
          gpuDevId(gpu_dev_id),
          hostId(std::move(host_id)),
          slotSize(slot_size),
          slotWindowLimit(slot_window_limit == 0 ? std::max<size_t>(1, slot_addrs.size() * 4) :
                                                    slot_window_limit),
          localSharedSlots(local_shared_slots),
          localSharedRegionId(local_shared_region_id),
          localSharedRegionCookie(std::move(local_shared_region_cookie)),
          localSharedPath(std::move(local_shared_path)),
          localSharedMappingSize(local_shared_mapping_size),
          slotAddrs(std::move(slot_addrs)),
          slotRkeys(std::move(slot_rkeys)) {}

    [[nodiscard]] bool
    tryAcquireSlotWindow() {
        const size_t limit = slotWindowLimit;
        size_t in_use = slotWindowInUse.load(std::memory_order_relaxed);
        while (in_use < limit) {
            if (slotWindowInUse.compare_exchange_weak(in_use,
                                                      in_use + 1,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

    void
    releaseSlotWindow() {
        const size_t previous = slotWindowInUse.fetch_sub(1, std::memory_order_acq_rel);
        if (previous == 0) {
            slotWindowInUse.fetch_add(1, std::memory_order_relaxed);
            NIXL_WARN << "UCX staged remote slot window release underflow";
        }
    }

    const ucx_connection_ptr_t conn;
    const std::string agent;
    const uintptr_t gpuBase;
    const size_t gpuLen;
    const uint64_t gpuDevId;
    const std::string hostId;
    const size_t slotSize;
    const size_t slotWindowLimit;
    const bool localSharedSlots;
    const uint64_t localSharedRegionId;
    const std::string localSharedRegionCookie;
    const std::string localSharedPath;
    const size_t localSharedMappingSize;
    const std::vector<uintptr_t> slotAddrs;
    const std::vector<std::vector<nixl::ucx::rkey>> slotRkeys;
    std::atomic<size_t> slotWindowInUse{0};
};

[[nodiscard]] nixl_blob_t
serializeStagedMetadata(const nixlUcxStagedPrivateMetadata &metadata) {
    nixlSerDes ser_des;
    const std::string magic(kUcxStagedMagic);
    const size_t slot_count = metadata.slots.size();

    ser_des.addStr("magic", magic);
    ser_des.addBuf("gpu_base", &metadata.gpuBase, sizeof(metadata.gpuBase));
    ser_des.addBuf("gpu_len", &metadata.gpuLen, sizeof(metadata.gpuLen));
    ser_des.addBuf("gpu_dev", &metadata.gpuDevId, sizeof(metadata.gpuDevId));
    ser_des.addBuf("slot_size", &metadata.slotSize, sizeof(metadata.slotSize));
    ser_des.addBuf("slot_count", &slot_count, sizeof(slot_count));
    ser_des.addStr("host_id", metadata.hostId);
    ser_des.addBuf("local_shared_slots",
                   &metadata.localSharedSlots,
                   sizeof(metadata.localSharedSlots));
    ser_des.addBuf("local_shared_region_id",
                   &metadata.localSharedRegionId,
                   sizeof(metadata.localSharedRegionId));
    if (!metadata.localSharedRegionCookie.empty()) {
        ser_des.addStr("local_shared_region_cookie", metadata.localSharedRegionCookie);
    }
    ser_des.addStr("local_shared_path", metadata.sharedPath);
    ser_des.addBuf("local_shared_mapping_size",
                   &metadata.sharedMappingSize,
                   sizeof(metadata.sharedMappingSize));

    for (const auto &slot : metadata.slots) {
        const uintptr_t host_addr = reinterpret_cast<uintptr_t>(slot.hostAddr);
        ser_des.addBuf("slot_addr", &host_addr, sizeof(host_addr));
        ser_des.addStr("slot_rkey", slot.rkeyStr);
    }

    return ser_des.exportStr();
}

[[nodiscard]] bool
isStagedMetadataBlob(const nixl_blob_t &blob) {
    return blob.compare(0, std::string_view("nixlSerDes|").size(), "nixlSerDes|") == 0;
}

[[nodiscard]] nixl_status_t
cudaSetDeviceForCopy(uint64_t dev_id, int &previous_device) {
    previous_device = -1;
    cudaError_t cuda_ret = cudaGetDevice(&previous_device);
    if (cuda_ret != cudaSuccess) {
        NIXL_ERROR << "cudaGetDevice failed: " << cudaGetErrorString(cuda_ret);
        return NIXL_ERR_BACKEND;
    }

    cuda_ret = cudaSetDevice(static_cast<int>(dev_id));
    if (cuda_ret != cudaSuccess) {
        NIXL_ERROR << "cudaSetDevice(" << dev_id << ") failed: " << cudaGetErrorString(cuda_ret);
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

void
cudaRestoreDevice(int previous_device) {
    if (previous_device < 0) {
        return;
    }

    const cudaError_t cuda_ret = cudaSetDevice(previous_device);
    if (cuda_ret != cudaSuccess) {
        NIXL_WARN << "cudaSetDevice restore to " << previous_device
                  << " failed: " << cudaGetErrorString(cuda_ret);
    }
}

struct nixlUcxStagedChunk {
    enum class State {
        PENDING,
        LOCAL_D2H_POSTED,
        LOCAL_READY,
        SLOT_REQ_POSTED,
        WAIT_SLOT_GRANT,
        RDMA_POSTED,
        FLUSH_POSTED,
        READY_AM_POSTED,
        WAIT_ACK,
        ACKED,
        FAILED,
    };

    nixlUcxStagedChunk(uint64_t chunk_id,
                       uintptr_t local_gpu_addr,
                       uint64_t local_gpu_dev,
                       uintptr_t remote_gpu_addr,
                       uint64_t remote_gpu_dev,
                       size_t chunk_size,
                       nixlUcxStagedPrivateMetadata *local_metadata,
                       nixlUcxStagedPublicMetadata *remote_metadata)
        : id(chunk_id),
          localGpuAddr(local_gpu_addr),
          localGpuDev(local_gpu_dev),
          remoteGpuAddr(remote_gpu_addr),
          remoteGpuDev(remote_gpu_dev),
          size(chunk_size),
          localMetadata(local_metadata),
          remoteMetadata(remote_metadata) {}

    const uint64_t id;
    const uintptr_t localGpuAddr;
    const uint64_t localGpuDev;
    const uintptr_t remoteGpuAddr;
    const uint64_t remoteGpuDev;
    const size_t size;
    nixlUcxStagedPrivateMetadata *localMetadata = nullptr;
    nixlUcxStagedPublicMetadata *remoteMetadata = nullptr;
    size_t localSlotId = 0;
    bool localSlotHeld = false;
    uint64_t remoteSlotId = 0;
    // Written by markGrant (progress thread) and by the local-ready/fallback paths
    // (app thread), read by markAck (progress thread): must be atomic.
    std::atomic<uint64_t> leaseId{0};
    bool remoteSlotHeld = false;
    bool remoteWindowHeld = false;
    bool localSharedWrite = false;
    bool localFallbackAttempted = false;
    State state = State::PENDING;
    std::unique_ptr<nixlUcxBackendReqH> req;
    std::atomic<bool> grantArrived{false};
    std::atomic<nixl_status_t> grantStatus{NIXL_IN_PROG};
    std::atomic<nixl_status_t> ackStatus{NIXL_IN_PROG};
    uint64_t slotReqPostedUs = 0;
    std::atomic<uint64_t> grantArrivedUs{0};
    cudaEvent_t d2hEvent = nullptr;
    uint64_t d2hPostedUs = 0;
    uint64_t rdmaPostedUs = 0;
    uint64_t readyPostedUs = 0;
    uint64_t ackWaitStartUs = 0;
    std::atomic<uint64_t> ackArrivedUs{0};
};

struct nixlUcxStagedFlushBatch {
    std::unique_ptr<nixlUcxBackendReqH> req;
    std::vector<nixlUcxStagedChunk *> chunks;
    nixlUcxStagedPublicMetadata *remoteMetadata = nullptr;
    uint64_t postedUs = 0;
};

class nixlUcxStagedBackendReqH : public nixlUcxBackendReqH {
public:
    struct SourceD2HStreams {
        std::vector<cudaStream_t> streams;
        size_t next = 0;
    };

    enum class State {
        INIT,
        RUNNING,
        USER_NOTIF_POSTED,
        COMPLETE,
        FAILED,
    };

    nixlUcxStagedBackendReqH(nixlUcxWorker *worker, size_t worker_id, uint64_t transfer_id)
        : nixlUcxBackendReqH(worker, worker_id),
          transferId(transfer_id) {}

    void
    releaseLocalSlot(nixlUcxStagedChunk &chunk) {
        if (chunk.d2hEvent) {
            cudaEventSynchronize(chunk.d2hEvent);
            cudaEventDestroy(chunk.d2hEvent);
            chunk.d2hEvent = nullptr;
        }
        if (chunk.localMetadata && chunk.localSlotHeld) {
            chunk.localMetadata->releaseSlot(chunk.localSlotId);
            chunk.localSlotHeld = false;
        }
    }

    void
    releaseRemoteWindow(nixlUcxStagedChunk &chunk) {
        if (chunk.remoteMetadata && chunk.remoteWindowHeld) {
            chunk.remoteMetadata->releaseSlotWindow();
            chunk.remoteWindowHeld = false;
        }
    }

    void
    releaseRemoteSlot(nixlUcxStagedChunk &chunk) {
        releaseRemoteWindow(chunk);
        chunk.remoteSlotHeld = false;
    }

    void
    releaseChunkSlots(nixlUcxStagedChunk &chunk) {
        releaseLocalSlot(chunk);
        releaseRemoteSlot(chunk);
    }

    void
    releaseAllChunkSlots() {
        for (const auto &chunk : chunks) {
            releaseChunkSlots(*chunk);
        }
    }

    // Returns false when the grant could not be bound to a chunk; a successful
    // grant left unbound must be released by the caller or the target slot leaks.
    [[nodiscard]] bool
    markGrant(uint64_t chunk_id, uint64_t slot_id, uint64_t lease_id, nixl_status_t status) {
        if (chunk_id >= chunks.size()) {
            NIXL_WARN << "Received UCX staged slot grant for out-of-range chunk id " << chunk_id
                      << " transfer_id=" << transferId << " chunks=" << chunks.size();
            return false;
        }

        auto &chunk = *chunks[chunk_id];
        chunk.remoteSlotId = slot_id;
        chunk.leaseId = lease_id;
        chunk.grantStatus.store(status);
        chunk.grantArrivedUs.store(profileNowUs());
        chunk.grantArrived.store(true);
        return true;
    }

    void
    markAck(uint64_t chunk_id, uint64_t lease_id, nixl_status_t status) {
        if (chunk_id >= chunks.size()) {
            NIXL_WARN << "Received UCX staged ACK for out-of-range chunk id " << chunk_id
                      << " transfer_id=" << transferId << " chunks=" << chunks.size();
            return;
        }

        auto &chunk = *chunks[chunk_id];
        const uint64_t expected_lease = chunk.leaseId.load();
        if (expected_lease != lease_id) {
            if (status == NIXL_SUCCESS) {
                // A success ACK must prove it belongs to the current attempt; a stale
                // lease would mean the data of this attempt is not actually in the GPU.
                NIXL_WARN << "Dropping UCX staged success ACK with mismatched lease id "
                          << lease_id << " transfer_id=" << transferId
                          << " chunk_id=" << chunk_id << " expected=" << expected_lease;
                return;
            }
            // Error ACKs are accepted on (transfer_id, chunk_id): the target may fail
            // before it can echo a valid lease (e.g. partial message parse), and
            // dropping the error would leave the chunk in WAIT_ACK forever.
            NIXL_WARN << "Accepting UCX staged error ACK with mismatched lease id "
                      << lease_id << " transfer_id=" << transferId
                      << " chunk_id=" << chunk_id << " expected=" << expected_lease
                      << " status=" << status;
        }

        chunk.ackStatus.store(status);
        chunk.ackArrivedUs.store(profileNowUs());
    }

    void
    release() override {
        releaseAllChunkSlots();
        for (const auto &chunk : chunks) {
            if (chunk->req) {
                chunk->req->release();
            }
        }
        for (const auto &batch : flushBatches) {
            if (batch && batch->req) {
                batch->req->release();
            }
        }
        if (openFlushBatch && openFlushBatch->req) {
            openFlushBatch->req->release();
        }
        flushBatches.clear();
        openFlushBatch.reset();
        for (auto &[gpu_dev, d2h_streams] : sourceD2HStreams) {
            int previous_device = -1;
            const nixl_status_t status = cudaSetDeviceForCopy(gpu_dev, previous_device);
            for (cudaStream_t stream : d2h_streams.streams) {
                if (status == NIXL_SUCCESS) {
                    cudaStreamDestroy(stream);
                }
            }
            cudaRestoreDevice(previous_device);
        }
        sourceD2HStreams.clear();
        nixlUcxBackendReqH::release();
    }

    const uint64_t transferId;
    std::string remoteAgent;
    size_t totalSize = 0;
    size_t completedChunks = 0;
    std::vector<std::unique_ptr<nixlUcxStagedChunk>> chunks;
    std::vector<std::unique_ptr<nixlUcxStagedFlushBatch>> flushBatches;
    std::unique_ptr<nixlUcxStagedFlushBatch> openFlushBatch;
    std::unordered_map<uint64_t, SourceD2HStreams> sourceD2HStreams;
    nixlUcxStagedProfile profile;
    bool profileLogged = false;
    bool pendingRegistered = false;
    State state = State::INIT;
    nixl_status_t lastStatus = NIXL_IN_PROG;
};

} // namespace

/****************************************
 * Progress thread management
*****************************************/

/*
 * This class encapsulates a thread that polls one or multiple UCX workers
 */
class nixlUcxThread {
public:
    nixlUcxThread(const nixlUcxEngine *engine, size_t num_workers) : engine_(engine) {
        workers_.reserve(num_workers);
    }

    virtual ~nixlUcxThread() {
        if (threadActive_) {
            join();
        }
    }

    void
    start() {
        NIXL_ASSERT(!threadActive_);
        threadActive_ = std::make_unique<std::promise<void>>();
        auto active = threadActive_->get_future();
        thread_ = std::make_unique<std::thread>(std::ref(*this));
        active.wait();
    }

    virtual void
    join() {
        NIXL_ASSERT(threadActive_);
        threadActive_.reset();
        thread_->join();
    }

    virtual void
    addWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(workers_.size() < workers_.capacity());
        workers_.push_back(worker);
        workerIds_.push_back(worker_id);
    }

    const std::vector<nixlUcxWorker *> &
    getWorkers() const {
        return workers_;
    }

    size_t
    getWorkerId(size_t idx = 0) const {
        return workerIds_[idx];
    }

    void
    operator()() {
        tlsThread() = this;
        threadActive_->set_value();
        run();
    }

    static nixlUcxThread *&
    tlsThread() {
        static thread_local nixlUcxThread *tls = nullptr;
        return tls;
    }

    static bool
    isProgressThread(const nixlUcxEngine *engine) noexcept {
        nixlUcxThread *thread = tlsThread();
        return thread && thread->engine_ == engine;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxThread &thread) {
        return os << "thread " << &thread << "{engine: " << thread.engine_ << ", worker_ids: ["
                  << absl::StrJoin(thread.workerIds_, ",") << "]}";
    }

protected:
    virtual void
    run() = 0;

private:
    const nixlUcxEngine *engine_;
    std::vector<nixlUcxWorker *> workers_;
    std::vector<size_t> workerIds_;
    std::unique_ptr<std::thread> thread_;
    std::unique_ptr<std::promise<void>> threadActive_;
};

class nixlUcxSharedThread : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine, size_t num_workers, nixlTime::us_t delay)
        : nixlUcxThread(engine, num_workers) {
        if (pipe(controlPipe_) < 0) {
            throw std::runtime_error("Couldn't create progress thread control pipe");
        }
        // TODO: We need delay to manual periodic wakeup/polling as a temporary
        // workaround for UCX bug (poll wouldn't wake up some fds in particular
        // circumstances)

        // This will ensure that the resulting delay is at least 1ms and fits into int in order for
        // it to be compatible with poll()
        int delay_us = std::min((int)delay, std::numeric_limits<int>::max());
        delay_ = std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(delay_us));

        pollFds_.resize(num_workers + 1);
        pollFds_.back() = {controlPipe_[0], POLLIN, 0};
    }

    ~nixlUcxSharedThread() {
        close(controlPipe_[0]);
        close(controlPipe_[1]);
    }

    void
    join() override {
        const char signal = 'X';
        int ret = write(controlPipe_[1], &signal, sizeof(signal));
        if (ret < 0) NIXL_PERROR << "write to progress thread control pipe failed";
        nixlUcxThread::join();
    }

    void
    addWorker(nixlUcxWorker *worker, size_t worker_id) override {
        pollFds_[getWorkers().size()] = {worker->getEfd(), POLLIN, 0};
        nixlUcxThread::addWorker(worker, worker_id);
    }

protected:
    void
    run() override {
        NIXL_DEBUG << "shared " << *this << " running";
        // Set timeout event so that the main loop would progress all workers on first iteration
        bool timeout = true;
        bool pthr_stop = false;
        while (!pthr_stop) {
            for (size_t i = 0; i < pollFds_.size() - 1; i++) {
                if (!(pollFds_[i].revents & POLLIN) && !timeout) continue;
                pollFds_[i].revents = 0;
                nixlUcxWorker *worker = getWorkers()[i];
                do {
                    worker->progressLoop();
                } while (worker->arm() == NIXL_IN_PROG);
            }
            timeout = false;

            int ret;
            while ((ret = poll(pollFds_.data(), pollFds_.size(), delay_.count())) < 0)
                NIXL_PTRACE << "Call to poll() was interrupted, retrying";

            if (!ret) {
                timeout = true;
            } else if (pollFds_.back().revents & POLLIN) {
                pollFds_.back().revents = 0;

                char signal;
                int ret = read(pollFds_.back().fd, &signal, sizeof(signal));
                if (ret < 0) NIXL_PERROR << "read() on control pipe failed";

                pthr_stop = true;
            }
        }

        NIXL_DEBUG << "shared " << *this << " exiting";
    }

private:
    std::chrono::milliseconds delay_;
    int controlPipe_[2];
    std::vector<pollfd> pollFds_;
};

nixlUcxThreadEngine::nixlUcxThreadEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    if (!nixlUcxMtLevelIsSupported(nixl::ucx::mt_mode_t::WORKER)) {
        throw std::invalid_argument("UCX library does not support multi-threading");
    }

    size_t num_workers = getWorkers().size();
    thread_ = std::make_unique<nixlUcxSharedThread>(this, num_workers, init_params.pthrDelay);
    for (size_t i = 0; i < num_workers; i++) {
        thread_->addWorker(getWorkers()[i].get(), i);
    }
    thread_->start();
}

nixlUcxThreadEngine::~nixlUcxThreadEngine() {
    stopStagedH2DWorker();
    thread_->join();
}

void
nixlUcxThreadEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    const std::lock_guard lock(notifMutex_);
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
}

nixl_status_t
nixlUcxThreadEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const std::lock_guard lock(notifMutex_);
    notifList_.swap(notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Threadpool engine
 ****************************************/

struct nixlUcxBackendSharedState;

/*
 * This class represents a chunk of a composite request.
 * It is used to encapsulate a batch of requests (subset of the larger batch)
 * performed by a dedicated worker thread of threadpool. It holds a shared state
 * with the main request to track its completion status and control the lifetime.
 */
class nixlUcxChunkBackendReqH : public nixlUcxBackendReqH {
public:
    nixlUcxChunkBackendReqH() : nixlUcxBackendReqH(nullptr, UINT64_MAX) {}

    void
    startXfer(const std::shared_ptr<nixlUcxBackendSharedState> &shared_state,
              nixlUcxWorker *worker,
              size_t worker_id) {
        NIXL_ASSERT(sharedState_.get() == nullptr);
        sharedState_ = shared_state;
        setWorker(worker, worker_id);
    }

    void
    complete(nixl_status_t status);

    [[nodiscard]] nixl_status_t
    status() override;

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxChunkBackendReqH &chunk) {
        return os << "chunk " << &chunk << "{worker_id: " << chunk.getWorkerId()
                  << ", state: " << chunk.sharedState_.get() << "}";
    }

private:
    std::shared_ptr<nixlUcxBackendSharedState> sharedState_;
};

/*
 * This class represents a shared state between a main request and all of its
 * chunks. It is used to track the completion status of the request and the
 * number of pending requests, and to control the lifetime of the chunks.
 */
struct nixlUcxBackendSharedState {
    std::atomic<nixl_status_t> status;
    std::atomic<size_t> pendingReqs;
    std::vector<nixlUcxChunkBackendReqH> chunks;

    nixlUcxBackendSharedState() : status(NIXL_SUCCESS), pendingReqs(0) {}

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxBackendSharedState &state) {
        return os << "state " << &state << "{status: " << state.status.load()
                  << ", pending=" << state.pendingReqs.load() << "}";
    }
};

void
nixlUcxChunkBackendReqH::complete(const nixl_status_t status) {
    NIXL_ASSERT(sharedState_.get() != nullptr);
    if (status != NIXL_SUCCESS) {
        nixlUcxBackendReqH::release();
        sharedState_->status.store(status);
    }
    sharedState_->pendingReqs.fetch_sub(1);
    NIXL_TRACE << *this << " completed with status: " << status << ", " << *sharedState_;
    setWorker(nullptr, UINT64_MAX);
    sharedState_.reset();
}

nixl_status_t
nixlUcxChunkBackendReqH::status() {
    // First check if entire request was cancelled or failed
    const nixl_status_t status = sharedState_->status.load();
    if (status != NIXL_SUCCESS) {
        return status;
    }
    return nixlUcxBackendReqH::status();
}

/*
 * This class represents a composite request handle for a UCX backend.
 * It is used to encapsulate multiple parallel requests performed by dedicated
 * worker threads of threadpool, with a single request handle, that it returned
 * to the user.
 */
class nixlUcxCompositeBackendReqH : public nixlUcxBackendReqH {
public:
    nixlUcxCompositeBackendReqH(nixlUcxWorker *worker,
                                size_t worker_id,
                                size_t chunk_size,
                                size_t num_chunks)
        : nixlUcxBackendReqH(worker, worker_id),
          sharedState_(std::make_shared<nixlUcxBackendSharedState>()),
          chunkSize_(chunk_size) {
        sharedState_->chunks.resize(num_chunks);
    }

    [[nodiscard]] size_t
    getChunkSize() const noexcept {
        return chunkSize_;
    }

    [[nodiscard]] size_t
    getNumChunks() const noexcept {
        return sharedState_ ? sharedState_->chunks.size() : 0;
    }

    void
    startXfer() {
        NIXL_ASSERT(sharedState_->pendingReqs.load() == 0);
        sharedState_->status.store(NIXL_SUCCESS);
        sharedState_->pendingReqs.store(getNumChunks());
    }

    [[nodiscard]] nixlUcxChunkBackendReqH *
    startChunk(size_t idx, nixlUcxWorker *worker, size_t worker_id) {
        nixlUcxChunkBackendReqH *chunk = &sharedState_->chunks[idx];
        chunk->startXfer(sharedState_, worker, worker_id);
        return chunk;
    }

    [[nodiscard]] bool
    isComposite() const noexcept override {
        return true;
    }

    void
    release() override {
        NIXL_TRACE << *this << " releasing";
        nixlUcxBackendReqH::release();
        if (sharedState_) {
            // Set failed status to stop progress chunks
            sharedState_->status.store(NIXL_ERR_NOT_FOUND);
            // Reset shared state - it will be effectively released when the last chunk
            // resets the shared state pointer
            sharedState_.reset();
        }
    }

    [[nodiscard]] nixl_status_t
    status() override {
        getWorker()->progressLoop();

        if (sharedState_->pendingReqs.load()) {
            return NIXL_IN_PROG;
        }

        const nixl_status_t status = nixlUcxBackendReqH::status();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        return sharedState_->status.load();
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxCompositeBackendReqH &handle) {
        os << "composite handle " << &handle << "{chunks: " << handle.getNumChunks();
        if (handle.sharedState_) {
            os << ", " << *handle.sharedState_;
        } else {
            os << ", state: nullptr";
        }
        return os << "}}";
    }

private:
    std::shared_ptr<nixlUcxBackendSharedState> sharedState_;
    size_t chunkSize_;
};

class nixlUcxDedicatedThread : public nixlUcxThread {
public:
    nixlUcxDedicatedThread(nixlUcxEngine *engine, asio::io_context &io)
        : nixlUcxThread(engine, 1),
          io_(io) {}

    static nixlUcxDedicatedThread *
    getDedicatedThread() {
        return (nixlUcxDedicatedThread *)tlsThread();
    }

    void
    addRequest(nixlUcxChunkBackendReqH *handle) {
        requests_.push_back(handle);
    }

protected:
    void
    run() override {
        const auto guard = asio::make_work_guard(io_);
        NIXL_DEBUG << "dedicated " << *this << " running";

        while (!io_.stopped()) {
            if (!requests_.empty()) {
                io_.poll_one();
            } else {
                NIXL_TRACE << "dedicated " << *this << " waiting for requests";
                io_.run_one();
            }

            if (requests_.empty()) {
                continue;
            }

            for (auto it = requests_.begin(); it != requests_.end();) {
                nixl_status_t status = (*it)->status();
                if (status != NIXL_IN_PROG) {
                    NIXL_TRACE << "dedicated " << *this << " completing " << *(*it)
                               << " with status: " << status;
                    (*it)->complete(status);
                    it = requests_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (!requests_.empty()) {
            NIXL_WARN << "dedicated " << *this << " dropping " << requests_.size()
                      << " requests on exit";
            for (auto it = requests_.begin(); it != requests_.end();) {
                NIXL_INFO << "dropping " << *(*it);
                (*it)->complete(NIXL_ERR_BACKEND);
            }
            requests_.clear();
        }

        NIXL_DEBUG << "dedicated " << *this << " exiting";
    }

private:
    asio::io_context &io_;
    std::vector<nixlUcxChunkBackendReqH *> requests_;
};

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    size_t num_threads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    numSharedWorkers_ = getWorkers().size() - num_threads;
    NIXL_ASSERT(numSharedWorkers_ > 0);

    splitBatchSize_ = nixl_b_params_get(init_params.customParams, "split_batch_size", 1024);

    if (init_params.enableProgTh) {
        sharedThread_ =
            std::make_unique<nixlUcxSharedThread>(this, numSharedWorkers_, init_params.pthrDelay);
        for (size_t i = 0; i < numSharedWorkers_; i++) {
            sharedThread_->addWorker(getWorkers()[i].get(), i);
        }
        sharedThread_->start();
    }

    if (num_threads > 0) {
        io_.reset(new asio::io_context());
        dedicatedThreads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            size_t worker_id = numSharedWorkers_ + i;
            dedicatedThreads_.emplace_back(std::make_unique<nixlUcxDedicatedThread>(this, *io_));
            dedicatedThreads_.back()->addWorker(getWorker(worker_id).get(), worker_id);
            dedicatedThreads_.back()->start();
        }
    }
}

nixlUcxThreadPoolEngine::~nixlUcxThreadPoolEngine() {
    stopStagedH2DWorker();
    if (sharedThread_) {
        sharedThread_->join();
    }

    if (io_) {
        io_->stop();
        for (auto &thread : dedicatedThreads_) {
            thread->join();
        }
    }
}

nixl_status_t
nixlUcxThreadPoolEngine::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH *&handle,
                                  const nixl_opt_b_args_t *opt_args) const {
    if (vramStagingEnabled() && (local.getType() == VRAM_SEG || remote.getType() == VRAM_SEG)) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    size_t batch_size = local.descCount();
    if (batch_size < splitBatchSize_) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    size_t chunk_size = std::max(batch_size / dedicatedThreads_.size(), splitBatchSize_);
    size_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    size_t worker_id = getWorkerId();
    const auto comp_handle = new nixlUcxCompositeBackendReqH(
        getWorker(worker_id).get(), worker_id, chunk_size, num_chunks);
    NIXL_TRACE << "created " << *comp_handle;
    handle = comp_handle;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxThreadPoolEngine::sendXferRange(const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH *handle,
                                       size_t start_idx,
                                       size_t end_idx) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    if (!int_handle->isComposite()) {
        return nixlUcxEngine::sendXferRange(
            operation, local, remote, remote_agent, handle, start_idx, end_idx);
    }

    const auto comp_handle = static_cast<nixlUcxCompositeBackendReqH *>(int_handle);
    comp_handle->startXfer();
    size_t chunk_size = comp_handle->getChunkSize();
    NIXL_TRACE << "sending " << *comp_handle;

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    std::atomic<size_t> remaining{comp_handle->getNumChunks()};
    std::atomic<nixl_status_t> status{NIXL_SUCCESS};

    for (size_t i = 0; i < comp_handle->getNumChunks(); i++) {
        io_->post([&, i]() {
            nixlUcxDedicatedThread *thread = nixlUcxDedicatedThread::getDedicatedThread();
            NIXL_ASSERT(thread != nullptr);

            nixlUcxChunkBackendReqH *chunk_handle =
                comp_handle->startChunk(i, thread->getWorkers()[0], thread->getWorkerId());
            NIXL_TRACE << "dedicated " << *thread << " starting " << *chunk_handle;

            size_t start_idx = i * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, (size_t)local.descCount());
            nixl_status_t ret = nixlUcxEngine::sendXferRange(
                operation, local, remote, remote_agent, chunk_handle, start_idx, end_idx);
            if (ret != NIXL_SUCCESS) {
                status.store(ret);
                chunk_handle->complete(ret);
            } else {
                NIXL_TRACE << "dedicated " << *thread << " sent " << *chunk_handle;
                thread->addRequest(chunk_handle);
            }

            if (remaining.fetch_sub(1) == 1) {
                promise.set_value();
            }
        });
    }

    future.wait();
    NIXL_TRACE << "sent " << *comp_handle << " with status: " << status.load();
    return status.load();
}

void
nixlUcxThreadPoolEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    const std::lock_guard lock(notifMutex_);
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
}

nixl_status_t
nixlUcxThreadPoolEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!sharedThread_) {
        progressLoop();
    }

    const std::lock_guard lock(notifMutex_);
    notifList_.swap(notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Constructor/Destructor
 *****************************************/

std::unique_ptr<nixlUcxEngine>
nixlUcxEngine::create(const nixlBackendInitParams &init_params) {
    nixlUcxEngine *engine;
    nixlBackendInitParams effective_init_params = init_params;
    const auto staging_config = makeVramStagingConfig(init_params.customParams);
    if (staging_config.enabled && staging_config.forceProgressThread &&
        !effective_init_params.enableProgTh) {
        NIXL_INFO << "UCX VRAM staging requested progress thread; enabling UCX progress thread";
        effective_init_params.enableProgTh = true;
    }

    size_t num_threads = nixl_b_params_get(effective_init_params.customParams, "num_threads", 0);
    if (num_threads > 0) {
        engine = new nixlUcxThreadPoolEngine(effective_init_params);
    } else if (effective_init_params.enableProgTh) {
        engine = new nixlUcxThreadEngine(effective_init_params);
    } else {
        engine = new nixlUcxEngine(effective_init_params);
    }
    return std::unique_ptr<nixlUcxEngine>(engine);
}

nixlUcxEngine::VramStagingConfig
nixlUcxEngine::makeVramStagingConfig(const nixl_b_params_t *custom_params) {
    VramStagingConfig config;
    config.enabled = nixl_b_params_get_bool(
        custom_params, nixl_ucx_vram_staging_param_name, config.enabled);
    config.chunkSize = nixl_b_params_get_size(
        custom_params, nixl_ucx_staging_chunk_size_param_name, config.chunkSize);
    config.slotsPerGpu =
        nixl_b_params_get_size(custom_params, nixl_ucx_staging_slots_param_name, config.slotsPerGpu);
    config.forceProgressThread = nixl_b_params_get_bool(custom_params,
                                                        nixl_ucx_staging_force_progress_param_name,
                                                        config.forceProgressThread);
    config.cudaCopyStreams = nixl_b_params_get_size(custom_params,
                                                    nixl_ucx_staging_cuda_streams_param_name,
                                                    config.cudaCopyStreams);
    config.slotRequestWindow = nixl_b_params_get_size(custom_params,
                                                      nixl_ucx_staging_slot_window_param_name,
                                                      config.slotRequestWindow);
    config.batchFlush = nixl_b_params_get_bool(custom_params,
                                               nixl_ucx_staging_batch_flush_param_name,
                                               config.batchFlush);
    config.targetH2DWorker = nixl_b_params_get_bool(
        custom_params, nixl_ucx_staging_target_h2d_worker_param_name, config.targetH2DWorker);
    config.sourceD2HPrefetch = nixl_b_params_get_bool(
        custom_params, nixl_ucx_staging_source_d2h_prefetch_param_name, config.sourceD2HPrefetch);
    config.leaseTimeoutMs = nixl_b_params_get_size(
        custom_params, nixl_ucx_staging_lease_timeout_param_name, config.leaseTimeoutMs);
    config.localStaging = nixl_b_params_get_bool(
        custom_params, nixl_ucx_vram_local_staging_param_name, config.localStaging);
    config.localStagingFallback = nixl_b_params_get_bool(custom_params,
                                                         nixl_ucx_local_staging_fallback_param_name,
                                                         config.localStagingFallback);
    if (custom_params) {
        const auto it = custom_params->find(std::string(nixl_ucx_local_staging_shm_dir_param_name));
        if (it != custom_params->end() && !trimAscii(it->second).empty()) {
            config.localStagingShmDir = trimAscii(it->second);
        }
    }
    config.enabled = nixl_env_get_bool(nixl_ucx_vram_staging_env_name, config.enabled);
    config.chunkSize = nixl_env_get_size(nixl_ucx_staging_chunk_size_env_name, config.chunkSize);
    config.slotsPerGpu = nixl_env_get_size(nixl_ucx_staging_slots_env_name, config.slotsPerGpu);
    config.forceProgressThread =
        nixl_env_get_bool(nixl_ucx_staging_force_progress_env_name, config.forceProgressThread);
    config.cudaCopyStreams =
        nixl_env_get_size(nixl_ucx_staging_cuda_streams_env_name, config.cudaCopyStreams);
    config.slotRequestWindow =
        nixl_env_get_size(nixl_ucx_staging_slot_window_env_name, config.slotRequestWindow);
    config.batchFlush =
        nixl_env_get_bool(nixl_ucx_staging_batch_flush_env_name, config.batchFlush);
    config.targetH2DWorker = nixl_env_get_bool(nixl_ucx_staging_target_h2d_worker_env_name,
                                               config.targetH2DWorker);
    config.sourceD2HPrefetch =
        nixl_env_get_bool(nixl_ucx_staging_source_d2h_prefetch_env_name,
                          config.sourceD2HPrefetch);
    config.leaseTimeoutMs =
        nixl_env_get_size(nixl_ucx_staging_lease_timeout_env_name, config.leaseTimeoutMs);
    config.localStaging =
        nixl_env_get_bool(nixl_ucx_vram_local_staging_env_name, config.localStaging);
    config.localStagingFallback =
        nixl_env_get_bool(nixl_ucx_local_staging_fallback_env_name,
                          config.localStagingFallback);
    config.localStagingShmDir =
        envString(nixl_ucx_local_staging_shm_dir_env_name, config.localStagingShmDir);
    if (config.localStaging) {
        if (!config.enabled) {
            config.enabled = true;
            config.localStagingAutoEnabled = true;
        }
        config.sourceD2HPrefetch = true;
    }
    return config;
}

nixlUcxEngine::nixlUcxEngine(const nixlBackendInitParams &init_params)
    : nixlBackendEngine(&init_params),
      sharedWorkerIndex_(1),
      vramStagingConfig_(makeVramStagingConfig(init_params.customParams)),
      nextStagedTransferId_(1) {
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t *custom_params = init_params.customParams;

    devs = getConfiguredUcxDevices(custom_params);

    size_t num_workers = nixl_b_params_get(custom_params, "num_workers", 1);
    size_t num_threads = nixl_b_params_get(custom_params, "num_threads", 0);
    size_t num_device_channels = nixl_b_params_get(custom_params, "ucx_num_device_channels", 4);

    if (num_workers <= num_threads) {
        /* There must be at least one shared worker */
        num_workers = num_threads + 1;
    }

    ucp_err_handling_mode_t err_handling_mode;
    const auto err_handling_mode_it =
        custom_params->find(std::string(nixl_ucx_err_handling_param_name));
    if (err_handling_mode_it == custom_params->end()) {
        err_handling_mode = UCP_ERR_HANDLING_MODE_PEER;
    } else {
        err_handling_mode = ucx_err_mode_from_string(err_handling_mode_it->second);
    }

    const auto engine_config_it = custom_params->find("engine_config");
    const auto engine_config =
        (engine_config_it != custom_params->end()) ? engine_config_it->second : "";

    uc = std::make_unique<nixlUcxContext>(devs,
                                          init_params.enableProgTh,
                                          num_workers,
                                          init_params.syncMode,
                                          num_device_channels,
                                          engine_config);

    uc->warnAboutHardwareSupportMismatch();

    for (size_t i = 0; i < num_workers; i++) {
        uws.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode));
    }

    auto &uw = uws.front();
    workerAddr = uw->epAddr();
    uw->regAmCallback(nixl::ucx::am_cb_op_t::NOTIF_STR, notifAmCb, this);
    uw->regAmCallback(nixl::ucx::am_cb_op_t::STAGED_SLOT_REQ, stagedSlotReqAmCb, this);
    uw->regAmCallback(nixl::ucx::am_cb_op_t::STAGED_SLOT_GRANT, stagedSlotGrantAmCb, this);
    uw->regAmCallback(nixl::ucx::am_cb_op_t::STAGED_SLOT_RELEASE, stagedSlotReleaseAmCb, this);
    uw->regAmCallback(nixl::ucx::am_cb_op_t::STAGED_WRITE_READY, stagedWriteReadyAmCb, this);
    uw->regAmCallback(nixl::ucx::am_cb_op_t::STAGED_LOCAL_WRITE_READY,
                      stagedLocalWriteReadyAmCb,
                      this);
    uw->regAmCallback(nixl::ucx::am_cb_op_t::STAGED_ACK, stagedAckAmCb, this);

    if (vramStagingConfig_.enabled) {
        if (vramStagingConfig_.localStagingAutoEnabled) {
            NIXL_WARN << "UCX local VRAM staging requested without vram_staging; enabling "
                         "UCX VRAM staging automatically";
        }
        NIXL_INFO << "UCX VRAM staging enabled: chunk_size=" << vramStagingConfig_.chunkSize
                  << " slots_per_gpu=" << vramStagingConfig_.slotsPerGpu
                  << " cuda_copy_streams=" << vramStagingConfig_.cudaCopyStreams
                  << " slot_request_window=" << vramStagingConfig_.slotRequestWindow
                  << " batch_flush=" << vramStagingConfig_.batchFlush
                  << " target_h2d_worker=" << vramStagingConfig_.targetH2DWorker
                  << " source_d2h_prefetch=" << vramStagingConfig_.sourceD2HPrefetch
                  << " lease_timeout_ms=" << vramStagingConfig_.leaseTimeoutMs
                  << " local_staging=" << vramStagingConfig_.localStaging
                  << " local_staging_fallback=" << vramStagingConfig_.localStagingFallback
                  << " local_staging_shm_dir=" << vramStagingConfig_.localStagingShmDir
                  << " force_progress_thread=" << vramStagingConfig_.forceProgressThread;
        if (vramStagingConfig_.targetH2DWorker) {
            startStagedH2DWorker();
        }
    }
}

nixl_mem_list_t nixlUcxEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

static std::unordered_map<const nixlUcxEngine *, size_t> &
tlsSharedWorkerMap() {
    static thread_local std::unordered_map<const nixlUcxEngine *, size_t> map;
    return map;
}

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine() {
    stopStagedH2DWorker();
    cleanupLocalSharedAttachments();
    tlsSharedWorkerMap().erase(this);
}

void
nixlUcxEngine::startStagedH2DWorker() {
    stagedH2DThread_ = std::thread([this]() { stagedH2DWorkerLoop(); });
}

void
nixlUcxEngine::stopStagedH2DWorker() {
    {
        const std::lock_guard lock(stagedH2DMutex_);
        stagedH2DStop_ = true;
    }
    stagedH2DCv_.notify_all();
    if (stagedH2DThread_.joinable()) {
        stagedH2DThread_.join();
    }
}

nixl_status_t
nixlUcxEngine::enqueueStagedH2D(StagedH2DTask &&task) const {
    {
        const std::lock_guard lock(stagedH2DMutex_);
        if (stagedH2DStop_) {
            return NIXL_ERR_BACKEND;
        }
        stagedH2DQueue_.push_back(std::move(task));
    }
    stagedH2DCv_.notify_one();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::getLocalSharedAttachment(
    const std::string &remote_agent,
    const std::string &path,
    size_t mapping_size,
    std::shared_ptr<LocalSharedAttachment> &attachment) const {
    if (remote_agent.empty() || path.empty() || mapping_size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto canonical_dir = canonicalPath(vramStagingConfig_.localStagingShmDir);
    const auto canonical_path = canonicalPath(path);
    if (!canonical_dir || !canonical_path ||
        !pathIsUnderDirectory(*canonical_path, *canonical_dir)) {
        NIXL_ERROR << "UCX local staging source path is outside configured staging directory: "
                   << path << " dir=" << vramStagingConfig_.localStagingShmDir;
        localSharedAttachFailures_.fetch_add(1, std::memory_order_relaxed);
        return NIXL_ERR_INVALID_PARAM;
    }

    {
        const std::lock_guard lock(localSharedAttachMutex_);
        const auto it = localSharedAttachments_.find(*canonical_path);
        if (it != localSharedAttachments_.end()) {
            if (it->second->mappingSize != mapping_size ||
                it->second->remoteAgent != remote_agent) {
                localSharedAttachFailures_.fetch_add(1, std::memory_order_relaxed);
                return NIXL_ERR_MISMATCH;
            }
            localSharedAttachCacheHits_.fetch_add(1, std::memory_order_relaxed);
            attachment = it->second;
            return NIXL_SUCCESS;
        }
    }

    localSharedAttachCacheMisses_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t attach_start_us = profileNowUs();
    auto new_attachment = std::make_shared<LocalSharedAttachment>();
    new_attachment->path = *canonical_path;
    new_attachment->remoteAgent = remote_agent;
    new_attachment->mappingSize = mapping_size;
    new_attachment->fd = open(canonical_path->c_str(), O_RDWR | O_NOFOLLOW);
    if (new_attachment->fd < 0) {
        NIXL_ERROR << "Failed to open UCX local staging source path " << *canonical_path << ": "
                   << std::strerror(errno);
        localSharedAttachFailures_.fetch_add(1, std::memory_order_relaxed);
        localSharedAttachUs_.fetch_add(profileNowUs() - attach_start_us,
                                       std::memory_order_relaxed);
        return NIXL_ERR_BACKEND;
    }

    new_attachment->base =
        mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_SHARED, new_attachment->fd, 0);
    if (new_attachment->base == MAP_FAILED) {
        NIXL_ERROR << "Failed to mmap UCX local staging source path " << *canonical_path << ": "
                   << std::strerror(errno);
        close(new_attachment->fd);
        new_attachment->fd = -1;
        new_attachment->base = nullptr;
        localSharedAttachFailures_.fetch_add(1, std::memory_order_relaxed);
        localSharedAttachUs_.fetch_add(profileNowUs() - attach_start_us,
                                       std::memory_order_relaxed);
        return NIXL_ERR_BACKEND;
    }

    const cudaError_t register_ret =
        cudaHostRegister(new_attachment->base, mapping_size, cudaHostRegisterDefault);
    if (register_ret != cudaSuccess) {
        NIXL_ERROR << "cudaHostRegister failed for UCX local staging source path "
                   << *canonical_path << ": " << cudaGetErrorString(register_ret);
        munmap(new_attachment->base, mapping_size);
        close(new_attachment->fd);
        new_attachment->base = nullptr;
        new_attachment->fd = -1;
        localSharedAttachFailures_.fetch_add(1, std::memory_order_relaxed);
        localSharedAttachUs_.fetch_add(profileNowUs() - attach_start_us,
                                       std::memory_order_relaxed);
        return NIXL_ERR_BACKEND;
    }
    new_attachment->hostRegistered = true;
    localSharedAttachUs_.fetch_add(profileNowUs() - attach_start_us, std::memory_order_relaxed);

    {
        const std::lock_guard lock(localSharedAttachMutex_);
        const auto [it, inserted] = localSharedAttachments_.emplace(*canonical_path, new_attachment);
        if (!inserted) {
            cudaHostUnregister(new_attachment->base);
            munmap(new_attachment->base, mapping_size);
            close(new_attachment->fd);
            if (it->second->mappingSize != mapping_size ||
                it->second->remoteAgent != remote_agent) {
                localSharedAttachFailures_.fetch_add(1, std::memory_order_relaxed);
                return NIXL_ERR_MISMATCH;
            }
            localSharedAttachCacheHits_.fetch_add(1, std::memory_order_relaxed);
            attachment = it->second;
            return NIXL_SUCCESS;
        }
    }

    attachment = std::move(new_attachment);
    return NIXL_SUCCESS;
}

void
nixlUcxEngine::releaseLocalSharedAttachment(std::shared_ptr<LocalSharedAttachment> &attachment) {
    if (!attachment) {
        return;
    }
    if (attachment->hostRegistered && attachment->base != nullptr) {
        cudaHostUnregister(attachment->base);
        attachment->hostRegistered = false;
    }
    if (attachment->base != nullptr) {
        munmap(attachment->base, attachment->mappingSize);
        attachment->base = nullptr;
    }
    if (attachment->fd >= 0) {
        close(attachment->fd);
        attachment->fd = -1;
    }
}

void
nixlUcxEngine::cleanupLocalSharedAttachments() {
    std::unordered_map<std::string, std::shared_ptr<LocalSharedAttachment>> attachments;
    {
        const std::lock_guard lock(localSharedAttachMutex_);
        attachments.swap(localSharedAttachments_);
        localSharedRegions_.clear();
    }

    for (auto &[path, attachment] : attachments) {
        releaseLocalSharedAttachment(attachment);
    }
}

void
nixlUcxEngine::cleanupLocalSharedAttachmentsForAgent(const std::string &remote_agent) {
    std::vector<std::shared_ptr<LocalSharedAttachment>> removed;
    {
        const std::lock_guard lock(localSharedAttachMutex_);
        for (auto it = localSharedAttachments_.begin(); it != localSharedAttachments_.end();) {
            if (it->second && it->second->remoteAgent == remote_agent) {
                removed.push_back(it->second);
                it = localSharedAttachments_.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = localSharedRegions_.begin(); it != localSharedRegions_.end();) {
            if (it->second.remoteAgent == remote_agent) {
                it = localSharedRegions_.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (auto &attachment : removed) {
        releaseLocalSharedAttachment(attachment);
    }
}

void
nixlUcxEngine::cleanupLocalSharedAttachmentPath(const std::string &path) const {
    std::shared_ptr<LocalSharedAttachment> removed;
    {
        const std::lock_guard lock(localSharedAttachMutex_);
        const auto canonical_path = canonicalPath(path);
        const std::string key = canonical_path ? *canonical_path : path;
        const auto it = localSharedAttachments_.find(key);
        if (it != localSharedAttachments_.end()) {
            removed = it->second;
            localSharedAttachments_.erase(it);
        }
    }

    releaseLocalSharedAttachment(removed);
}

namespace {

[[nodiscard]] std::string
localSharedRegionKey(const std::string &remote_agent, uint64_t region_id) {
    return remote_agent + "#" + std::to_string(region_id);
}

struct RawAmSendCtx {
    std::string *buffer = nullptr;
};

void
rawAmSendCallback(void *request, ucs_status_t status, void *user_data) {
    if (status != UCS_OK) {
        NIXL_ERROR << "UCX AM reply send failed with status " << status << " ("
                   << ucs_status_string(status) << ")";
    }

    auto *ctx = static_cast<RawAmSendCtx *>(user_data);
    delete ctx->buffer;
    delete ctx;
    if (request != nullptr) {
        ucp_request_free(request);
    }
}

nixl_status_t
sendRawAmOnEp(ucp_ep_h ep, nixl::ucx::am_cb_op_t msg_id, std::string *buffer) {
    if (ep == nullptr || buffer == nullptr) {
        delete buffer;
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *ctx = new RawAmSendCtx{buffer};
    ucp_request_param_t param = {0};
    param.op_attr_mask |=
        UCP_OP_ATTR_FIELD_FLAGS | UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    param.flags = UCP_AM_SEND_FLAG_EAGER;
    param.cb.send = rawAmSendCallback;
    param.user_data = ctx;

    const ucs_status_ptr_t request =
        ucp_am_send_nbx(ep, unsigned(msg_id), nullptr, 0, buffer->data(), buffer->size(), &param);
    if (UCS_PTR_IS_PTR(request)) {
        return NIXL_IN_PROG;
    }

    const ucs_status_t status = UCS_PTR_STATUS(request);
    rawAmSendCallback(nullptr, status, ctx);
    return nixl::ucx::ucsToNixlStatus(status);
}

} // namespace

nixl_status_t
nixlUcxEngine::sendStagedControlAm(const std::string &remote_agent,
                                   ucp_ep_h reply_ep,
                                   nixl::ucx::am_cb_op_t msg_id,
                                   std::string *buffer) const {
    if (buffer == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (reply_ep != nullptr) {
        return sendRawAmOnEp(reply_ep, msg_id, buffer);
    }

    const auto conn = getConnection(remote_agent);
    if (!conn) {
        NIXL_ERROR << "Cannot send UCX staged control AM to unknown agent " << remote_agent
                   << " msg_id=" << static_cast<unsigned>(msg_id);
        delete buffer;
        return NIXL_ERR_NOT_FOUND;
    }

    auto deleter = [buffer](void *completed_request, void *ptr) {
        delete buffer;
        if (completed_request != nullptr) {
            ucp_request_free(completed_request);
        }
    };

    return conn->getEp(0)->sendAm(msg_id,
                                  nullptr,
                                  0,
                                  (void *)buffer->data(),
                                  buffer->size(),
                                  UCP_AM_SEND_FLAG_EAGER,
                                  nullptr,
                                  deleter);
}

void
nixlUcxEngine::registerLocalSharedRegion(const std::string &remote_agent,
                                         const nixlBackendMD *metadata) const {
    const auto *staged = dynamic_cast<const nixlUcxStagedPublicMetadata *>(metadata);
    if (!staged || !staged->localSharedSlots || staged->localSharedRegionId == 0 ||
        staged->localSharedPath.empty() || staged->localSharedMappingSize == 0 ||
        staged->slotSize == 0 || staged->slotAddrs.empty()) {
        return;
    }

    const std::lock_guard lock(localSharedAttachMutex_);
    auto &region = localSharedRegions_[localSharedRegionKey(remote_agent,
                                                            staged->localSharedRegionId)];
    if (region.refCount == 0) {
        region.remoteAgent = remote_agent;
        region.regionId = staged->localSharedRegionId;
        region.regionCookie = staged->localSharedRegionCookie;
        region.sharedPath = staged->localSharedPath;
        region.mappingSize = staged->localSharedMappingSize;
        region.slotSize = staged->slotSize;
        region.slotCount = staged->slotAddrs.size();
    } else if (region.regionCookie != staged->localSharedRegionCookie ||
               region.sharedPath != staged->localSharedPath ||
               region.mappingSize != staged->localSharedMappingSize ||
               region.slotSize != staged->slotSize ||
               region.slotCount != staged->slotAddrs.size()) {
        NIXL_WARN << "Conflicting UCX local shared region metadata for agent "
                  << remote_agent << " region_id=" << staged->localSharedRegionId;
    }
    ++region.refCount;
}

void
nixlUcxEngine::unregisterLocalSharedRegion(const nixlBackendMD *metadata) const {
    const auto *staged = dynamic_cast<const nixlUcxStagedPublicMetadata *>(metadata);
    if (!staged || !staged->localSharedSlots || staged->localSharedRegionId == 0) {
        return;
    }

    std::string cleanup_path;
    {
        const std::lock_guard lock(localSharedAttachMutex_);
        const auto key = localSharedRegionKey(staged->agent, staged->localSharedRegionId);
        const auto it = localSharedRegions_.find(key);
        if (it == localSharedRegions_.end()) {
            return;
        }
        if (it->second.refCount > 1) {
            --it->second.refCount;
            return;
        }
        cleanup_path = it->second.sharedPath;
        localSharedRegions_.erase(it);
    }

    cleanupLocalSharedAttachmentPath(cleanup_path);
}

bool
nixlUcxEngine::validateLocalSharedReady(const std::string &remote_agent,
                                        uint64_t region_id,
                                        const std::string &region_cookie,
                                        const std::string &shared_path,
                                        uint64_t slot_id,
                                        size_t slot_offset,
                                        size_t mapping_size,
                                        size_t size) const {
    if (remote_agent.empty() || region_id == 0 || shared_path.empty() ||
        mapping_size == 0 || size == 0) {
        return false;
    }

    const std::lock_guard lock(localSharedAttachMutex_);
    const auto it = localSharedRegions_.find(localSharedRegionKey(remote_agent, region_id));
    if (it == localSharedRegions_.end()) {
        const auto canonical_dir = canonicalPath(vramStagingConfig_.localStagingShmDir);
        const auto canonical_path = canonicalPath(shared_path);
        const std::string expected_agent = sanitizePathComponent(remote_agent);
        if (!canonical_dir || !canonical_path ||
            !pathIsUnderDirectory(*canonical_path, *canonical_dir) ||
            expected_agent.empty() ||
            canonical_path->find(expected_agent) == std::string::npos ||
            canonical_path->find(std::to_string(region_id)) == std::string::npos ||
            (!region_cookie.empty() && canonical_path->find(region_cookie) == std::string::npos) ||
            size > vramStagingConfig_.chunkSize) {
            return false;
        }

        const size_t slot_span = roundUp(vramStagingConfig_.chunkSize, pageSize());
        if (slot_id > std::numeric_limits<size_t>::max() / slot_span) {
            return false;
        }
        const size_t expected_offset = static_cast<size_t>(slot_id) * slot_span;
        return slot_offset == expected_offset &&
               slot_offset <= mapping_size &&
               size <= mapping_size - slot_offset;
    }

    const auto &region = it->second;
    if (region.sharedPath != shared_path ||
        (!region.regionCookie.empty() && region_cookie != region.regionCookie) ||
        region.mappingSize != mapping_size ||
        slot_id >= region.slotCount || size > region.slotSize) {
        return false;
    }

    const size_t slot_span = roundUp(region.slotSize, pageSize());
    if (slot_id > std::numeric_limits<size_t>::max() / slot_span) {
        return false;
    }
    const size_t expected_offset = static_cast<size_t>(slot_id) * slot_span;
    return slot_offset == expected_offset &&
           slot_offset <= mapping_size &&
           size <= mapping_size - slot_offset;
}

void
nixlUcxEngine::stagedH2DWorkerLoop() const {
    struct DeviceStreams {
        std::vector<cudaStream_t> streams;
        size_t next = 0;
    };

    struct InflightH2D {
        StagedH2DTask task;
        nixlUcxStagedPrivateMetadata *region = nullptr;
        cudaEvent_t event = nullptr;
        uint64_t h2dStartUs = 0;
    };

    std::unordered_map<uint64_t, DeviceStreams> device_streams;
    std::vector<InflightH2D> inflight;

    auto finish_task = [&](const StagedH2DTask &task,
                           nixlUcxStagedPrivateMetadata *region,
                           nixl_status_t status,
                           uint64_t h2d_us) {
        if (region) {
            region->finishRemoteLease(task.slotId, task.leaseId, status);
        }

        const nixl_status_t ack_status =
            sendStagedAck(task.remoteAgent,
                          task.transferId,
                          task.chunkId,
                          task.leaseId,
                          status,
                          task.replyEp);
        if (ack_status != NIXL_SUCCESS && ack_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to send UCX staged ACK transfer_id=" << task.transferId
                       << " chunk_id=" << task.chunkId << " status=" << ack_status;
        }

        if (task.profileEnabled) {
            const uint64_t ready_count = stagedProfileTargetReadyCount_.fetch_add(1) + 1;
            const uint64_t total_bytes = stagedProfileTargetBytes_.fetch_add(task.size) + task.size;
            const uint64_t total_h2d_us = stagedProfileTargetH2DUs_.fetch_add(h2d_us) + h2d_us;
            const uint64_t total_callback_us =
                stagedProfileTargetCallbackUs_.fetch_add(task.callbackUs) + task.callbackUs;
            if ((ready_count % 64) == 0) {
                NIXL_INFO << "UCX staged profile target ready_count=" << ready_count
                          << " bytes=" << total_bytes
                          << " h2d_total_us=" << total_h2d_us
                          << " h2d_avg_us=" << (total_h2d_us / ready_count)
                          << " callback_total_us=" << total_callback_us
                          << " callback_avg_us=" << (total_callback_us / ready_count);
            }
        }
    };

    auto poll_inflight = [&]() {
        for (size_t i = 0; i < inflight.size();) {
            InflightH2D &item = inflight[i];
            const cudaError_t query = cudaEventQuery(item.event);
            if (query == cudaErrorNotReady) {
                ++i;
                continue;
            }

            nixl_status_t status = NIXL_SUCCESS;
            if (query != cudaSuccess) {
                NIXL_ERROR << "UCX staged async H2D event failed for transfer_id="
                           << item.task.transferId << " chunk_id=" << item.task.chunkId << ": "
                           << cudaGetErrorString(query);
                status = NIXL_ERR_BACKEND;
            }

            uint64_t h2d_us = 0;
            if (item.task.profileEnabled && item.h2dStartUs != 0) {
                h2d_us = profileNowUs() - item.h2dStartUs;
            }
            cudaEventDestroy(item.event);
            finish_task(item.task, item.region, status, h2d_us);
            inflight.erase(inflight.begin() + i);
        }
    };

    auto submit_task = [&](StagedH2DTask &&task) {
        auto *region = dynamic_cast<nixlUcxStagedPrivateMetadata *>(task.region);
        if (!region) {
            finish_task(task, nullptr, NIXL_ERR_MISMATCH, 0);
            return;
        }

        int previous_device = -1;
        nixl_status_t status = cudaSetDeviceForCopy(task.gpuDev, previous_device);
        if (status != NIXL_SUCCESS) {
            finish_task(task, region, status, 0);
            return;
        }

        DeviceStreams &streams = device_streams[task.gpuDev];
        if (streams.streams.empty()) {
            const size_t stream_count = std::max<size_t>(1, vramStagingConfig_.cudaCopyStreams);
            streams.streams.reserve(stream_count);
            for (size_t i = 0; i < stream_count; ++i) {
                cudaStream_t stream = nullptr;
                const cudaError_t cuda_ret = cudaStreamCreateWithFlags(&stream,
                                                                       cudaStreamNonBlocking);
                if (cuda_ret != cudaSuccess) {
                    NIXL_ERROR << "UCX staged H2D stream creation failed for gpu_dev="
                               << task.gpuDev << ": " << cudaGetErrorString(cuda_ret);
                    cudaRestoreDevice(previous_device);
                    finish_task(task, region, NIXL_ERR_BACKEND, 0);
                    return;
                }
                streams.streams.push_back(stream);
            }
        }

        cudaEvent_t event = nullptr;
        cudaError_t cuda_ret = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (cuda_ret != cudaSuccess) {
            NIXL_ERROR << "UCX staged H2D event creation failed for transfer_id="
                       << task.transferId << " chunk_id=" << task.chunkId << ": "
                       << cudaGetErrorString(cuda_ret);
            cudaRestoreDevice(previous_device);
            finish_task(task, region, NIXL_ERR_BACKEND, 0);
            return;
        }

        cudaStream_t stream = streams.streams[streams.next++ % streams.streams.size()];
        const uint64_t h2d_start_us = task.profileEnabled ? profileNowUs() : 0;
        cuda_ret =
            cudaMemcpyAsync((void *)task.gpuAddr,
                            task.hostAddr,
                            task.size,
                            cudaMemcpyHostToDevice,
                            stream);
        if (cuda_ret == cudaSuccess) {
            cuda_ret = cudaEventRecord(event, stream);
        }
        cudaRestoreDevice(previous_device);

        if (cuda_ret != cudaSuccess) {
            NIXL_ERROR << "UCX staged async H2D submit failed for transfer_id=" << task.transferId
                       << " chunk_id=" << task.chunkId << ": " << cudaGetErrorString(cuda_ret);
            cudaEventDestroy(event);
            finish_task(task, region, NIXL_ERR_BACKEND, 0);
            return;
        }

        inflight.push_back({std::move(task), region, event, h2d_start_us});
    };

    auto destroy_streams = [&]() {
        for (auto &[gpu_dev, streams] : device_streams) {
            int previous_device = -1;
            const nixl_status_t status = cudaSetDeviceForCopy(gpu_dev, previous_device);
            for (cudaStream_t stream : streams.streams) {
                if (status == NIXL_SUCCESS) {
                    cudaStreamDestroy(stream);
                }
            }
            cudaRestoreDevice(previous_device);
        }
    };

    while (true) {
        std::deque<StagedH2DTask> ready_tasks;
        {
            std::unique_lock lock(stagedH2DMutex_);
            stagedH2DCv_.wait_for(lock, std::chrono::microseconds(50), [this]() {
                return stagedH2DStop_ || !stagedH2DQueue_.empty();
            });
            if (stagedH2DStop_ && stagedH2DQueue_.empty() && inflight.empty()) {
                destroy_streams();
                return;
            }
            while (!stagedH2DQueue_.empty()) {
                ready_tasks.push_back(std::move(stagedH2DQueue_.front()));
                stagedH2DQueue_.pop_front();
            }
        }

        for (auto &task : ready_tasks) {
            submit_task(std::move(task));
        }

        poll_inflight();
    }
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlUcxEngine::checkConn(const std::string &remote_agent) {
    const std::shared_lock lock(remoteConnMapMutex_);
    return remoteConnMap.count(remote_agent) ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
    str = workerAddr;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
    if(remote_agent == localAgent) {
        return loadRemoteConnInfo(remote_agent, workerAddr);
    }

    const std::shared_lock lock(remoteConnMapMutex_);
    return (remoteConnMap.find(remote_agent) == remoteConnMap.end()) ? NIXL_ERR_NOT_FOUND :
                                                                       NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {
    // Reclaim staged leases even when no connection exists: a staging target
    // (e.g. SGLang decode) may hold leases from an initiator it never connected to.
    size_t reclaimed = 0;
    {
        const std::lock_guard region_lock(stagedRegionMutex_);
        for (nixlBackendMD *region : stagedRegions_) {
            if (auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(region)) {
                reclaimed += staged->releaseRemoteLeasesForOwner(remote_agent);
            }
        }
    }
    if (reclaimed != 0) {
        NIXL_INFO << "Reclaimed " << reclaimed << " UCX staged slot lease(s) held by "
                  << remote_agent << " on disconnect";
    }

    {
        const std::unique_lock lock(remoteConnMapMutex_);
        const auto it = remoteConnMap.find(remote_agent);

        if (it == remoteConnMap.end()) {
            return NIXL_ERR_NOT_FOUND;
        }

        remoteConnMap.erase(it);
    }
    cleanupLocalSharedAttachmentsForAgent(remote_agent);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                                 const std::string &remote_conn_info)
{
    size_t size = remote_conn_info.size();
    std::vector<char> addr(size);

    {
        const std::shared_lock lock(remoteConnMapMutex_);
        if(remoteConnMap.count(remote_agent)) {
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    nixlSerDes::_stringToBytes(addr.data(), remote_conn_info, size);
    std::shared_ptr<nixlUcxConnection> conn = std::make_shared<nixlUcxConnection>();
    for (auto &uw : uws) {
        std::unique_ptr<nixlUcxEp> result = uw->connect(addr.data(), size);
        if (!result) {
            return NIXL_ERR_BACKEND;
        }
        conn->eps.push_back(std::move(result));
    }

    {
        const std::unique_lock lock(remoteConnMapMutex_);
        const auto [it, inserted] = remoteConnMap.emplace(remote_agent, std::move(conn));
        if (!inserted) {
            // Lost a concurrent load of the same agent; keep the existing connection.
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/
nixl_status_t nixlUcxEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    if (vramStagingEnabled() && nixl_mem == VRAM_SEG) {
        if (mem.len == 0 || vramStagingConfig_.chunkSize == 0 ||
            vramStagingConfig_.slotsPerGpu == 0) {
            NIXL_ERROR << "Invalid UCX VRAM staging configuration: chunk_size="
                       << vramStagingConfig_.chunkSize
                       << " slots_per_gpu=" << vramStagingConfig_.slotsPerGpu
                       << " mem_len=" << mem.len;
            return NIXL_ERR_INVALID_PARAM;
        }

        auto staged = std::make_unique<nixlUcxStagedPrivateMetadata>(mem);
        staged->slotSize = std::min(vramStagingConfig_.chunkSize, mem.len);
        staged->leaseTimeoutUs =
            static_cast<uint64_t>(vramStagingConfig_.leaseTimeoutMs) * 1000;
        staged->slots.resize(vramStagingConfig_.slotsPerGpu);
        staged->slotLeases.resize(vramStagingConfig_.slotsPerGpu);
        staged->localSlotGenerations.resize(vramStagingConfig_.slotsPerGpu, 0);
        staged->hostId = localHostId();
        staged->localSharedSlots = vramStagingConfig_.localStaging;
        staged->localSharedRegionId = reinterpret_cast<uintptr_t>(staged.get());
        if (staged->localSharedSlots) {
            staged->localSharedRegionCookie = newLocalSharedRegionCookie();
        }

        auto cleanup_staged = [&](size_t upto) {
            for (size_t j = 0; j < upto && j < staged->slots.size(); ++j) {
                if (staged->slots[j].ucxRegistered) {
                    uc->memDereg(staged->slots[j].mem);
                    staged->slots[j].ucxRegistered = false;
                }
                if (!staged->localSharedSlots && staged->slots[j].hostAddr != nullptr) {
                    cudaFreeHost(staged->slots[j].hostAddr);
                    staged->slots[j].hostAddr = nullptr;
                }
            }
            if (staged->sharedHostRegistered && staged->sharedBase != nullptr) {
                cudaHostUnregister(staged->sharedBase);
                staged->sharedHostRegistered = false;
            }
            if (staged->sharedBase != nullptr) {
                munmap(staged->sharedBase, staged->sharedMappingSize);
                staged->sharedBase = nullptr;
            }
            if (staged->sharedFd >= 0) {
                close(staged->sharedFd);
                staged->sharedFd = -1;
            }
            if (staged->unlinkSharedPath && !staged->sharedPath.empty()) {
                unlink(staged->sharedPath.c_str());
                staged->unlinkSharedPath = false;
            }
        };

        if (staged->localSharedSlots) {
            nixl_status_t dir_status = ensureDirectory(vramStagingConfig_.localStagingShmDir);
            if (dir_status != NIXL_SUCCESS) {
                return dir_status;
            }

            const size_t slot_span = roundUp(staged->slotSize, pageSize());
            staged->sharedMappingSize = slot_span * staged->slots.size();
            staged->sharedPath = vramStagingConfig_.localStagingShmDir + "/nixl-ucx-local-" +
                                 sanitizePathComponent(localAgent) + "-" +
                                 std::to_string(getpid()) + "-" +
                                 std::to_string(staged->localSharedRegionId) + "-" +
                                 staged->localSharedRegionCookie + ".bin";
            staged->sharedFd =
                open(staged->sharedPath.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
            if (staged->sharedFd < 0) {
                NIXL_ERROR << "Failed to create UCX local staging file " << staged->sharedPath
                           << ": " << std::strerror(errno);
                return NIXL_ERR_BACKEND;
            }
            staged->unlinkSharedPath = true;

            if (ftruncate(staged->sharedFd,
                          static_cast<off_t>(staged->sharedMappingSize)) != 0) {
                NIXL_ERROR << "Failed to size UCX local staging file " << staged->sharedPath
                           << ": " << std::strerror(errno);
                cleanup_staged(0);
                return NIXL_ERR_BACKEND;
            }

            staged->sharedBase = mmap(nullptr,
                                      staged->sharedMappingSize,
                                      PROT_READ | PROT_WRITE,
                                      MAP_SHARED,
                                      staged->sharedFd,
                                      0);
            if (staged->sharedBase == MAP_FAILED) {
                NIXL_ERROR << "Failed to mmap UCX local staging file " << staged->sharedPath
                           << ": " << std::strerror(errno);
                staged->sharedBase = nullptr;
                cleanup_staged(0);
                return NIXL_ERR_BACKEND;
            }

            const cudaError_t register_ret =
                cudaHostRegister(staged->sharedBase,
                                 staged->sharedMappingSize,
                                 cudaHostRegisterDefault);
            if (register_ret != cudaSuccess) {
                NIXL_ERROR << "cudaHostRegister failed for UCX local staging file "
                           << staged->sharedPath << ": " << cudaGetErrorString(register_ret);
                cleanup_staged(0);
                return NIXL_ERR_BACKEND;
            }
            staged->sharedHostRegistered = true;

            for (size_t i = 0; i < staged->slots.size(); ++i) {
                staged->slots[i].hostAddr =
                    static_cast<char *>(staged->sharedBase) + (i * slot_span);
            }
        }

        for (size_t i = 0; i < staged->slots.size(); ++i) {
            nixlUcxStagedSlot &slot = staged->slots[i];
            slot.size = staged->slotSize;

            if (!staged->localSharedSlots) {
                const cudaError_t alloc_ret = cudaMallocHost(&slot.hostAddr, slot.size);
                if (alloc_ret != cudaSuccess) {
                    NIXL_ERROR << "cudaMallocHost failed for UCX VRAM staging slot " << i << ": "
                               << cudaGetErrorString(alloc_ret);
                    cleanup_staged(i);
                    return NIXL_ERR_BACKEND;
                }
            }

            const int reg_ret = uc->memReg(slot.hostAddr, slot.size, slot.mem, DRAM_SEG);
            if (reg_ret) {
                NIXL_ERROR << "UCX host staging memory registration failed for slot " << i;
                cleanup_staged(i + 1);
                return NIXL_ERR_BACKEND;
            }
            slot.ucxRegistered = true;

            slot.rkeyStr = uc->packRkey(slot.mem);
            if (slot.rkeyStr.empty()) {
                NIXL_ERROR << "UCX host staging rkey pack failed for slot " << i;
                cleanup_staged(i + 1);
                return NIXL_ERR_BACKEND;
            }
        }

        NIXL_INFO << "Registered UCX staged VRAM region gpu_base=" << (void *)mem.addr
                  << " gpu_len=" << mem.len << " gpu_dev=" << mem.devId
                  << " slot_size=" << staged->slotSize << " slots=" << staged->slots.size()
                  << " local_shared_slots=" << staged->localSharedSlots
                  << " shared_path=" << staged->sharedPath;
        registerStagedRegion(staged.get());
        out = staged.release();
        return NIXL_SUCCESS;
    }

    auto priv = std::make_unique<nixlUcxPrivateMetadata>();

    // TODO: Add nixl_mem check?
    const int ret = uc->memReg((void*) mem.addr, mem.len, priv->mem, nixl_mem);
    if (ret) {
        return NIXL_ERR_BACKEND;
    }
    priv->rkeyStr = uc->packRkey(priv->mem);

    if (priv->rkeyStr.empty()) {
        return NIXL_ERR_BACKEND;
    }
    out = priv.release();
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::deregisterMem (nixlBackendMD* meta)
{
    if (auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(meta)) {
        {
            // Check-idle and remove-from-list must be one critical section: the AM
            // handlers grant leases only on regions found in stagedRegions_ under
            // this same lock, so no new lease can appear between the check and the
            // removal. Lock order (stagedRegionMutex_ -> slotMutex) matches the AM
            // handlers.
            const std::lock_guard lock(stagedRegionMutex_);
            if (staged->hasActiveSlots()) {
                NIXL_ERROR << "Cannot deregister UCX staged VRAM region with active staging slots"
                           << " gpu_base=" << reinterpret_cast<void *>(staged->gpuBase)
                           << " gpu_len=" << staged->gpuLen
                           << " gpu_dev=" << staged->gpuDevId;
                return NIXL_ERR_NOT_ALLOWED;
            }
            stagedRegions_.erase(
                std::remove(stagedRegions_.begin(), stagedRegions_.end(), staged),
                stagedRegions_.end());
        }
        for (auto &slot : staged->slots) {
            if (slot.ucxRegistered) {
                uc->memDereg(slot.mem);
                slot.ucxRegistered = false;
            }
            if (!staged->localSharedSlots && slot.hostAddr != nullptr) {
                cudaFreeHost(slot.hostAddr);
                slot.hostAddr = nullptr;
            }
        }
        if (staged->sharedHostRegistered && staged->sharedBase != nullptr) {
            cudaHostUnregister(staged->sharedBase);
            staged->sharedHostRegistered = false;
        }
        if (staged->sharedBase != nullptr) {
            munmap(staged->sharedBase, staged->sharedMappingSize);
            staged->sharedBase = nullptr;
        }
        if (staged->sharedFd >= 0) {
            close(staged->sharedFd);
            staged->sharedFd = -1;
        }
        if (staged->unlinkSharedPath && !staged->sharedPath.empty()) {
            unlink(staged->sharedPath.c_str());
            staged->unlinkSharedPath = false;
        }
        delete staged;
        return NIXL_SUCCESS;
    }

    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    uc->memDereg(priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData (const nixlBackendMD* meta,
                                            std::string &str) const {
    if (const auto *staged = dynamic_cast<const nixlUcxStagedPrivateMetadata *>(meta)) {
        str = serializeStagedMetadata(*staged);
        return NIXL_SUCCESS;
    }

    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}

namespace {

[[nodiscard]] std::vector<nixl::ucx::rkey>
makePublicMetadataRkeys(const ucx_connection_ptr_t &conn, const size_t count, const void *buffer) {
    std::vector<nixl::ucx::rkey> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        result.emplace_back(*conn->getEp(i), buffer);
    }
    return result;
}

} // namespace

nixlUcxPublicMetadata::nixlUcxPublicMetadata(const ucx_connection_ptr_t &conn,
                                             std::vector<nixl::ucx::rkey> &&rkeys)
    : nixlBackendMD(false),
      conn(conn),
      rkeys_(std::move(rkeys)) {}

nixl_status_t
nixlUcxEngine::internalMDHelper (const nixl_blob_t &blob,
                                 const std::string &agent,
                                 nixlBackendMD* &output) {
    try {
        const auto conn = getConnection(agent);
        if (!conn) {
            // TODO: err: remote connection not found
            return NIXL_ERR_NOT_FOUND;
        }
        // nixlSerDes::_stringToBytes() was used to "unpack" blob here.
        output = new nixlUcxPublicMetadata(
            conn, makePublicMetadataRkeys(conn, uws.size(), blob.data()));
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::internalStagedMDHelper(const nixl_blob_t &blob,
                                      const std::string &agent,
                                      nixlBackendMD *&output) {
    try {
        const auto conn = getConnection(agent);
        if (!conn) {
            return NIXL_ERR_NOT_FOUND;
        }

        nixlSerDes ser_des;
        if (ser_des.importStr(blob) != NIXL_SUCCESS) {
            return NIXL_ERR_MISMATCH;
        }

        const std::string magic = ser_des.getStr("magic");
        if (magic != kUcxStagedMagic) {
            return NIXL_ERR_MISMATCH;
        }

        uintptr_t gpu_base = 0;
        size_t gpu_len = 0;
        uint64_t gpu_dev_id = 0;
        size_t slot_size = 0;
        size_t slot_count = 0;
        bool local_shared_slots = false;
        uint64_t local_shared_region_id = 0;
        size_t local_shared_mapping_size = 0;
        std::string host_id;
        std::string local_shared_region_cookie;

        if (ser_des.getBuf("gpu_base", &gpu_base, sizeof(gpu_base)) != NIXL_SUCCESS ||
            ser_des.getBuf("gpu_len", &gpu_len, sizeof(gpu_len)) != NIXL_SUCCESS ||
            ser_des.getBuf("gpu_dev", &gpu_dev_id, sizeof(gpu_dev_id)) != NIXL_SUCCESS ||
            ser_des.getBuf("slot_size", &slot_size, sizeof(slot_size)) != NIXL_SUCCESS ||
            ser_des.getBuf("slot_count", &slot_count, sizeof(slot_count)) != NIXL_SUCCESS) {
            return NIXL_ERR_MISMATCH;
        }
        host_id = ser_des.getStr("host_id");
        if (host_id.empty()) {
            return NIXL_ERR_MISMATCH;
        }
        (void)ser_des.getBuf("local_shared_slots",
                             &local_shared_slots,
                             sizeof(local_shared_slots));
        (void)ser_des.getBuf("local_shared_region_id",
                             &local_shared_region_id,
                             sizeof(local_shared_region_id));
        if (blob.find("local_shared_region_cookie") != std::string::npos) {
            local_shared_region_cookie = ser_des.getStr("local_shared_region_cookie");
        }
        const std::string local_shared_path = ser_des.getStr("local_shared_path");
        (void)ser_des.getBuf("local_shared_mapping_size",
                             &local_shared_mapping_size,
                             sizeof(local_shared_mapping_size));
        if (!local_shared_slots) {
            local_shared_region_id = 0;
            local_shared_mapping_size = 0;
            local_shared_region_cookie.clear();
        }

        std::vector<uintptr_t> slot_addrs;
        std::vector<std::vector<nixl::ucx::rkey>> slot_rkeys;
        slot_addrs.reserve(slot_count);
        slot_rkeys.reserve(slot_count);

        for (size_t i = 0; i < slot_count; ++i) {
            uintptr_t slot_addr = 0;
            if (ser_des.getBuf("slot_addr", &slot_addr, sizeof(slot_addr)) != NIXL_SUCCESS) {
                return NIXL_ERR_MISMATCH;
            }
            const std::string rkey_blob = ser_des.getStr("slot_rkey");
            if (rkey_blob.empty()) {
                return NIXL_ERR_MISMATCH;
            }

            slot_addrs.push_back(slot_addr);
            slot_rkeys.emplace_back(
                makePublicMetadataRkeys(conn, uws.size(), rkey_blob.data()));
        }

        auto *staged_output = new nixlUcxStagedPublicMetadata(conn,
                                                              agent,
                                                              gpu_base,
                                                              gpu_len,
                                                              gpu_dev_id,
                                                              std::move(host_id),
                                                              slot_size,
                                                              vramStagingConfig_.slotRequestWindow,
                                                              local_shared_slots,
                                                              local_shared_region_id,
                                                              std::move(local_shared_region_cookie),
                                                              local_shared_path,
                                                              local_shared_mapping_size,
                                                              std::move(slot_addrs),
                                                              std::move(slot_rkeys));
        output = staged_output;
        registerLocalSharedRegion(agent, staged_output);
        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

uint64_t
nixlUcxEngine::nextStagedTransferId() const noexcept {
    return nextStagedTransferId_.fetch_add(1, std::memory_order_relaxed);
}

nixl_status_t
nixlUcxEngine::registerPendingStagedReq(uint64_t transfer_id, nixlBackendReqH *handle) const {
    const std::lock_guard lock(stagedReqMutex_);
    const auto [it, inserted] = pendingStagedReqs_.emplace(transfer_id, handle);
    if (!inserted) {
        NIXL_ERROR << "Duplicate UCX staged transfer id " << transfer_id;
        return NIXL_ERR_NOT_ALLOWED;
    }
    return NIXL_SUCCESS;
}

void
nixlUcxEngine::unregisterPendingStagedReq(uint64_t transfer_id, nixlBackendReqH *handle) const {
    const std::lock_guard lock(stagedReqMutex_);
    const auto it = pendingStagedReqs_.find(transfer_id);
    if (it != pendingStagedReqs_.end() && it->second == handle) {
        pendingStagedReqs_.erase(it);
    }
}

void
nixlUcxEngine::completePendingStagedSlotGrant(const std::string &remote_agent,
                                              uint64_t transfer_id,
                                              uint64_t chunk_id,
                                              uint64_t slot_id,
                                              uint64_t lease_id,
                                              nixl_status_t status) const {
    {
        const std::lock_guard lock(stagedReqMutex_);
        const auto it = pendingStagedReqs_.find(transfer_id);
        if (it != pendingStagedReqs_.end()) {
            auto *handle = dynamic_cast<nixlUcxStagedBackendReqH *>(it->second);
            if (handle) {
                if (handle->markGrant(chunk_id, slot_id, lease_id, status)) {
                    return;
                }
                // markGrant logged the out-of-range chunk id; fall through and
                // release the unbound grant.
            } else {
                NIXL_WARN << "Received UCX staged slot grant for non-staged transfer id "
                          << transfer_id;
            }
        } else {
            NIXL_WARN << "Received UCX staged slot grant for unknown transfer id " << transfer_id
                      << " chunk_id=" << chunk_id;
        }
    }

    // The grant could not be bound to a live chunk (transfer gone, non-staged
    // handle, or out-of-range chunk id). Return the lease, otherwise the target
    // slot stays REMOTE_RESERVED until timeout reclaim. The zero gpu fields tell
    // the target to resolve the lease by slot/lease id alone.
    if (status == NIXL_SUCCESS) {
        sendStagedSlotRelease(remote_agent, transfer_id, chunk_id, slot_id, lease_id, 0, 0, 0);
    }
}

void
nixlUcxEngine::completePendingStagedReq(uint64_t transfer_id,
                                        uint64_t chunk_id,
                                        uint64_t lease_id,
                                        nixl_status_t status) const {
    const std::lock_guard lock(stagedReqMutex_);
    const auto it = pendingStagedReqs_.find(transfer_id);
    if (it == pendingStagedReqs_.end()) {
        NIXL_WARN << "Received UCX staged ACK for unknown transfer id " << transfer_id
                  << " chunk_id=" << chunk_id;
        return;
    }

    auto *handle = dynamic_cast<nixlUcxStagedBackendReqH *>(it->second);
    if (!handle) {
        NIXL_WARN << "Received UCX staged ACK for non-staged transfer id " << transfer_id;
        return;
    }

    handle->markAck(chunk_id, lease_id, status);
}

void
nixlUcxEngine::registerStagedRegion(nixlBackendMD *metadata) {
    const std::lock_guard lock(stagedRegionMutex_);
    stagedRegions_.push_back(metadata);
}

nixl_status_t
nixlUcxEngine::sendStagedSlotReq(const std::string &remote_agent,
                                 uint64_t transfer_id,
                                 uint64_t chunk_id,
                                 uintptr_t remote_gpu_addr,
                                 uint64_t remote_gpu_dev,
                                 size_t size,
                                 const std::unique_ptr<nixlUcxEp> &ep,
                                 nixlUcxReq *req) const {
    nixlSerDes ser_des;

    ser_des.addStr("name", localAgent);
    ser_des.addBuf("xfer_id", &transfer_id, sizeof(transfer_id));
    ser_des.addBuf("chunk_id", &chunk_id, sizeof(chunk_id));
    ser_des.addBuf("gpu_addr", &remote_gpu_addr, sizeof(remote_gpu_addr));
    ser_des.addBuf("gpu_dev", &remote_gpu_dev, sizeof(remote_gpu_dev));
    ser_des.addBuf("size", &size, sizeof(size));

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer, req](void *completed_request, void *ptr) {
        delete buffer;
        if ((req == nullptr) && (completed_request != nullptr)) {
            ucp_request_free(completed_request);
        }
    };

    NIXL_TRACE << "Sending UCX staged SLOT_REQ transfer_id=" << transfer_id
               << " chunk_id=" << chunk_id << " remote_agent=" << remote_agent
               << " bytes=" << size;

    return ep->sendAm(nixl::ucx::am_cb_op_t::STAGED_SLOT_REQ,
                      nullptr,
                      0,
                      (void *)buffer->data(),
                      buffer->size(),
                      UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_REPLY,
                      req,
                      deleter);
}

nixl_status_t
nixlUcxEngine::sendStagedSlotGrant(const std::string &remote_agent,
                                   uint64_t transfer_id,
                                   uint64_t chunk_id,
                                   uint64_t slot_id,
                                   uint64_t lease_id,
                                   nixl_status_t status,
                                   ucp_ep_h reply_ep) const {
    nixlSerDes ser_des;
    ser_des.addStr("name", localAgent);
    ser_des.addBuf("xfer_id", &transfer_id, sizeof(transfer_id));
    ser_des.addBuf("chunk_id", &chunk_id, sizeof(chunk_id));
    ser_des.addBuf("slot_id", &slot_id, sizeof(slot_id));
    ser_des.addBuf("lease_id", &lease_id, sizeof(lease_id));
    ser_des.addBuf("status", &status, sizeof(status));

    std::string *buffer = new std::string(ser_des.exportStr());

    NIXL_TRACE << "Sending UCX staged SLOT_GRANT transfer_id=" << transfer_id
               << " chunk_id=" << chunk_id << " remote_agent=" << remote_agent
               << " slot_id=" << slot_id << " lease_id=" << lease_id
               << " status=" << status;

    const nixl_status_t send_status =
        sendStagedControlAm(remote_agent,
                            reply_ep,
                            nixl::ucx::am_cb_op_t::STAGED_SLOT_GRANT,
                            buffer);
    if (send_status == NIXL_ERR_NOT_FOUND) {
        NIXL_ERROR << "Cannot send UCX staged SLOT_GRANT to unknown agent " << remote_agent
                   << " transfer_id=" << transfer_id << " chunk_id=" << chunk_id;
    }
    return send_status;
}

nixl_status_t
nixlUcxEngine::sendStagedSlotRelease(const std::string &remote_agent,
                                     uint64_t transfer_id,
                                     uint64_t chunk_id,
                                     uint64_t slot_id,
                                     uint64_t lease_id,
                                     uintptr_t remote_gpu_addr,
                                     uint64_t remote_gpu_dev,
                                     size_t size) const {
    const auto conn = getConnection(remote_agent);
    if (!conn) {
        NIXL_WARN << "Cannot send UCX staged SLOT_RELEASE to unknown agent " << remote_agent
                  << " transfer_id=" << transfer_id << " chunk_id=" << chunk_id;
        return NIXL_ERR_NOT_FOUND;
    }

    nixlSerDes ser_des;
    ser_des.addStr("name", localAgent);
    ser_des.addBuf("xfer_id", &transfer_id, sizeof(transfer_id));
    ser_des.addBuf("chunk_id", &chunk_id, sizeof(chunk_id));
    ser_des.addBuf("slot_id", &slot_id, sizeof(slot_id));
    ser_des.addBuf("lease_id", &lease_id, sizeof(lease_id));
    ser_des.addBuf("gpu_addr", &remote_gpu_addr, sizeof(remote_gpu_addr));
    ser_des.addBuf("gpu_dev", &remote_gpu_dev, sizeof(remote_gpu_dev));
    ser_des.addBuf("size", &size, sizeof(size));

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer](void *completed_request, void *ptr) {
        delete buffer;
        if (completed_request != nullptr) {
            ucp_request_free(completed_request);
        }
    };

    NIXL_TRACE << "Sending UCX staged SLOT_RELEASE transfer_id=" << transfer_id
               << " chunk_id=" << chunk_id << " remote_agent=" << remote_agent
               << " slot_id=" << slot_id << " lease_id=" << lease_id;

    return conn->getEp(0)->sendAm(nixl::ucx::am_cb_op_t::STAGED_SLOT_RELEASE,
                                  nullptr,
                                  0,
                                  (void *)buffer->data(),
                                  buffer->size(),
                                  UCP_AM_SEND_FLAG_EAGER,
                                  nullptr,
                                  deleter);
}

nixl_status_t
nixlUcxEngine::sendStagedWriteReady(const std::string &remote_agent,
                                    uint64_t transfer_id,
                                    uint64_t chunk_id,
                                    uint64_t remote_slot_id,
                                    uint64_t lease_id,
                                    uintptr_t remote_gpu_addr,
                                    uint64_t remote_gpu_dev,
                                    size_t size,
                                    const std::unique_ptr<nixlUcxEp> &ep,
                                    nixlUcxReq *req) const {
    nixlSerDes ser_des;

    ser_des.addStr("name", localAgent);
    ser_des.addBuf("xfer_id", &transfer_id, sizeof(transfer_id));
    ser_des.addBuf("chunk_id", &chunk_id, sizeof(chunk_id));
    ser_des.addBuf("slot_id", &remote_slot_id, sizeof(remote_slot_id));
    ser_des.addBuf("lease_id", &lease_id, sizeof(lease_id));
    ser_des.addBuf("gpu_addr", &remote_gpu_addr, sizeof(remote_gpu_addr));
    ser_des.addBuf("gpu_dev", &remote_gpu_dev, sizeof(remote_gpu_dev));
    ser_des.addBuf("size", &size, sizeof(size));

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer, req](void *completed_request, void *ptr) {
        delete buffer;
        if ((req == nullptr) && (completed_request != nullptr)) {
            ucp_request_free(completed_request);
        }
    };

    NIXL_TRACE << "Sending UCX staged WRITE_READY transfer_id=" << transfer_id
               << " chunk_id=" << chunk_id << " remote_agent=" << remote_agent
               << " slot_id=" << remote_slot_id << " lease_id=" << lease_id
               << " bytes=" << size;

    return ep->sendAm(nixl::ucx::am_cb_op_t::STAGED_WRITE_READY,
                      nullptr,
                      0,
                      (void *)buffer->data(),
                      buffer->size(),
                      UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_REPLY,
                      req,
                      deleter);
}

nixl_status_t
nixlUcxEngine::sendStagedLocalWriteReady(const std::string &remote_agent,
                                         uint64_t transfer_id,
                                         uint64_t chunk_id,
                                         uint64_t source_region_id,
                                         const std::string &source_region_cookie,
                                         uint64_t source_slot_id,
                                         uint64_t source_slot_generation,
                                         const std::string &source_shared_path,
                                         size_t source_slot_offset,
                                         size_t source_mapping_size,
                                         uintptr_t remote_gpu_addr,
                                         uint64_t remote_gpu_dev,
                                         size_t size,
                                         const std::unique_ptr<nixlUcxEp> &ep,
                                         nixlUcxReq *req) const {
    nixlSerDes ser_des;

    ser_des.addStr("name", localAgent);
    ser_des.addBuf("xfer_id", &transfer_id, sizeof(transfer_id));
    ser_des.addBuf("chunk_id", &chunk_id, sizeof(chunk_id));
    ser_des.addBuf("source_region_id", &source_region_id, sizeof(source_region_id));
    if (!source_region_cookie.empty()) {
        ser_des.addStr("source_region_cookie", source_region_cookie);
    }
    ser_des.addBuf("source_slot_id", &source_slot_id, sizeof(source_slot_id));
    ser_des.addBuf("source_slot_generation", &source_slot_generation, sizeof(source_slot_generation));
    ser_des.addStr("source_shared_path", source_shared_path);
    ser_des.addBuf("source_slot_offset", &source_slot_offset, sizeof(source_slot_offset));
    ser_des.addBuf("source_mapping_size", &source_mapping_size, sizeof(source_mapping_size));
    ser_des.addBuf("gpu_addr", &remote_gpu_addr, sizeof(remote_gpu_addr));
    ser_des.addBuf("gpu_dev", &remote_gpu_dev, sizeof(remote_gpu_dev));
    ser_des.addBuf("size", &size, sizeof(size));

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer, req](void *completed_request, void *ptr) {
        delete buffer;
        if ((req == nullptr) && (completed_request != nullptr)) {
            ucp_request_free(completed_request);
        }
    };

    NIXL_TRACE << "Sending UCX staged LOCAL_WRITE_READY transfer_id=" << transfer_id
               << " chunk_id=" << chunk_id << " remote_agent=" << remote_agent
               << " source_region_id=" << source_region_id
               << " source_slot_id=" << source_slot_id
               << " source_slot_generation=" << source_slot_generation
               << " bytes=" << size;

    return ep->sendAm(nixl::ucx::am_cb_op_t::STAGED_LOCAL_WRITE_READY,
                      nullptr,
                      0,
                      (void *)buffer->data(),
                      buffer->size(),
                      UCP_AM_SEND_FLAG_EAGER | UCP_AM_SEND_FLAG_REPLY,
                      req,
                      deleter);
}

nixl_status_t
nixlUcxEngine::sendStagedAck(const std::string &remote_agent,
                             uint64_t transfer_id,
                             uint64_t chunk_id,
                             uint64_t lease_id,
                             nixl_status_t status,
                             ucp_ep_h reply_ep) const {
    nixlSerDes ser_des;
    ser_des.addStr("name", localAgent);
    ser_des.addBuf("xfer_id", &transfer_id, sizeof(transfer_id));
    ser_des.addBuf("chunk_id", &chunk_id, sizeof(chunk_id));
    ser_des.addBuf("lease_id", &lease_id, sizeof(lease_id));
    ser_des.addBuf("status", &status, sizeof(status));

    std::string *buffer = new std::string(ser_des.exportStr());

    NIXL_TRACE << "Sending UCX staged ACK transfer_id=" << transfer_id
               << " chunk_id=" << chunk_id << " remote_agent=" << remote_agent
               << " lease_id=" << lease_id << " status=" << status;

    const nixl_status_t send_status =
        sendStagedControlAm(remote_agent,
                            reply_ep,
                            nixl::ucx::am_cb_op_t::STAGED_ACK,
                            buffer);
    if (send_status == NIXL_ERR_NOT_FOUND) {
        NIXL_ERROR << "Cannot send UCX staged ACK to unknown agent " << remote_agent
                   << " transfer_id=" << transfer_id << " chunk_id=" << chunk_id;
    }
    return send_status;
}

nixl_status_t
nixlUcxEngine::handleStagedSlotReq(const nixl_blob_t &message, ucp_ep_h reply_ep) const {
    nixlSerDes ser_des;
    if (ser_des.importStr(message) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize UCX staged SLOT_REQ message";
        return NIXL_ERR_MISMATCH;
    }

    const std::string remote_agent = ser_des.getStr("name");
    uint64_t transfer_id = 0;
    uint64_t chunk_id = 0;
    uintptr_t gpu_addr = 0;
    uint64_t gpu_dev = 0;
    size_t size = 0;

    nixl_status_t status =
        ser_des.getBuf("xfer_id", &transfer_id, sizeof(transfer_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("chunk_id", &chunk_id, sizeof(chunk_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("gpu_addr", &gpu_addr, sizeof(gpu_addr)) == NIXL_SUCCESS &&
                ser_des.getBuf("gpu_dev", &gpu_dev, sizeof(gpu_dev)) == NIXL_SUCCESS &&
                ser_des.getBuf("size", &size, sizeof(size)) == NIXL_SUCCESS ?
            NIXL_SUCCESS :
            NIXL_ERR_MISMATCH;

    uint64_t slot_id = 0;
    uint64_t lease_id = 0;
    if (remote_agent.empty()) {
        status = NIXL_ERR_MISMATCH;
    }

    if (status == NIXL_SUCCESS) {
        const std::lock_guard lock(stagedRegionMutex_);
        status = NIXL_ERR_NOT_FOUND;
        for (nixlBackendMD *region : stagedRegions_) {
            auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(region);
            if (!staged || staged->gpuDevId != gpu_dev ||
                !rangeCovers(staged->gpuBase, staged->gpuLen, gpu_addr, size)) {
                continue;
            }

            const nixlUcxStagedSlotGrant grant =
                staged->reserveRemoteSlot(remote_agent, transfer_id, chunk_id, gpu_addr, gpu_dev, size);
            status = grant.status;
            slot_id = grant.slotId;
            lease_id = grant.leaseId;
            break;
        }
    }

    if (!remote_agent.empty()) {
        const nixl_status_t send_status =
            sendStagedSlotGrant(remote_agent,
                                transfer_id,
                                chunk_id,
                                slot_id,
                                lease_id,
                                status,
                                reply_ep);
        if (send_status != NIXL_SUCCESS && send_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to send UCX staged SLOT_GRANT transfer_id=" << transfer_id
                       << " chunk_id=" << chunk_id << " status=" << send_status;
            return send_status;
        }
    }

    return status;
}

nixl_status_t
nixlUcxEngine::handleStagedSlotRelease(const nixl_blob_t &message) const {
    nixlSerDes ser_des;
    if (ser_des.importStr(message) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize UCX staged SLOT_RELEASE message";
        return NIXL_ERR_MISMATCH;
    }

    const std::string remote_agent = ser_des.getStr("name");
    uint64_t transfer_id = 0;
    uint64_t chunk_id = 0;
    uint64_t slot_id = 0;
    uint64_t lease_id = 0;
    uintptr_t gpu_addr = 0;
    uint64_t gpu_dev = 0;
    size_t size = 0;

    nixl_status_t status =
        ser_des.getBuf("xfer_id", &transfer_id, sizeof(transfer_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("chunk_id", &chunk_id, sizeof(chunk_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("slot_id", &slot_id, sizeof(slot_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("lease_id", &lease_id, sizeof(lease_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("gpu_addr", &gpu_addr, sizeof(gpu_addr)) == NIXL_SUCCESS &&
                ser_des.getBuf("gpu_dev", &gpu_dev, sizeof(gpu_dev)) == NIXL_SUCCESS &&
                ser_des.getBuf("size", &size, sizeof(size)) == NIXL_SUCCESS ?
            NIXL_SUCCESS :
            NIXL_ERR_MISMATCH;

    if (remote_agent.empty()) {
        return NIXL_ERR_MISMATCH;
    }

    if (status == NIXL_SUCCESS) {
        // size == 0 marks a release-by-id: the initiator no longer knows the gpu range
        // (e.g. the transfer was released before the grant arrived), so try every
        // region; the lease fields are sufficient to identify the slot.
        const bool by_id = (size == 0);
        const std::lock_guard lock(stagedRegionMutex_);
        status = NIXL_ERR_NOT_FOUND;
        for (nixlBackendMD *region : stagedRegions_) {
            auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(region);
            if (!staged) {
                continue;
            }
            if (!by_id &&
                (staged->gpuDevId != gpu_dev ||
                 !rangeCovers(staged->gpuBase, staged->gpuLen, gpu_addr, size))) {
                continue;
            }

            if (staged->releaseRemoteLease(remote_agent,
                                           transfer_id,
                                           chunk_id,
                                           slot_id,
                                           lease_id)) {
                status = NIXL_SUCCESS;
                break;
            }
            if (!by_id) {
                status = NIXL_ERR_MISMATCH;
                break;
            }
        }
    }

    if (status != NIXL_SUCCESS) {
        NIXL_WARN << "UCX staged SLOT_RELEASE did not release a lease transfer_id="
                  << transfer_id << " chunk_id=" << chunk_id << " slot_id=" << slot_id
                  << " lease_id=" << lease_id << " status=" << status;
    }
    return status;
}

nixl_status_t
nixlUcxEngine::handleStagedWriteReady(const nixl_blob_t &message, ucp_ep_h reply_ep) const {
    const bool profile_enabled = stagingProfileEnabled();
    const uint64_t callback_start_us = profile_enabled ? profileNowUs() : 0;
    uint64_t h2d_us = 0;
    nixlSerDes ser_des;
    if (ser_des.importStr(message) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize UCX staged WRITE_READY message";
        return NIXL_ERR_MISMATCH;
    }

    const std::string remote_agent = ser_des.getStr("name");
    uint64_t transfer_id = 0;
    uint64_t chunk_id = 0;
    uint64_t slot_id = 0;
    uint64_t lease_id = 0;
    uintptr_t gpu_addr = 0;
    uint64_t gpu_dev = 0;
    size_t size = 0;

    nixl_status_t status =
        ser_des.getBuf("xfer_id", &transfer_id, sizeof(transfer_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("chunk_id", &chunk_id, sizeof(chunk_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("slot_id", &slot_id, sizeof(slot_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("lease_id", &lease_id, sizeof(lease_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("gpu_addr", &gpu_addr, sizeof(gpu_addr)) == NIXL_SUCCESS &&
                ser_des.getBuf("gpu_dev", &gpu_dev, sizeof(gpu_dev)) == NIXL_SUCCESS &&
                ser_des.getBuf("size", &size, sizeof(size)) == NIXL_SUCCESS ?
            NIXL_SUCCESS :
            NIXL_ERR_MISMATCH;

    if (remote_agent.empty()) {
        status = NIXL_ERR_MISMATCH;
    }

    nixlBackendMD *h2d_region = nullptr;
    nixlUcxStagedReadyLease ready;
    if (status == NIXL_SUCCESS) {
        {
            const std::lock_guard lock(stagedRegionMutex_);
            status = NIXL_ERR_NOT_FOUND;
            for (nixlBackendMD *region : stagedRegions_) {
                auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(region);
                if (!staged || staged->gpuDevId != gpu_dev ||
                    !rangeCovers(staged->gpuBase, staged->gpuLen, gpu_addr, size)) {
                    continue;
                }

                status = staged->beginRemoteH2D(remote_agent,
                                                transfer_id,
                                                chunk_id,
                                                slot_id,
                                                lease_id,
                                                gpu_addr,
                                                gpu_dev,
                                                size,
                                                ready);
                if (status == NIXL_SUCCESS) {
                    h2d_region = region;
                }
                break;
            }
        }

        if (status == NIXL_SUCCESS) {
            if (vramStagingConfig_.targetH2DWorker) {
                StagedH2DTask task;
                task.region = h2d_region;
                task.hostAddr = ready.hostAddr;
                task.remoteAgent = remote_agent;
                task.replyEp = reply_ep;
                task.transferId = transfer_id;
                task.chunkId = chunk_id;
                task.slotId = slot_id;
                task.leaseId = lease_id;
                task.gpuAddr = gpu_addr;
                task.gpuDev = gpu_dev;
                task.size = size;
                task.profileEnabled = profile_enabled;
                task.callbackUs = profile_enabled ? profileNowUs() - callback_start_us : 0;

                status = enqueueStagedH2D(std::move(task));
                if (status == NIXL_SUCCESS) {
                    return NIXL_SUCCESS;
                }

                if (auto *region = dynamic_cast<nixlUcxStagedPrivateMetadata *>(h2d_region)) {
                    region->finishRemoteLease(slot_id, lease_id, status);
                }
            } else {
                int previous_device = -1;
                status = cudaSetDeviceForCopy(gpu_dev, previous_device);
                if (status == NIXL_SUCCESS) {
                    const uint64_t h2d_start_us = profile_enabled ? profileNowUs() : 0;
                    const cudaError_t cuda_ret =
                        cudaMemcpy((void *)gpu_addr, ready.hostAddr, size, cudaMemcpyHostToDevice);
                    if (profile_enabled) {
                        h2d_us = profileNowUs() - h2d_start_us;
                    }
                    if (cuda_ret != cudaSuccess) {
                        NIXL_ERROR << "UCX staged H2D failed for transfer_id=" << transfer_id
                                   << " chunk_id=" << chunk_id << ": "
                                   << cudaGetErrorString(cuda_ret);
                        status = NIXL_ERR_BACKEND;
                    }
                }
                cudaRestoreDevice(previous_device);
                if (auto *region = dynamic_cast<nixlUcxStagedPrivateMetadata *>(h2d_region)) {
                    region->finishRemoteLease(slot_id, lease_id, status);
                }
            }
        }
    }

    if (!remote_agent.empty()) {
        const nixl_status_t ack_status =
            sendStagedAck(remote_agent, transfer_id, chunk_id, lease_id, status, reply_ep);
        if (ack_status != NIXL_SUCCESS && ack_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to send UCX staged ACK transfer_id=" << transfer_id
                       << " chunk_id=" << chunk_id << " status=" << ack_status;
            return ack_status;
        }
    }

    if (profile_enabled && !vramStagingConfig_.targetH2DWorker && status == NIXL_SUCCESS) {
        const uint64_t ready_count = stagedProfileTargetReadyCount_.fetch_add(1) + 1;
        const uint64_t total_bytes = stagedProfileTargetBytes_.fetch_add(size) + size;
        const uint64_t total_h2d_us = stagedProfileTargetH2DUs_.fetch_add(h2d_us) + h2d_us;
        const uint64_t callback_us = profileNowUs() - callback_start_us;
        const uint64_t total_callback_us =
            stagedProfileTargetCallbackUs_.fetch_add(callback_us) + callback_us;
        if ((ready_count % 64) == 0) {
            NIXL_INFO << "UCX staged profile target ready_count=" << ready_count
                      << " bytes=" << total_bytes
                      << " h2d_total_us=" << total_h2d_us
                      << " h2d_avg_us=" << (total_h2d_us / ready_count)
                      << " callback_total_us=" << total_callback_us
                      << " callback_avg_us=" << (total_callback_us / ready_count);
        }
    }

    return status;
}

nixl_status_t
nixlUcxEngine::handleStagedLocalWriteReady(const nixl_blob_t &message, ucp_ep_h reply_ep) const {
    const bool profile_enabled = stagingProfileEnabled();
    const uint64_t callback_start_us = profile_enabled ? profileNowUs() : 0;
    uint64_t h2d_us = 0;

    nixlSerDes ser_des;
    if (ser_des.importStr(message) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize UCX staged LOCAL_WRITE_READY message";
        return NIXL_ERR_MISMATCH;
    }

    const std::string remote_agent = ser_des.getStr("name");
    uint64_t transfer_id = 0;
    uint64_t chunk_id = 0;
    uint64_t source_region_id = 0;
    std::string source_region_cookie;
    uint64_t source_slot_id = 0;
    uint64_t source_slot_generation = 0;
    size_t source_slot_offset = 0;
    size_t source_mapping_size = 0;
    uintptr_t gpu_addr = 0;
    uint64_t gpu_dev = 0;
    size_t size = 0;

    nixl_status_t status =
        ser_des.getBuf("xfer_id", &transfer_id, sizeof(transfer_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("chunk_id", &chunk_id, sizeof(chunk_id)) == NIXL_SUCCESS &&
                ser_des.getBuf("source_region_id",
                               &source_region_id,
                               sizeof(source_region_id)) == NIXL_SUCCESS ?
            NIXL_SUCCESS :
            NIXL_ERR_MISMATCH;
    if (status == NIXL_SUCCESS &&
        message.find("source_region_cookie") != std::string::npos) {
        source_region_cookie = ser_des.getStr("source_region_cookie");
    }
    if (status == NIXL_SUCCESS) {
        status =
            ser_des.getBuf("source_slot_id", &source_slot_id, sizeof(source_slot_id)) ==
                        NIXL_SUCCESS &&
                    ser_des.getBuf("source_slot_generation",
                                   &source_slot_generation,
                                   sizeof(source_slot_generation)) == NIXL_SUCCESS ?
                NIXL_SUCCESS :
                NIXL_ERR_MISMATCH;
    }
    const std::string source_shared_path =
        status == NIXL_SUCCESS ? ser_des.getStr("source_shared_path") : std::string();
    if (status == NIXL_SUCCESS) {
        status =
            ser_des.getBuf("source_slot_offset",
                           &source_slot_offset,
                           sizeof(source_slot_offset)) == NIXL_SUCCESS &&
                    ser_des.getBuf("source_mapping_size",
                                   &source_mapping_size,
                                   sizeof(source_mapping_size)) == NIXL_SUCCESS &&
                    ser_des.getBuf("gpu_addr", &gpu_addr, sizeof(gpu_addr)) == NIXL_SUCCESS &&
                    ser_des.getBuf("gpu_dev", &gpu_dev, sizeof(gpu_dev)) == NIXL_SUCCESS &&
                    ser_des.getBuf("size", &size, sizeof(size)) == NIXL_SUCCESS ?
                NIXL_SUCCESS :
                NIXL_ERR_MISMATCH;
    }

    if (remote_agent.empty() || source_region_id == 0 || source_slot_generation == 0 ||
        source_shared_path.empty() || size == 0 || source_slot_offset > source_mapping_size ||
        size > source_mapping_size - source_slot_offset) {
        status = NIXL_ERR_MISMATCH;
    }

    if (status == NIXL_SUCCESS &&
        !validateLocalSharedReady(remote_agent,
                                  source_region_id,
                                  source_region_cookie,
                                  source_shared_path,
                                  source_slot_id,
                                  source_slot_offset,
                                  source_mapping_size,
                                  size)) {
        NIXL_ERROR << "UCX local staged READY failed source metadata validation"
                   << " remote_agent=" << remote_agent
                   << " transfer_id=" << transfer_id
                   << " chunk_id=" << chunk_id
                   << " region_id=" << source_region_id
                   << " slot_id=" << source_slot_id
                   << " path=" << source_shared_path;
        status = NIXL_ERR_MISMATCH;
    }

    if (status == NIXL_SUCCESS) {
        {
            const std::lock_guard lock(stagedRegionMutex_);
            status = NIXL_ERR_NOT_FOUND;
            for (nixlBackendMD *region : stagedRegions_) {
                auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(region);
                if (!staged || staged->gpuDevId != gpu_dev ||
                    !rangeCovers(staged->gpuBase, staged->gpuLen, gpu_addr, size)) {
                    continue;
                }
                status = NIXL_SUCCESS;
                break;
            }
        }
    }

    std::shared_ptr<LocalSharedAttachment> attachment;
    void *host_addr = nullptr;

    if (status == NIXL_SUCCESS) {
        if (localStagingForceAttachFail()) {
            NIXL_WARN << "Forcing UCX local staged attach failure for transfer_id="
                      << transfer_id << " chunk_id=" << chunk_id;
            localSharedAttachFailures_.fetch_add(1, std::memory_order_relaxed);
            status = NIXL_ERR_BACKEND;
        } else {
            status = getLocalSharedAttachment(remote_agent,
                                              source_shared_path,
                                              source_mapping_size,
                                              attachment);
        }
        if (status == NIXL_SUCCESS) {
            host_addr = static_cast<char *>(attachment->base) + source_slot_offset;
        }
    }

    if (status == NIXL_SUCCESS) {
        int previous_device = -1;
        status = cudaSetDeviceForCopy(gpu_dev, previous_device);
        if (status == NIXL_SUCCESS) {
            const uint64_t h2d_start_us = profile_enabled ? profileNowUs() : 0;
            const cudaError_t cuda_ret =
                cudaMemcpy((void *)gpu_addr, host_addr, size, cudaMemcpyHostToDevice);
            if (profile_enabled) {
                h2d_us = profileNowUs() - h2d_start_us;
            }
            if (cuda_ret != cudaSuccess) {
                NIXL_ERROR << "UCX local staged H2D failed for transfer_id=" << transfer_id
                           << " chunk_id=" << chunk_id << ": "
                           << cudaGetErrorString(cuda_ret);
                status = NIXL_ERR_BACKEND;
            }
        }
        cudaRestoreDevice(previous_device);
    }

    if (!remote_agent.empty()) {
        const nixl_status_t ack_status =
            sendStagedAck(remote_agent,
                          transfer_id,
                          chunk_id,
                          source_slot_generation,
                          status,
                          reply_ep);
        if (ack_status != NIXL_SUCCESS && ack_status != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to send UCX local staged ACK transfer_id=" << transfer_id
                       << " chunk_id=" << chunk_id << " status=" << ack_status;
            return ack_status;
        }
    }

    if (profile_enabled) {
        const uint64_t ready_count = stagedProfileLocalReadyCount_.fetch_add(1) + 1;
        const uint64_t error_count =
            status == NIXL_SUCCESS ? stagedProfileLocalErrors_.load(std::memory_order_relaxed) :
                                     stagedProfileLocalErrors_.fetch_add(1) + 1;
        const uint64_t total_bytes = stagedProfileLocalBytes_.fetch_add(size) + size;
        const uint64_t total_h2d_us = stagedProfileLocalH2DUs_.fetch_add(h2d_us) + h2d_us;
        const uint64_t callback_us = profileNowUs() - callback_start_us;
        const uint64_t total_callback_us =
            stagedProfileLocalCallbackUs_.fetch_add(callback_us) + callback_us;
        if (status != NIXL_SUCCESS || ready_count == 1 || (ready_count % 64) == 0) {
            NIXL_INFO << "UCX local staged profile target ready_count=" << ready_count
                      << " errors=" << error_count
                      << " bytes=" << total_bytes
                      << " h2d_total_us=" << total_h2d_us
                      << " h2d_avg_us=" << (total_h2d_us / ready_count)
                      << " callback_total_us=" << total_callback_us
                      << " callback_avg_us=" << (total_callback_us / ready_count)
                      << " attach_cache_hits="
                      << localSharedAttachCacheHits_.load(std::memory_order_relaxed)
                      << " attach_cache_misses="
                      << localSharedAttachCacheMisses_.load(std::memory_order_relaxed)
                      << " attach_failures="
                      << localSharedAttachFailures_.load(std::memory_order_relaxed)
                      << " attach_total_us="
                      << localSharedAttachUs_.load(std::memory_order_relaxed);
        }
    }

    return status;
}

nixl_status_t
nixlUcxEngine::loadLocalMD (nixlBackendMD* input,
                            nixlBackendMD* &output)
{
    if (auto *staged = dynamic_cast<nixlUcxStagedPrivateMetadata *>(input)) {
        return internalStagedMDHelper(serializeStagedMetadata(*staged), localAgent, output);
    }

    nixlUcxPrivateMetadata* input_md = (nixlUcxPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD (const nixlBlobDesc &input,
                                           const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent,
                                           nixlBackendMD* &output)
{
    if (vramStagingEnabled() && nixl_mem == VRAM_SEG) {
        if (!isStagedMetadataBlob(input.metaInfo)) {
            NIXL_ERROR << "UCX VRAM staging expected staged VRAM metadata from " << remote_agent;
            return NIXL_ERR_MISMATCH;
        }
        return internalStagedMDHelper(input.metaInfo, remote_agent, output);
    }

    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD (nixlBackendMD* input) {
    unregisterLocalSharedRegion(input);
    delete input;
    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

size_t
nixlUcxEngine::getWorkerId(const nixl_opt_b_args_t *opt_args) const noexcept {
    if (opt_args) {
        const std::optional<size_t> worker_id = getWorkerIdFromOptArgs(*opt_args);
        if (worker_id) {
            return *worker_id;
        }
    }

    auto it = tlsSharedWorkerMap().find(this);
    if (it == tlsSharedWorkerMap().end()) {
        const size_t index = sharedWorkerIndex_.fetch_add(1) % getSharedWorkersSize();
        it = tlsSharedWorkerMap().emplace(this, index).first;
        NIXL_DEBUG << "engine " << this << " bound shared worker " << index << " to thread "
                   << std::this_thread::get_id();
    }
    return it->second;
}

std::optional<size_t>
nixlUcxEngine::getWorkerIdFromOptArgs(const nixl_opt_b_args_t &opt_args) const noexcept {
    constexpr std::string_view worker_id_key = "worker_id=";
    size_t pos = opt_args.customParam.find(worker_id_key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }

    try {
        size_t worker_id = std::stoull(opt_args.customParam.substr(pos + worker_id_key.length()));

        if (worker_id >= getSharedWorkersSize()) {
            NIXL_WARN << "Invalid worker_id " << worker_id << " (must be < "
                      << getSharedWorkersSize() << ")";
            return std::nullopt;
        }

        return worker_id;
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Failed to parse worker_id from customParam: " << e.what();
        return std::nullopt;
    }
}

nixl_status_t nixlUcxEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    if (local.descCount() == 0 || remote.descCount() == 0) {
        NIXL_ERROR << "Local or remote descriptor list is empty";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (vramStagingEnabled() && (local.getType() == VRAM_SEG || remote.getType() == VRAM_SEG)) {
        if (operation != NIXL_WRITE) {
            NIXL_ERROR << "UCX VRAM staging currently supports NIXL_WRITE only";
            return NIXL_ERR_NOT_SUPPORTED;
        }
        if (local.getType() != VRAM_SEG || remote.getType() != VRAM_SEG) {
            NIXL_ERROR << "UCX VRAM staging requires both local and remote descriptors to be VRAM";
            return NIXL_ERR_NOT_SUPPORTED;
        }

        constexpr size_t worker_id = 0;
        handle = new nixlUcxStagedBackendReqH(
            getWorker(worker_id).get(), worker_id, nextStagedTransferId());
        return NIXL_SUCCESS;
    }

    const size_t worker_id = getWorkerId(opt_args);
    /* TODO: try to get from a pool first */
    handle = new nixlUcxBackendReqH(getWorker(worker_id).get(), worker_id);

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::estimateXferCost (const nixl_xfer_op_t &operation,
                                               const nixl_meta_dlist_t &local,
                                               const nixl_meta_dlist_t &remote,
                                               const std::string &remote_agent,
                                               nixlBackendReqH* const &handle,
                                               std::chrono::microseconds &duration,
                                               std::chrono::microseconds &err_margin,
                                               nixl_cost_t &method,
                                               const nixl_opt_args_t* opt_args) const
{
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const size_t worker_id = int_handle->getWorkerId();

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Local (" << local.descCount() << ") and remote (" << remote.descCount()
                   << ") descriptor lists differ in size for cost estimation";
        return NIXL_ERR_MISMATCH;
    }

    duration = std::chrono::microseconds(0);
    err_margin = std::chrono::microseconds(0);

    if (local.descCount() == 0) {
        // Nothing to do, use a default value
        method = nixl_cost_t::ANALYTICAL_BACKEND;
        return NIXL_SUCCESS;
    }

    if (vramStagingEnabled() && (local.getType() == VRAM_SEG || remote.getType() == VRAM_SEG)) {
        if (local.getType() != VRAM_SEG || remote.getType() != VRAM_SEG) {
            return NIXL_ERR_NOT_SUPPORTED;
        }

        for (int i = 0; i < local.descCount(); i++) {
            const size_t lsize = local[i].len;
            const size_t rsize = remote[i].len;
            const auto rmd = dynamic_cast<nixlUcxStagedPublicMetadata *>(remote[i].metadataP);

            if (!rmd || lsize != rsize) {
                return NIXL_ERR_MISMATCH;
            }

            std::chrono::microseconds msg_duration;
            std::chrono::microseconds msg_err_margin;
            nixl_cost_t msg_method;
            const nixl_status_t ret = rmd->conn->getEp(worker_id)->estimateCost(
                lsize, msg_duration, msg_err_margin, msg_method);
            if (ret != NIXL_SUCCESS) {
                return ret;
            }

            duration += msg_duration;
            err_margin += msg_err_margin;
            method = msg_method;
        }
        return NIXL_SUCCESS;
    }

    for (int i = 0; i < local.descCount(); i++) {
        const size_t lsize = local[i].len;
        const size_t rsize = remote[i].len;

        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);

        NIXL_ASSERT(lmd && rmd) << "No metadata found in descriptor lists at index " << i << " during cost estimation";
        NIXL_ASSERT(lsize == rsize) << "Local size (" << lsize << ") != Remote size (" << rsize
                                    << ") at index " << i << " during cost estimation";

        std::chrono::microseconds msg_duration;
        std::chrono::microseconds msg_err_margin;
        nixl_cost_t msg_method;
        const nixl_status_t ret = rmd->conn->getEp(worker_id)->estimateCost(
            lsize, msg_duration, msg_err_margin, msg_method);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Worker failed to estimate cost for segment " << i << " status: " << ret;
            return ret;
        }

        duration += msg_duration;
        err_margin += msg_err_margin;
        method = msg_method;
    }

    return NIXL_SUCCESS;
}

nixlUcxEngine::batchResult
nixlUcxEngine::sendXferRangeBatch(nixlUcxEp &ep,
                                  nixl_xfer_op_t operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  size_t worker_id,
                                  size_t start_idx,
                                  size_t end_idx) {
    batchResult result = {NIXL_SUCCESS, 0, nullptr};

    for (size_t i = start_idx; i < end_idx; ++i) {
        void *laddr = (void *)local[i].addr;
        size_t lsize = local[i].len;
        uint64_t raddr = static_cast<uint64_t>(remote[i].addr);
        NIXL_ASSERT(lsize == remote[i].len);

        const auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        auto &rmd_ep = rmd->conn->getEp(worker_id);
        if (__builtin_expect(rmd_ep.get() != &ep, 0)) {
            break;
        }

        ++result.size;
        nixlUcxReq req;
        const nixl_status_t ret = operation == NIXL_READ ?
            ep.read(raddr, rmd->getRkey(worker_id), laddr, lmd->mem, lsize, req) :
            ep.write(laddr, lmd->mem, raddr, rmd->getRkey(worker_id), lsize, req);

        if (ret == NIXL_IN_PROG) {
            if (__builtin_expect(result.req != nullptr, 1)) {
                ucp_request_free(result.req);
            }
            result.req = req;
        } else if (ret != NIXL_SUCCESS) {
            result.status = ret;
            if (result.req != nullptr) {
                ucp_request_free(result.req);
                result.req = nullptr;
            }
            break;
        }
    }

    if (result.status == NIXL_SUCCESS && result.req) {
        result.status = NIXL_IN_PROG;
    }
    return result;
}

nixl_status_t
nixlUcxEngine::sendXferRange(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *handle,
                             size_t start_idx,
                             size_t end_idx) const {
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const size_t worker_id = int_handle->getWorkerId();

    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        return NIXL_ERR_INVALID_PARAM;
    }

    /* Assuming we have a single EP, we need 3 requests: one pending request,
     * one flush request, and one notification request */
    int_handle->reserve(3);

    for (size_t i = start_idx; i < end_idx;) {
        /* Send requests to a single EP */
        const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        auto &ep = rmd->conn->getEp(worker_id);
        const batchResult result =
            sendXferRangeBatch(*ep, operation, local, remote, worker_id, i, end_idx);

        /* Append a single pending request for the entire EP batch */
        const nixl_status_t ret = int_handle->append(result.status, result.req, rmd->conn);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }

        i += result.size;
    }

    /*
     * Flush keeps int_handle non-empty until the operation is actually
     * completed, which can happen after local requests completion.
     * We need to flush all distinct connections to ensure that the operation
     * is actually completed.
     */
    for (auto &conn : int_handle->getConnections()) {
        nixlUcxReq req;
        const nixl_status_t ret = conn->getEp(worker_id)->flushEp(req);
        if (int_handle->append(ret, req, conn) != NIXL_SUCCESS) {
            return ret;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::postStagedWrite(const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               nixlBackendReqH *handle,
                               const nixl_opt_b_args_t *opt_args) const {
    auto *staged_handle = dynamic_cast<nixlUcxStagedBackendReqH *>(handle);
    if (!staged_handle) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (staged_handle->state != nixlUcxStagedBackendReqH::State::INIT) {
        NIXL_ERROR << "UCX VRAM staging request repost is not supported yet";
        return NIXL_ERR_NOT_ALLOWED;
    }

    const int local_desc_count = local.descCount();
    const int remote_desc_count = remote.descCount();
    if (local_desc_count <= 0 || remote_desc_count <= 0 ||
        local_desc_count != remote_desc_count) {
        return NIXL_ERR_INVALID_PARAM;
    }

    staged_handle->remoteAgent = remote_agent;
    staged_handle->totalSize = 0;

    const size_t desc_count = static_cast<size_t>(local_desc_count);
    for (size_t desc_id = 0; desc_id < desc_count; ++desc_id) {
        const auto &local_desc = local[desc_id];
        const auto &remote_desc = remote[desc_id];
        if (local_desc.len != remote_desc.len || local_desc.len == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }

        auto *lmd = dynamic_cast<nixlUcxStagedPrivateMetadata *>(local_desc.metadataP);
        auto *rmd = dynamic_cast<nixlUcxStagedPublicMetadata *>(remote_desc.metadataP);
        if (!lmd || !rmd) {
            NIXL_ERROR << "UCX VRAM staging metadata mismatch for descriptor " << desc_id;
            return NIXL_ERR_MISMATCH;
        }

        const size_t desc_size = local_desc.len;
        if (!rangeCovers(lmd->gpuBase, lmd->gpuLen, local_desc.addr, desc_size) ||
            !rangeCovers(rmd->gpuBase, rmd->gpuLen, remote_desc.addr, desc_size)) {
            NIXL_ERROR << "UCX VRAM staging descriptor " << desc_id
                       << " is outside registered region";
            return NIXL_ERR_INVALID_PARAM;
        }

        const size_t chunk_size = std::min(lmd->slotSize, rmd->slotSize);
        if (chunk_size == 0) {
            return NIXL_ERR_INVALID_PARAM;
        }

        if (rmd->slotAddrs.empty() || rmd->slotRkeys.empty()) {
            return NIXL_ERR_MISMATCH;
        }
        const bool local_shared_write = vramStagingConfig_.localStaging &&
                                        lmd->localSharedSlots &&
                                        rmd->localSharedSlots &&
                                        !rmd->hostId.empty() &&
                                        rmd->hostId == lmd->hostId;

        if (desc_size > std::numeric_limits<size_t>::max() - staged_handle->totalSize) {
            return NIXL_ERR_INVALID_PARAM;
        }
        staged_handle->totalSize += desc_size;

        for (size_t offset = 0; offset < desc_size; offset += chunk_size) {
            const size_t this_chunk_size = std::min(chunk_size, desc_size - offset);
            auto chunk = std::make_unique<nixlUcxStagedChunk>(
                static_cast<uint64_t>(staged_handle->chunks.size()),
                local_desc.addr + offset,
                local_desc.devId,
                remote_desc.addr + offset,
                remote_desc.devId,
                this_chunk_size,
                lmd,
                rmd);
            chunk->localSharedWrite = local_shared_write;
            if (local_shared_write) {
                ++staged_handle->profile.localSharedChunks;
                staged_handle->profile.localSharedBytes += this_chunk_size;
            }
            chunk->req =
                std::make_unique<nixlUcxBackendReqH>(staged_handle->getWorker(),
                                                     staged_handle->getWorkerId());
            chunk->req->reserve(3);
            staged_handle->chunks.push_back(std::move(chunk));
        }
    }

    if (staged_handle->chunks.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t ret = registerPendingStagedReq(staged_handle->transferId, staged_handle);
    if (ret != NIXL_SUCCESS) {
        staged_handle->release();
        return ret;
    }
    staged_handle->pendingRegistered = true;
    staged_handle->state = nixlUcxStagedBackendReqH::State::RUNNING;

    if (opt_args && opt_args->hasNotif) {
        staged_handle->notif.emplace(remote_agent, opt_args->notifMsg);
    }

    staged_handle->profile.bytes = staged_handle->totalSize;
    staged_handle->profile.startUs = profileNowUs();

    return checkStagedXfer(staged_handle);
}

nixl_status_t
nixlUcxEngine::checkStagedXfer(nixlBackendReqH *handle) const {
    auto *staged_handle = dynamic_cast<nixlUcxStagedBackendReqH *>(handle);
    if (!staged_handle) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto fail = [&](nixl_status_t status) {
        staged_handle->lastStatus = status;
        staged_handle->state = nixlUcxStagedBackendReqH::State::FAILED;
        // Unregister first: grants/ACKs arriving after this point take the
        // unknown-transfer path (which releases granted slots) instead of writing
        // into chunks while we tear them down. unregisterPendingStagedReq serializes
        // with in-flight callbacks via stagedReqMutex_.
        if (staged_handle->pendingRegistered) {
            unregisterPendingStagedReq(staged_handle->transferId, staged_handle);
            staged_handle->pendingRegistered = false;
        }
        for (const auto &chunk_ptr : staged_handle->chunks) {
            nixlUcxStagedChunk &chunk = *chunk_ptr;
            // A grant that arrived but was not yet processed by the state machine
            // holds a target lease even though remoteSlotHeld is still false.
            const bool granted_unprocessed = !chunk.remoteSlotHeld &&
                chunk.grantArrived.load() && chunk.grantStatus.load() == NIXL_SUCCESS;
            if (chunk.remoteSlotHeld || granted_unprocessed) {
                sendStagedSlotRelease(staged_handle->remoteAgent,
                                      staged_handle->transferId,
                                      chunk.id,
                                      chunk.remoteSlotId,
                                      chunk.leaseId,
                                      chunk.remoteGpuAddr,
                                      chunk.remoteGpuDev,
                                      chunk.size);
                staged_handle->releaseRemoteSlot(chunk);
            }
        }
        staged_handle->releaseAllChunkSlots();
        staged_handle->release();
        return status;
    };

    auto log_profile = [&]() {
        if (!stagingProfileEnabled() || staged_handle->profileLogged) {
            return;
        }
        staged_handle->profileLogged = true;
        const uint64_t total_us = profileNowUs() - staged_handle->profile.startUs;
        double gib_per_sec = 0.0;
        if (total_us != 0) {
            gib_per_sec = static_cast<double>(staged_handle->profile.bytes) /
                          (static_cast<double>(total_us) / 1000000.0) /
                          static_cast<double>(1024 * 1024 * 1024);
        }
        NIXL_INFO << "UCX staged profile initiator transfer_id="
                  << staged_handle->transferId
                  << " chunks=" << staged_handle->chunks.size()
                  << " bytes=" << staged_handle->profile.bytes
                  << " total_us=" << total_us
                  << " gib_per_sec=" << gib_per_sec
                  << " slot_req_sent=" << staged_handle->profile.slotReqSent
                  << " slot_grant_success=" << staged_handle->profile.slotGrantSuccess
                  << " slot_grant_inprog=" << staged_handle->profile.slotGrantInProg
                  << " local_slot_miss=" << staged_handle->profile.localSlotMiss
                  << " remote_window_miss=" << staged_handle->profile.remoteWindowMiss
                  << " stale_grant_releases=" << staged_handle->profile.staleGrantReleases
                  << " local_shared_chunks=" << staged_handle->profile.localSharedChunks
                  << " local_shared_bytes=" << staged_handle->profile.localSharedBytes
                  << " local_shared_ack_errors="
                  << staged_handle->profile.localSharedAckErrors
                  << " local_shared_fallbacks="
                  << staged_handle->profile.localSharedFallbacks
                  << " rdma_write_posted=" << staged_handle->profile.rdmaWritePosted
                  << " flush_posted=" << staged_handle->profile.flushPosted
                  << " ready_sent=" << staged_handle->profile.readySent
                  << " ack_received=" << staged_handle->profile.ackReceived
                  << " grant_wait_us=" << staged_handle->profile.grantWaitUs
                  << " d2h_us=" << staged_handle->profile.d2hUs
                  << " rdma_flush_wait_us=" << staged_handle->profile.rdmaFlushWaitUs
                  << " ready_wait_us=" << staged_handle->profile.readyWaitUs
                  << " ack_wait_us=" << staged_handle->profile.ackWaitUs;
    };

    auto send_slot_request = [&](nixlUcxStagedChunk &chunk) -> nixl_status_t {
        if (!chunk.remoteMetadata || !chunk.remoteMetadata->tryAcquireSlotWindow()) {
            ++staged_handle->profile.remoteWindowMiss;
            return NIXL_IN_PROG;
        }
        chunk.remoteWindowHeld = true;

        const auto conn = getConnection(staged_handle->remoteAgent);
        if (!conn) {
            staged_handle->releaseRemoteWindow(chunk);
            return NIXL_ERR_NOT_FOUND;
        }

        chunk.grantArrived.store(false);
        chunk.grantStatus.store(NIXL_IN_PROG);
        // Stamp before the send: grant_is_stale assumes this time precedes the
        // target's grant timestamp, which only holds if it is taken before the
        // SLOT_REQ leaves this process.
        chunk.slotReqPostedUs = profileNowUs();

        nixlUcxReq req = nullptr;
        const nixl_status_t send_status =
            sendStagedSlotReq(staged_handle->remoteAgent,
                              staged_handle->transferId,
                              chunk.id,
                              chunk.remoteGpuAddr,
                              chunk.remoteGpuDev,
                              chunk.size,
                              conn->getEp(staged_handle->getWorkerId()),
                              &req);
        const nixl_status_t append_status = chunk.req->append(send_status, req, conn);
        if (append_status != NIXL_SUCCESS) {
            chunk.slotReqPostedUs = 0;
            staged_handle->releaseRemoteWindow(chunk);
            return append_status;
        }

        ++staged_handle->profile.slotReqSent;
        chunk.state = nixlUcxStagedChunk::State::SLOT_REQ_POSTED;
        return NIXL_SUCCESS;
    };

    auto get_source_d2h_stream = [&](uint64_t gpu_dev, cudaStream_t &stream) -> nixl_status_t {
        auto &streams = staged_handle->sourceD2HStreams[gpu_dev];
        if (streams.streams.empty()) {
            int previous_device = -1;
            nixl_status_t status = cudaSetDeviceForCopy(gpu_dev, previous_device);
            if (status != NIXL_SUCCESS) {
                return status;
            }

            const size_t stream_count = std::max<size_t>(1, vramStagingConfig_.cudaCopyStreams);
            streams.streams.reserve(stream_count);
            for (size_t i = 0; i < stream_count; ++i) {
                cudaStream_t new_stream = nullptr;
                const cudaError_t cuda_ret =
                    cudaStreamCreateWithFlags(&new_stream, cudaStreamNonBlocking);
                if (cuda_ret != cudaSuccess) {
                    cudaRestoreDevice(previous_device);
                    NIXL_ERROR << "UCX staged source D2H stream creation failed for gpu_dev="
                               << gpu_dev << ": " << cudaGetErrorString(cuda_ret);
                    return NIXL_ERR_BACKEND;
                }
                streams.streams.push_back(new_stream);
            }
            cudaRestoreDevice(previous_device);
        }

        stream = streams.streams[streams.next++ % streams.streams.size()];
        return NIXL_SUCCESS;
    };

    auto start_local_d2h_prefetch = [&](nixlUcxStagedChunk &chunk) -> nixl_status_t {
        if (!chunk.localMetadata) {
            return NIXL_ERR_MISMATCH;
        }

        const auto local_slot_id = chunk.localMetadata->acquireSlot();
        if (!local_slot_id) {
            ++staged_handle->profile.localSlotMiss;
            return NIXL_IN_PROG;
        }

        chunk.localSlotId = *local_slot_id;
        chunk.localSlotHeld = true;

        cudaStream_t stream = nullptr;
        nixl_status_t ret = get_source_d2h_stream(chunk.localGpuDev, stream);
        if (ret != NIXL_SUCCESS) {
            staged_handle->releaseLocalSlot(chunk);
            return ret;
        }

        int previous_device = -1;
        ret = cudaSetDeviceForCopy(chunk.localGpuDev, previous_device);
        if (ret != NIXL_SUCCESS) {
            staged_handle->releaseLocalSlot(chunk);
            return ret;
        }

        cudaEvent_t event = nullptr;
        cudaError_t cuda_ret = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (cuda_ret != cudaSuccess) {
            cudaRestoreDevice(previous_device);
            staged_handle->releaseLocalSlot(chunk);
            NIXL_ERROR << "UCX staged source D2H event creation failed for transfer_id="
                       << staged_handle->transferId << " chunk_id=" << chunk.id << ": "
                       << cudaGetErrorString(cuda_ret);
            return NIXL_ERR_BACKEND;
        }

        nixlUcxStagedSlot &local_slot = chunk.localMetadata->slots[chunk.localSlotId];
        const auto local_gpu_addr = reinterpret_cast<void *>(chunk.localGpuAddr);
        cuda_ret = cudaMemcpyAsync(local_slot.hostAddr,
                                   local_gpu_addr,
                                   chunk.size,
                                   cudaMemcpyDeviceToHost,
                                   stream);
        if (cuda_ret == cudaSuccess) {
            cuda_ret = cudaEventRecord(event, stream);
        }
        cudaRestoreDevice(previous_device);

        if (cuda_ret != cudaSuccess) {
            NIXL_ERROR << "UCX staged async D2H submit failed for transfer_id="
                       << staged_handle->transferId << " chunk_id=" << chunk.id << ": "
                       << cudaGetErrorString(cuda_ret);
            cudaEventDestroy(event);
            staged_handle->releaseLocalSlot(chunk);
            return NIXL_ERR_BACKEND;
        }

        chunk.d2hEvent = event;
        chunk.d2hPostedUs = profileNowUs();
        chunk.state = nixlUcxStagedChunk::State::LOCAL_D2H_POSTED;
        return NIXL_SUCCESS;
    };

    auto check_local_d2h_prefetch = [&](nixlUcxStagedChunk &chunk) -> nixl_status_t {
        if (!chunk.d2hEvent) {
            return NIXL_ERR_MISMATCH;
        }

        const cudaError_t query = cudaEventQuery(chunk.d2hEvent);
        if (query == cudaErrorNotReady) {
            return NIXL_IN_PROG;
        }

        cudaEventDestroy(chunk.d2hEvent);
        chunk.d2hEvent = nullptr;
        if (query != cudaSuccess) {
            NIXL_ERROR << "UCX staged async D2H failed for transfer_id="
                       << staged_handle->transferId << " chunk_id=" << chunk.id << ": "
                       << cudaGetErrorString(query);
            staged_handle->releaseLocalSlot(chunk);
            return NIXL_ERR_BACKEND;
        }

        if (chunk.d2hPostedUs != 0) {
            staged_handle->profile.d2hUs += profileNowUs() - chunk.d2hPostedUs;
        }
        chunk.state = nixlUcxStagedChunk::State::LOCAL_READY;
        return NIXL_SUCCESS;
    };

    // Initiator-side stale-grant guard. The target reclaims REMOTE_RESERVED leases
    // after leaseTimeoutMs, so refuse to use a grant once half of that budget has
    // elapsed. Age is measured from slotReqPostedUs, which starts before the
    // target's grant timestamp, so the guard errs on the conservative side. Both
    // sides are assumed to run with the same staging_lease_timeout_ms. This narrows,
    // but cannot fully close, the corruption window: a process frozen after the RDMA
    // write was posted can still write into a reclaimed slot.
    const auto grant_is_stale = [&](const nixlUcxStagedChunk &chunk) -> bool {
        if (vramStagingConfig_.leaseTimeoutMs == 0 || chunk.slotReqPostedUs == 0) {
            return false;
        }
        const uint64_t guard_us =
            static_cast<uint64_t>(vramStagingConfig_.leaseTimeoutMs) * 1000 / 2;
        return profileNowUs() - chunk.slotReqPostedUs >= guard_us;
    };

    auto release_stale_grant = [&](nixlUcxStagedChunk &chunk) {
        ++staged_handle->profile.staleGrantReleases;
        NIXL_WARN << "Releasing stale UCX staged slot grant transfer_id="
                  << staged_handle->transferId << " chunk_id=" << chunk.id
                  << " slot_id=" << chunk.remoteSlotId << " lease_id=" << chunk.leaseId
                  << " age_us=" << (profileNowUs() - chunk.slotReqPostedUs);
        sendStagedSlotRelease(staged_handle->remoteAgent,
                              staged_handle->transferId,
                              chunk.id,
                              chunk.remoteSlotId,
                              chunk.leaseId,
                              chunk.remoteGpuAddr,
                              chunk.remoteGpuDev,
                              chunk.size);
        staged_handle->releaseRemoteSlot(chunk);
        chunk.grantArrived.store(false);
        chunk.grantStatus.store(NIXL_IN_PROG);
        chunk.leaseId = 0;
        if (vramStagingConfig_.sourceD2HPrefetch && chunk.localSlotHeld) {
            // Prefetched data in the local slot is still valid; only the remote
            // grant is stale. Request a fresh slot.
            chunk.state = nixlUcxStagedChunk::State::LOCAL_READY;
        } else {
            staged_handle->releaseLocalSlot(chunk);
            chunk.state = nixlUcxStagedChunk::State::PENDING;
        }
    };

    auto start_granted_chunk = [&](nixlUcxStagedChunk &chunk) -> nixl_status_t {
        if (!chunk.localMetadata || !chunk.remoteMetadata) {
            return NIXL_ERR_MISMATCH;
        }

        if (grant_is_stale(chunk)) {
            release_stale_grant(chunk);
            return NIXL_IN_PROG;
        }

        nixl_status_t ret = NIXL_SUCCESS;
        if (!vramStagingConfig_.sourceD2HPrefetch) {
            const auto local_slot_id = chunk.localMetadata->acquireSlot();
            if (!local_slot_id) {
                ++staged_handle->profile.localSlotMiss;
                return NIXL_IN_PROG;
            }

            chunk.localSlotId = *local_slot_id;
            chunk.localSlotHeld = true;

            int previous_device = -1;
            ret = cudaSetDeviceForCopy(chunk.localGpuDev, previous_device);
            if (ret != NIXL_SUCCESS) {
                staged_handle->releaseChunkSlots(chunk);
                return ret;
            }

            nixlUcxStagedSlot &local_slot = chunk.localMetadata->slots[chunk.localSlotId];
            const auto local_gpu_addr = reinterpret_cast<void *>(chunk.localGpuAddr);
            const uint64_t d2h_start_us = profileNowUs();
            const cudaError_t cuda_ret =
                cudaMemcpy(local_slot.hostAddr, local_gpu_addr, chunk.size, cudaMemcpyDeviceToHost);
            staged_handle->profile.d2hUs += profileNowUs() - d2h_start_us;
            cudaRestoreDevice(previous_device);
            if (cuda_ret != cudaSuccess) {
                NIXL_ERROR << "UCX staged D2H failed for transfer_id="
                           << staged_handle->transferId << " chunk_id=" << chunk.id << ": "
                           << cudaGetErrorString(cuda_ret);
                sendStagedSlotRelease(staged_handle->remoteAgent,
                                      staged_handle->transferId,
                                      chunk.id,
                                      chunk.remoteSlotId,
                                      chunk.leaseId,
                                      chunk.remoteGpuAddr,
                                      chunk.remoteGpuDev,
                                      chunk.size);
                staged_handle->releaseChunkSlots(chunk);
                return NIXL_ERR_BACKEND;
            }
        } else if (!chunk.localSlotHeld) {
            return NIXL_ERR_MISMATCH;
        }

        nixlUcxStagedSlot &local_slot = chunk.localMetadata->slots[chunk.localSlotId];
        const size_t worker_id = staged_handle->getWorkerId();
        const auto *rmd = chunk.remoteMetadata;
        if (chunk.remoteSlotId >= rmd->slotRkeys.size() ||
            worker_id >= rmd->slotRkeys[chunk.remoteSlotId].size()) {
            sendStagedSlotRelease(staged_handle->remoteAgent,
                                  staged_handle->transferId,
                                  chunk.id,
                                  chunk.remoteSlotId,
                                  chunk.leaseId,
                                  chunk.remoteGpuAddr,
                                  chunk.remoteGpuDev,
                                  chunk.size);
            staged_handle->releaseChunkSlots(chunk);
            return NIXL_ERR_MISMATCH;
        }

        // Final stale check at the last moment before the data-plane write: the
        // synchronous D2H above (or a long local-slot wait across polls) may have
        // consumed a large part of the lease budget.
        if (grant_is_stale(chunk)) {
            release_stale_grant(chunk);
            return NIXL_IN_PROG;
        }

        const auto &ep = rmd->conn->getEp(worker_id);
        nixlUcxReq req = nullptr;
        ret = ep->write(local_slot.hostAddr,
                        local_slot.mem,
                        rmd->slotAddrs[chunk.remoteSlotId],
                        rmd->slotRkeys[chunk.remoteSlotId][worker_id],
                        chunk.size,
                        req);
        if (vramStagingConfig_.batchFlush) {
            if (!staged_handle->openFlushBatch) {
                auto batch = std::make_unique<nixlUcxStagedFlushBatch>();
                batch->req =
                    std::make_unique<nixlUcxBackendReqH>(staged_handle->getWorker(),
                                                         staged_handle->getWorkerId());
                batch->req->reserve(staged_handle->chunks.size() + 1);
                batch->remoteMetadata = chunk.remoteMetadata;
                staged_handle->openFlushBatch = std::move(batch);
            }
            if (staged_handle->openFlushBatch->remoteMetadata != chunk.remoteMetadata) {
                staged_handle->releaseChunkSlots(chunk);
                return NIXL_ERR_NOT_SUPPORTED;
            }

            ret = staged_handle->openFlushBatch->req->append(ret, req, rmd->conn);
            if (ret != NIXL_SUCCESS) {
                sendStagedSlotRelease(staged_handle->remoteAgent,
                                      staged_handle->transferId,
                                      chunk.id,
                                      chunk.remoteSlotId,
                                      chunk.leaseId,
                                      chunk.remoteGpuAddr,
                                      chunk.remoteGpuDev,
                                      chunk.size);
                staged_handle->releaseChunkSlots(chunk);
                return ret;
            }
            staged_handle->openFlushBatch->chunks.push_back(&chunk);
        } else {
            ret = chunk.req->append(ret, req, rmd->conn);
            if (ret != NIXL_SUCCESS) {
                sendStagedSlotRelease(staged_handle->remoteAgent,
                                      staged_handle->transferId,
                                      chunk.id,
                                      chunk.remoteSlotId,
                                      chunk.leaseId,
                                      chunk.remoteGpuAddr,
                                      chunk.remoteGpuDev,
                                      chunk.size);
                staged_handle->releaseChunkSlots(chunk);
                return ret;
            }

            nixlUcxReq flush_req = nullptr;
            ret = ep->flushEp(flush_req);
            ret = chunk.req->append(ret, flush_req, rmd->conn);
            if (ret != NIXL_SUCCESS) {
                sendStagedSlotRelease(staged_handle->remoteAgent,
                                      staged_handle->transferId,
                                      chunk.id,
                                      chunk.remoteSlotId,
                                      chunk.leaseId,
                                      chunk.remoteGpuAddr,
                                      chunk.remoteGpuDev,
                                      chunk.size);
                staged_handle->releaseChunkSlots(chunk);
                return ret;
            }
            ++staged_handle->profile.flushPosted;
        }
        ++staged_handle->profile.rdmaWritePosted;

        chunk.rdmaPostedUs = profileNowUs();
        chunk.state = nixlUcxStagedChunk::State::RDMA_POSTED;
        return NIXL_SUCCESS;
    };

    auto send_ready = [&](nixlUcxStagedChunk &chunk) -> nixl_status_t {
        const auto conn = getConnection(staged_handle->remoteAgent);
        if (!conn) {
            return NIXL_ERR_NOT_FOUND;
        }

        nixlUcxReq req = nullptr;
        const nixl_status_t send_status =
            sendStagedWriteReady(staged_handle->remoteAgent,
                                 staged_handle->transferId,
                                 chunk.id,
                                 chunk.remoteSlotId,
                                 chunk.leaseId,
                                 chunk.remoteGpuAddr,
                                 chunk.remoteGpuDev,
                                 chunk.size,
                                 conn->getEp(staged_handle->getWorkerId()),
                                 &req);
        const nixl_status_t append_status = chunk.req->append(send_status, req, conn);
        if (append_status != NIXL_SUCCESS) {
            return append_status;
        }

        ++staged_handle->profile.readySent;
        chunk.readyPostedUs = profileNowUs();
        chunk.state = nixlUcxStagedChunk::State::READY_AM_POSTED;
        return NIXL_SUCCESS;
    };

    auto send_local_ready = [&](nixlUcxStagedChunk &chunk) -> nixl_status_t {
        if (!chunk.localMetadata || !chunk.localMetadata->localSharedSlots ||
            !chunk.localMetadata->sharedBase || chunk.localMetadata->sharedPath.empty()) {
            return NIXL_ERR_MISMATCH;
        }

        const auto conn = getConnection(staged_handle->remoteAgent);
        if (!conn) {
            return NIXL_ERR_NOT_FOUND;
        }

        if (!chunk.localSlotHeld || chunk.localSlotId >= chunk.localMetadata->slots.size()) {
            return NIXL_ERR_MISMATCH;
        }

        const auto &slot = chunk.localMetadata->slots[chunk.localSlotId];
        const auto *base = static_cast<const char *>(chunk.localMetadata->sharedBase);
        const auto *slot_addr = static_cast<const char *>(slot.hostAddr);
        if (slot_addr < base ||
            static_cast<size_t>(slot_addr - base) > chunk.localMetadata->sharedMappingSize) {
            return NIXL_ERR_MISMATCH;
        }

        const size_t slot_offset = static_cast<size_t>(slot_addr - base);
        if (chunk.size > chunk.localMetadata->sharedMappingSize - slot_offset) {
            return NIXL_ERR_MISMATCH;
        }

        const uint64_t slot_generation = chunk.localMetadata->slotGeneration(chunk.localSlotId);
        if (slot_generation == 0) {
            return NIXL_ERR_MISMATCH;
        }
        chunk.leaseId = slot_generation;

        nixlUcxReq req = nullptr;
        const nixl_status_t send_status =
            sendStagedLocalWriteReady(staged_handle->remoteAgent,
                                      staged_handle->transferId,
                                      chunk.id,
                                      chunk.localMetadata->localSharedRegionId,
                                      chunk.localMetadata->localSharedRegionCookie,
                                      chunk.localSlotId,
                                      slot_generation,
                                      chunk.localMetadata->sharedPath,
                                      slot_offset,
                                      chunk.localMetadata->sharedMappingSize,
                                      chunk.remoteGpuAddr,
                                      chunk.remoteGpuDev,
                                      chunk.size,
                                      conn->getEp(staged_handle->getWorkerId()),
                                      &req);
        const nixl_status_t append_status = chunk.req->append(send_status, req, conn);
        if (append_status != NIXL_SUCCESS) {
            return append_status;
        }

        ++staged_handle->profile.readySent;
        chunk.readyPostedUs = profileNowUs();
        chunk.state = nixlUcxStagedChunk::State::READY_AM_POSTED;
        return NIXL_SUCCESS;
    };

    auto close_open_flush_batch = [&](bool &made_progress) -> nixl_status_t {
        if (!staged_handle->openFlushBatch || staged_handle->openFlushBatch->chunks.empty()) {
            return NIXL_SUCCESS;
        }

        auto batch = std::move(staged_handle->openFlushBatch);
        nixlUcxStagedPublicMetadata *batch_rmd = batch->remoteMetadata;
        if (!batch_rmd) {
            return NIXL_ERR_MISMATCH;
        }

        const auto &ep = batch_rmd->conn->getEp(staged_handle->getWorkerId());
        nixlUcxReq flush_req = nullptr;
        nixl_status_t ret = ep->flushEp(flush_req);
        ret = batch->req->append(ret, flush_req, batch_rmd->conn);
        if (ret != NIXL_SUCCESS) {
            return ret;
        }

        ++staged_handle->profile.flushPosted;
        batch->postedUs = profileNowUs();
        for (auto *chunk : batch->chunks) {
            chunk->state = nixlUcxStagedChunk::State::FLUSH_POSTED;
        }
        staged_handle->flushBatches.push_back(std::move(batch));
        made_progress = true;
        return NIXL_SUCCESS;
    };

    auto process_flush_batches = [&](bool &made_progress) -> nixl_status_t {
        size_t batch_id = 0;
        while (batch_id < staged_handle->flushBatches.size()) {
            auto &batch = staged_handle->flushBatches[batch_id];
            const nixl_status_t status = batch->req->status();
            if (status == NIXL_IN_PROG) {
                ++batch_id;
                continue;
            }
            if (status != NIXL_SUCCESS) {
                for (auto *chunk : batch->chunks) {
                    chunk->state = nixlUcxStagedChunk::State::FAILED;
                }
                return status;
            }

            const uint64_t flush_done_us = profileNowUs();
            for (auto *chunk : batch->chunks) {
                if (chunk->rdmaPostedUs != 0) {
                    staged_handle->profile.rdmaFlushWaitUs +=
                        flush_done_us - chunk->rdmaPostedUs;
                }
                staged_handle->releaseLocalSlot(*chunk);
                const nixl_status_t ready_status = send_ready(*chunk);
                if (ready_status != NIXL_SUCCESS) {
                    chunk->state = nixlUcxStagedChunk::State::FAILED;
                    return ready_status;
                }
            }

            staged_handle->flushBatches.erase(staged_handle->flushBatches.begin() + batch_id);
            made_progress = true;
        }
        return NIXL_SUCCESS;
    };

    auto schedule_ready_chunks = [&]() -> nixl_status_t {
        for (const auto &chunk_ptr : staged_handle->chunks) {
            nixlUcxStagedChunk &chunk = *chunk_ptr;
            if (chunk.state == nixlUcxStagedChunk::State::PENDING) {
                const nixl_status_t status = vramStagingConfig_.sourceD2HPrefetch ?
                    start_local_d2h_prefetch(chunk) :
                    send_slot_request(chunk);
                if (status == NIXL_IN_PROG) {
                    continue;
                }
                if (status != NIXL_SUCCESS) {
                    return status;
                }
            }

            if (vramStagingConfig_.sourceD2HPrefetch &&
                chunk.state == nixlUcxStagedChunk::State::LOCAL_READY) {
                const nixl_status_t status = chunk.localSharedWrite ?
                    send_local_ready(chunk) :
                    send_slot_request(chunk);
                if (status == NIXL_IN_PROG) {
                    continue;
                }
                if (status != NIXL_SUCCESS) {
                    return status;
                }
            }
        }
        return NIXL_SUCCESS;
    };

    while (true) {
        switch (staged_handle->state) {
        case nixlUcxStagedBackendReqH::State::INIT:
            return NIXL_ERR_NOT_POSTED;

        case nixlUcxStagedBackendReqH::State::RUNNING: {
            staged_handle->getWorker()->progressLoop();
            bool made_progress = false;
            bool defer_schedule = false;

            if (vramStagingConfig_.batchFlush) {
                const nixl_status_t flush_progress_status = process_flush_batches(made_progress);
                if (flush_progress_status != NIXL_SUCCESS) {
                    return fail(flush_progress_status);
                }
            }

            for (const auto &chunk_ptr : staged_handle->chunks) {
                nixlUcxStagedChunk &chunk = *chunk_ptr;

                if (chunk.state == nixlUcxStagedChunk::State::LOCAL_D2H_POSTED) {
                    const nixl_status_t status = check_local_d2h_prefetch(chunk);
                    if (status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(status);
                    }
                    made_progress = true;
                }

                if (chunk.state == nixlUcxStagedChunk::State::SLOT_REQ_POSTED) {
                    const nixl_status_t status = chunk.req->status();
                    if (status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(status);
                    }

                    chunk.state = nixlUcxStagedChunk::State::WAIT_SLOT_GRANT;
                    made_progress = true;
                }

                if (chunk.state == nixlUcxStagedChunk::State::WAIT_SLOT_GRANT) {
                    if (!chunk.grantArrived.load()) {
                        continue;
                    }

                    const nixl_status_t grant_status = chunk.grantStatus.load();
                    if (grant_status == NIXL_IN_PROG) {
                        ++staged_handle->profile.slotGrantInProg;
                        staged_handle->releaseRemoteWindow(chunk);
                        chunk.state = vramStagingConfig_.sourceD2HPrefetch ?
                            nixlUcxStagedChunk::State::LOCAL_READY :
                            nixlUcxStagedChunk::State::PENDING;
                        chunk.grantArrived.store(false);
                        defer_schedule = true;
                        continue;
                    }
                    if (grant_status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(grant_status);
                    }

                    if (!chunk.remoteSlotHeld) {
                        // Count each grant once: after a local-slot miss the chunk
                        // stays in WAIT_SLOT_GRANT with grantArrived still set, and
                        // re-entries would recount the same grant.
                        ++staged_handle->profile.slotGrantSuccess;
                        const uint64_t grant_arrived_us = chunk.grantArrivedUs.load();
                        if (grant_arrived_us > chunk.slotReqPostedUs) {
                            staged_handle->profile.grantWaitUs +=
                                grant_arrived_us - chunk.slotReqPostedUs;
                        }
                        chunk.remoteSlotHeld = true;
                    }
                    const nixl_status_t start_status = start_granted_chunk(chunk);
                    if (start_status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (start_status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(start_status);
                    }
                    made_progress = true;
                }

                if (chunk.state == nixlUcxStagedChunk::State::RDMA_POSTED) {
                    if (vramStagingConfig_.batchFlush) {
                        continue;
                    }

                    const nixl_status_t status = chunk.req->status();
                    if (status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(status);
                    }

                    staged_handle->releaseLocalSlot(chunk);
                    if (chunk.rdmaPostedUs != 0) {
                        staged_handle->profile.rdmaFlushWaitUs +=
                            profileNowUs() - chunk.rdmaPostedUs;
                    }

                    const nixl_status_t ready_status = send_ready(chunk);
                    if (ready_status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(ready_status);
                    }
                    made_progress = true;
                }

                if (chunk.state == nixlUcxStagedChunk::State::READY_AM_POSTED) {
                    const nixl_status_t status = chunk.req->status();
                    if (status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (status != NIXL_SUCCESS) {
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(status);
                    }

                    chunk.state = nixlUcxStagedChunk::State::WAIT_ACK;
                    if (chunk.readyPostedUs != 0) {
                        staged_handle->profile.readyWaitUs +=
                            profileNowUs() - chunk.readyPostedUs;
                    }
                    chunk.ackWaitStartUs = profileNowUs();
                    made_progress = true;
                }

                if (chunk.state == nixlUcxStagedChunk::State::WAIT_ACK) {
                    const nixl_status_t status = chunk.ackStatus.load();
                    if (status == NIXL_IN_PROG) {
                        continue;
                    }
                    if (status != NIXL_SUCCESS) {
                        if (chunk.localSharedWrite) {
                            ++staged_handle->profile.localSharedAckErrors;
                        }
                        if (chunk.localSharedWrite && vramStagingConfig_.localStagingFallback &&
                            !chunk.localFallbackAttempted && chunk.localSlotHeld) {
                            ++staged_handle->profile.localSharedFallbacks;
                            chunk.localFallbackAttempted = true;
                            chunk.localSharedWrite = false;
                            chunk.leaseId = 0;
                            chunk.ackStatus.store(NIXL_IN_PROG);
                            chunk.ackArrivedUs.store(0);
                            chunk.readyPostedUs = 0;
                            chunk.ackWaitStartUs = 0;
                            chunk.state = nixlUcxStagedChunk::State::LOCAL_READY;
                            made_progress = true;
                            NIXL_WARN << "UCX local staged WRITE failed for transfer_id="
                                      << staged_handle->transferId << " chunk_id=" << chunk.id
                                      << " status=" << status
                                      << "; falling back to UCX staged host transfer";
                            continue;
                        }
                        chunk.state = nixlUcxStagedChunk::State::FAILED;
                        return fail(status);
                    }

                    if (chunk.localSharedWrite) {
                        staged_handle->releaseLocalSlot(chunk);
                    } else {
                        staged_handle->releaseRemoteSlot(chunk);
                    }
                    ++staged_handle->profile.ackReceived;
                    const uint64_t ack_arrived_us = chunk.ackArrivedUs.load();
                    if (ack_arrived_us > chunk.ackWaitStartUs) {
                        staged_handle->profile.ackWaitUs +=
                            ack_arrived_us - chunk.ackWaitStartUs;
                    }
                    chunk.state = nixlUcxStagedChunk::State::ACKED;
                    ++staged_handle->completedChunks;
                    made_progress = true;
                }
            }

            if (vramStagingConfig_.batchFlush) {
                const nixl_status_t flush_post_status = close_open_flush_batch(made_progress);
                if (flush_post_status != NIXL_SUCCESS) {
                    return fail(flush_post_status);
                }
            }

            if (!defer_schedule) {
                const nixl_status_t schedule_status = schedule_ready_chunks();
                if (schedule_status != NIXL_SUCCESS && schedule_status != NIXL_IN_PROG) {
                    return fail(schedule_status);
                }
            }

            if (staged_handle->completedChunks != staged_handle->chunks.size()) {
                if (made_progress) {
                    continue;
                }
                return NIXL_IN_PROG;
            }

            if (staged_handle->pendingRegistered) {
                unregisterPendingStagedReq(staged_handle->transferId, staged_handle);
                staged_handle->pendingRegistered = false;
            }

            log_profile();

            if (!staged_handle->notif) {
                staged_handle->lastStatus = NIXL_SUCCESS;
                staged_handle->state = nixlUcxStagedBackendReqH::State::COMPLETE;
                return NIXL_SUCCESS;
            }

            const nixlUcxBackendReqH::Notif notif(std::move(staged_handle->notif).value());
            staged_handle->notif.reset();

            const ucx_connection_ptr_t conn = getConnection(notif.agent);
            if (!conn) {
                return fail(NIXL_ERR_NOT_FOUND);
            }

            nixlUcxReq req = nullptr;
            const auto &ep = conn->getEp(staged_handle->getWorkerId());
            const nixl_status_t send_status = notifSendPriv(notif.agent, notif.payload, ep, &req);
            const nixl_status_t append_status = staged_handle->append(send_status, req, conn);
            if (append_status != NIXL_SUCCESS) {
                return fail(append_status);
            }

            staged_handle->state = nixlUcxStagedBackendReqH::State::USER_NOTIF_POSTED;
            continue;
        }

        case nixlUcxStagedBackendReqH::State::USER_NOTIF_POSTED: {
            const nixl_status_t status = staged_handle->nixlUcxBackendReqH::status();
            if (status == NIXL_IN_PROG) {
                return status;
            }
            if (status != NIXL_SUCCESS) {
                return fail(status);
            }
            staged_handle->lastStatus = NIXL_SUCCESS;
            staged_handle->state = nixlUcxStagedBackendReqH::State::COMPLETE;
            return NIXL_SUCCESS;
        }

        case nixlUcxStagedBackendReqH::State::COMPLETE:
            return NIXL_SUCCESS;

        case nixlUcxStagedBackendReqH::State::FAILED:
            return staged_handle->lastStatus;
        }
    }
}

nixl_status_t
nixlUcxEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    const size_t lcnt = local.descCount();
    const size_t rcnt = remote.descCount();
    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    nixl_status_t ret;

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local (" << lcnt << ") and remote (" << rcnt
                   << ") descriptor lists differ in size";
        return NIXL_ERR_INVALID_PARAM;
    }

    if (dynamic_cast<nixlUcxStagedBackendReqH *>(handle)) {
        if (operation != NIXL_WRITE) {
            return NIXL_ERR_NOT_SUPPORTED;
        }
        return postStagedWrite(local, remote, remote_agent, handle, opt_args);
    }

    // TODO: assert that handle is empty/completed, as we can't post request before completion

    ret = sendXferRange(operation, local, remote, remote_agent, handle, 0, lcnt);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    ret = int_handle->status();
    if (opt_args && opt_args->hasNotif) {
        if (ret == NIXL_SUCCESS) {
            nixlUcxReq req;
            const auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[0].metadataP);
            ret = notifSendPriv(remote_agent,
                                opt_args->notifMsg,
                                rmd->conn->getEp(int_handle->getWorkerId()),
                                &req);
            if (int_handle->append(ret, req, rmd->conn) != NIXL_SUCCESS) {
                return ret;
            }

            ret = int_handle->status();
        } else if (ret == NIXL_IN_PROG) {
            int_handle->notif.emplace(remote_agent, opt_args->notifMsg);
        }
    }

    return ret;
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle) const
{
    if (dynamic_cast<nixlUcxStagedBackendReqH *>(handle)) {
        return checkStagedXfer(handle);
    }

    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    const nixl_status_t handle_status = int_handle->status();

    if ((handle_status == NIXL_IN_PROG) || !int_handle->notif) {
        return handle_status;
    }

    const nixlUcxBackendReqH::Notif notif(std::move(int_handle->notif).value());
    int_handle->notif.reset();

    if (__builtin_expect(handle_status != NIXL_SUCCESS, 0)) {
        return handle_status;
    }

    const ucx_connection_ptr_t conn = getConnection(notif.agent);
    if (__builtin_expect(!conn, 0)) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxReq req;
    const auto &ep = conn->getEp(int_handle->getWorkerId());
    const nixl_status_t status = notifSendPriv(notif.agent, notif.payload, ep, &req);

    if (int_handle->append(status, req, conn) != NIXL_SUCCESS) {
        return status;
    }

    return int_handle->status();
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle) const
{
    if (auto *staged_handle = dynamic_cast<nixlUcxStagedBackendReqH *>(handle)) {
        // Same ordering as the staged fail path: unregister before touching chunks so
        // late grants/ACKs go through the unknown-transfer path instead of racing
        // with the teardown below.
        if (staged_handle->pendingRegistered) {
            unregisterPendingStagedReq(staged_handle->transferId, staged_handle);
            staged_handle->pendingRegistered = false;
        }
        for (const auto &chunk_ptr : staged_handle->chunks) {
            nixlUcxStagedChunk &chunk = *chunk_ptr;
            const bool granted_unprocessed = !chunk.remoteSlotHeld &&
                chunk.grantArrived.load() && chunk.grantStatus.load() == NIXL_SUCCESS;
            if (chunk.remoteSlotHeld || granted_unprocessed) {
                sendStagedSlotRelease(staged_handle->remoteAgent,
                                      staged_handle->transferId,
                                      chunk.id,
                                      chunk.remoteSlotId,
                                      chunk.leaseId,
                                      chunk.remoteGpuAddr,
                                      chunk.remoteGpuDev,
                                      chunk.size);
                staged_handle->releaseRemoteSlot(chunk);
            }
        }
        staged_handle->release();
        delete staged_handle;
        return NIXL_SUCCESS;
    }

    const auto int_handle = static_cast<nixlUcxBackendReqH *>(handle);
    int_handle->release();

    /* TODO: return to a pool instead. */
    delete int_handle;

    return NIXL_SUCCESS;
}

unsigned
nixlUcxEngine::progress() {
    // TODO: add listen for connection handling if necessary
    unsigned ret = 0;
    for (auto &uw : uws) {
        ret += uw->progress();
    }
    return ret;
}

void
nixlUcxEngine::progressLoop() {
    while (progress() != 0)
        ;
}

/****************************************
 * Notifications
*****************************************/

//agent will provide cached msg
nixl_status_t
nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                             const std::string &msg,
                             const std::unique_ptr<nixlUcxEp> &ep,
                             nixlUcxReq *req) const {
    nixlSerDes ser_des;

    ser_des.addStr("name", localAgent);
    ser_des.addStr("msg", msg);
    // TODO: replace with mpool for performance

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer, req](void *completed_request, void *ptr) {
        delete buffer;
        if ((req == nullptr) && (completed_request != nullptr)) {
            /* Caller is not interested in the request, free it */
            ucp_request_free(completed_request);
        }
    };

    return ep->sendAm(nixl::ucx::am_cb_op_t::NOTIF_STR,
                      nullptr,
                      0,
                      (void *)buffer->data(),
                      buffer->size(),
                      UCP_AM_SEND_FLAG_EAGER,
                      req,
                      deleter);
}

ucx_connection_ptr_t
nixlUcxEngine::getConnection(const std::string &remote_agent) const {
    const std::shared_lock lock(remoteConnMapMutex_);
    const auto it = remoteConnMap.find(remote_agent);
    return (it != remoteConnMap.end()) ? it->second : nullptr;
}

void
nixlUcxEngine::appendNotif(std::string &&remote_name, std::string &&msg) {
    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    notifList_.emplace_back(std::move(remote_name), std::move(msg));
}

ucs_status_t
nixlUcxEngine::notifAmCb(void *arg, const void *header,
                         size_t header_length, void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param)
{
    nixlSerDes ser_des;

    std::string ser_str( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;

    // send_am should be forcing EAGER protocol
    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    ser_des.importStr(ser_str);
    std::string remote_name = ser_des.getStr("name");
    std::string msg = ser_des.getStr("msg");

    engine->appendNotif(std::move(remote_name), std::move(msg));
    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::stagedSlotReqAmCb(void *arg,
                                 const void *header,
                                 size_t header_length,
                                 void *data,
                                 size_t length,
                                 const ucp_am_recv_param_t *param) {
    auto *engine = (nixlUcxEngine *)arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    const std::string ser_str((char *)data, length);
    const ucp_ep_h reply_ep =
        (param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP) ? param->reply_ep : nullptr;
    engine->handleStagedSlotReq(ser_str, reply_ep);
    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::stagedSlotGrantAmCb(void *arg,
                                   const void *header,
                                   size_t header_length,
                                   void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param) {
    nixlSerDes ser_des;
    auto *engine = (nixlUcxEngine *)arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    const std::string ser_str((char *)data, length);
    if (ser_des.importStr(ser_str) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize UCX staged SLOT_GRANT message";
        return UCS_OK;
    }

    const std::string remote_agent = ser_des.getStr("name");
    uint64_t transfer_id = 0;
    uint64_t chunk_id = 0;
    uint64_t slot_id = 0;
    uint64_t lease_id = 0;
    nixl_status_t status = NIXL_ERR_MISMATCH;
    if (remote_agent.empty() ||
        ser_des.getBuf("xfer_id", &transfer_id, sizeof(transfer_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("chunk_id", &chunk_id, sizeof(chunk_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("slot_id", &slot_id, sizeof(slot_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("lease_id", &lease_id, sizeof(lease_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("status", &status, sizeof(status)) != NIXL_SUCCESS) {
        NIXL_ERROR << "Malformed UCX staged SLOT_GRANT message";
        return UCS_OK;
    }

    engine->completePendingStagedSlotGrant(
        remote_agent, transfer_id, chunk_id, slot_id, lease_id, status);
    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::stagedSlotReleaseAmCb(void *arg,
                                     const void *header,
                                     size_t header_length,
                                     void *data,
                                     size_t length,
                                     const ucp_am_recv_param_t *param) {
    auto *engine = (nixlUcxEngine *)arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    const std::string ser_str((char *)data, length);
    engine->handleStagedSlotRelease(ser_str);
    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::stagedWriteReadyAmCb(void *arg,
                                    const void *header,
                                    size_t header_length,
                                    void *data,
                                    size_t length,
                                    const ucp_am_recv_param_t *param) {
    auto *engine = (nixlUcxEngine *)arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    const std::string ser_str((char *)data, length);
    const ucp_ep_h reply_ep =
        (param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP) ? param->reply_ep : nullptr;
    engine->handleStagedWriteReady(ser_str, reply_ep);
    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::stagedLocalWriteReadyAmCb(void *arg,
                                         const void *header,
                                         size_t header_length,
                                         void *data,
                                         size_t length,
                                         const ucp_am_recv_param_t *param) {
    auto *engine = (nixlUcxEngine *)arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    const std::string ser_str((char *)data, length);
    const ucp_ep_h reply_ep =
        (param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP) ? param->reply_ep : nullptr;
    engine->handleStagedLocalWriteReady(ser_str, reply_ep);
    return UCS_OK;
}

ucs_status_t
nixlUcxEngine::stagedAckAmCb(void *arg,
                             const void *header,
                             size_t header_length,
                             void *data,
                             size_t length,
                             const ucp_am_recv_param_t *param) {
    nixlSerDes ser_des;
    auto *engine = (nixlUcxEngine *)arg;

    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    const std::string ser_str((char *)data, length);
    if (ser_des.importStr(ser_str) != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deserialize UCX staged ACK message";
        return UCS_OK;
    }

    const std::string remote_agent = ser_des.getStr("name");
    uint64_t transfer_id = 0;
    uint64_t chunk_id = 0;
    uint64_t lease_id = 0;
    nixl_status_t status = NIXL_ERR_MISMATCH;
    if (remote_agent.empty() ||
        ser_des.getBuf("xfer_id", &transfer_id, sizeof(transfer_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("chunk_id", &chunk_id, sizeof(chunk_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("lease_id", &lease_id, sizeof(lease_id)) != NIXL_SUCCESS ||
        ser_des.getBuf("status", &status, sizeof(status)) != NIXL_SUCCESS) {
        NIXL_ERROR << "Malformed UCX staged ACK message";
        return UCS_OK;
    }

    engine->completePendingStagedReq(transfer_id, chunk_id, lease_id, status);
    return UCS_OK;
}

nixl_status_t
nixlUcxEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    progressLoop();

    // In the "no progress thread" case the lock in nixlAgent is sufficient.
    notifList_.swap(notif_list);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    const auto conn = getConnection(remote_agent);
    if (!conn) {
        return NIXL_ERR_NOT_FOUND;
    }

    const nixl_status_t ret = notifSendPriv(remote_agent, msg, conn->getEp(getWorkerId()));
    if (ret == NIXL_IN_PROG) {
        return NIXL_SUCCESS;
    }
    return ret;
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, worker_id, *getWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare remote memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, *getWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare local memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

void
nixlUcxEngine::releaseMemView(nixlMemViewH mem_view) const {
    nixl::ucx::releaseMemList(mem_view);
}
