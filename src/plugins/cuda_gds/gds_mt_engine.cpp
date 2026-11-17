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
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include <sys/types.h>

#include "common/backend.h"
#include "common/nixl_log.h"
#include "gds_mt_engine.h"

size_t
defaultGdsMtThreadCount() noexcept {
    static const size_t thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);
    return thread_count;
}

nixlGdsMtVectorReq::nixlGdsMtVectorReq(const gdsXferReq &first)
    : fh(first.fh),
      op(first.op),
      file_offset(first.file_offset) {
    append(first);
}

bool
nixlGdsMtVectorReq::canAppend(const gdsXferReq &req, size_t max_entries) const {
    if (entries.size() >= max_entries || req.fh != fh || req.op != op) {
        return false;
    }
    if (req.size > std::numeric_limits<size_t>::max() - total_size) {
        return false;
    }
    if (total_size + req.size > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
        return false;
    }
    if (total_size > std::numeric_limits<size_t>::max() - file_offset) {
        return false;
    }
    return req.file_offset == file_offset + total_size;
}

void
nixlGdsMtVectorReq::append(const gdsXferReq &req) {
    entries.push_back(req);
    total_size += req.size;
}

namespace {
constexpr size_t optimized_pooled_piov_limit = 32;
constexpr size_t default_vector_max_entries = optimized_pooled_piov_limit;

[[nodiscard]] bool
isCuFileVectorIoCompiledIn() {
#if defined(NIXL_HAVE_CUFILE_VECTOR_IO)
    return true;
#else
    return false;
#endif
}

[[nodiscard]] size_t
getThreadCount(const nixlBackendInitParams *init_params) {
    nixl_b_params_t *params = init_params->customParams;
    const size_t default_thread_count = defaultGdsMtThreadCount();
    const size_t count =
        nixl::getBackendParamDefaulted(params, "thread_count", default_thread_count);
    return (count > 0) ? count : default_thread_count;
}

// cuFileRead/cuFileWrite report filesystem errors as -1 with errno set, and
// cuFile-level errors either as a negative CUfileOpError return value (as
// documented) or, as observed in practice, as -1 with the CUfileOpError code
// stored in errno. strerror is only meaningful for the first case.
void
logCuFileIOError(const char *api, ssize_t nbytes) {
    if (nbytes == -1 && !IS_CUFILE_ERR(errno)) {
        NIXL_ERROR << "GDS_MT: " << api << " failed: " << nixl_strerror(errno);
    } else {
        const int err = (nbytes == -1) ? errno : (int)-nbytes;
        NIXL_ERROR << "GDS_MT: " << api << " failed: " << CUFILE_ERRSTR(err) << " (err=" << err
                   << ")";
    }
}

[[nodiscard]] std::string
toLower(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return str;
}

[[nodiscard]] nixlGdsMtVectorIoMode
getVectorIoMode(const nixlBackendInitParams *init_params) {
    const std::string value = toLower(nixl::getBackendParamDefaulted(
        init_params->customParams, "vector_io", std::string("auto")));
    if (value == "auto") {
        return nixlGdsMtVectorIoMode::AUTO;
    }
    if (value == "off") {
        return nixlGdsMtVectorIoMode::OFF;
    }
    if (value == "on") {
        return nixlGdsMtVectorIoMode::ON;
    }
    throw std::runtime_error("GDS_MT: vector_io must be one of: auto, off, on");
}

[[nodiscard]] const char *
vectorIoModeToString(nixlGdsMtVectorIoMode mode) {
    switch (mode) {
    case nixlGdsMtVectorIoMode::AUTO:
        return "auto";
    case nixlGdsMtVectorIoMode::OFF:
        return "off";
    case nixlGdsMtVectorIoMode::ON:
        return "on";
    }
    return "unknown";
}

[[nodiscard]] size_t
getVectorMaxEntries(const nixlBackendInitParams *init_params) {
    const size_t requested = nixl::getBackendParamDefaulted(
        init_params->customParams, "vector_max_entries", default_vector_max_entries);
    if (requested == 0) {
        return 1;
    }
    return std::min(requested, static_cast<size_t>(std::numeric_limits<int>::max()));
}

[[nodiscard]] bool
vectorIoCanBeUsed(nixlGdsMtVectorIoMode mode) {
    return mode != nixlGdsMtVectorIoMode::OFF && isCuFileVectorIoCompiledIn();
}

[[nodiscard]] bool
canVectorizeRequest(const gdsXferReq &req) {
    return req.addr != nullptr && req.fh != nullptr && req.size > 0 &&
        req.size <= static_cast<size_t>(std::numeric_limits<ssize_t>::max());
}

void
flushVectorGroup(std::optional<nixlGdsMtVectorReq> &group,
                 std::vector<nixlGdsMtWorkItem> &work_list) {
    if (!group) {
        return;
    }
    if (group->entries.size() == 1) {
        work_list.emplace_back(group->entries.front());
    } else {
        work_list.emplace_back(std::move(*group));
    }
    group.reset();
}

void
appendWorkItem(const gdsXferReq &req,
               bool use_vector_io,
               size_t vector_max_entries,
               std::optional<nixlGdsMtVectorReq> &group,
               std::vector<nixlGdsMtWorkItem> &work_list) {
    if (!use_vector_io || !canVectorizeRequest(req)) {
        flushVectorGroup(group, work_list);
        work_list.emplace_back(req);
        return;
    }

    if (group && group->canAppend(req, vector_max_entries)) {
        group->append(req);
        return;
    }

    flushVectorGroup(group, work_list);
    group.emplace(req);
}

} // namespace

std::vector<nixlGdsMtWorkItem>
buildGdsMtWorkList(const std::vector<gdsXferReq> &reqs,
                   bool use_vector_io,
                   size_t vector_max_entries) {
    std::vector<nixlGdsMtWorkItem> work_list;
    work_list.reserve(reqs.size());

    std::optional<nixlGdsMtVectorReq> vector_group;
    for (const gdsXferReq &req : reqs) {
        appendWorkItem(req, use_vector_io, vector_max_entries, vector_group, work_list);
    }
    flushVectorGroup(vector_group, work_list);
    return work_list;
}

namespace {

[[nodiscard]] const char *
opName(CUfileOpcode_t op) {
    return (op == CUFILE_READ) ? "read" : "write";
}

void
runCuFileScalarOp(const gdsXferReq *req, std::atomic<nixl_status_t> *overall_status) {
    ssize_t nbytes = 0;
    if (req->op == CUFILE_READ) {
        nbytes = cuFileRead(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            logCuFileIOError("cuFileRead", nbytes);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    } else if (req->op == CUFILE_WRITE) {
        nbytes = cuFileWrite(req->fh, req->addr, req->size, req->file_offset, 0);
        if (nbytes < 0) {
            logCuFileIOError("cuFileWrite", nbytes);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    } else {
        overall_status->store(NIXL_ERR_INVALID_PARAM);
        return;
    }

    if (static_cast<size_t>(nbytes) != req->size) {
        NIXL_ERROR << "GDS_MT: error: short " << opName(req->op) << ": " << nbytes << " out of "
                   << req->size << " bytes - address=" << req->addr;
        overall_status->store(NIXL_ERR_BACKEND);
    }
}

void
runCuFileVectorOp(const nixlGdsMtVectorReq *req, std::atomic<nixl_status_t> *overall_status) {
#if defined(NIXL_HAVE_CUFILE_VECTOR_IO)
    std::vector<CUfileIOVec_t> iovecs;
    iovecs.reserve(req->entries.size());
    for (const gdsXferReq &entry : req->entries) {
        CUfileIOVec_t iov = {};
        iov.base = entry.addr;
        iov.len = entry.size;
        iovecs.push_back(iov);
    }

    ssize_t nbytes = 0;
    if (req->op == CUFILE_READ) {
        nbytes = cuFileReadv(
            req->fh, iovecs.data(), iovecs.size(), static_cast<off_t>(req->file_offset), 0);
        if (nbytes < 0) {
            logCuFileIOError("cuFileReadv", nbytes);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    } else if (req->op == CUFILE_WRITE) {
        nbytes = cuFileWritev(
            req->fh, iovecs.data(), iovecs.size(), static_cast<off_t>(req->file_offset), 0);
        if (nbytes < 0) {
            logCuFileIOError("cuFileWritev", nbytes);
            overall_status->store(NIXL_ERR_BACKEND);
            return;
        }
    } else {
        overall_status->store(NIXL_ERR_INVALID_PARAM);
        return;
    }
#else
    NIXL_ERROR << "GDS_MT: vector I/O work item reached a build without cuFile vector support";
    overall_status->store(NIXL_ERR_BACKEND);
    return;
#endif

#if defined(NIXL_HAVE_CUFILE_VECTOR_IO)
    if (static_cast<size_t>(nbytes) != req->total_size) {
        NIXL_ERROR << "GDS_MT: error: short vector " << opName(req->op) << ": " << nbytes
                   << " out of " << req->total_size << " bytes across " << req->entries.size()
                   << " entries";
        overall_status->store(NIXL_ERR_BACKEND);
    }
#endif
}

void
runWorkItem(nixlGdsMtWorkItem *work, std::atomic<nixl_status_t> *overall_status) {
    std::visit(
        [overall_status](const auto &req) {
            using request_t = std::decay_t<decltype(req)>;
            if constexpr (std::is_same_v<request_t, nixlGdsMtVectorReq>) {
                runCuFileVectorOp(&req, overall_status);
            } else {
                runCuFileScalarOp(&req, overall_status);
            }
        },
        *work);
}
} // namespace

nixlGdsMtReqH::~nixlGdsMtReqH() {
    if (running_transfer.valid()) {
        // TODO: A follow-up should make active release nonblocking. This wait
        // preserves the pre-consolidation GDS_MT request-lifetime behavior.
        running_transfer.wait();
    }
}

nixlGdsMtEngine::nixlGdsMtEngine(const nixlBackendInitParams *init_params)
    : nixlGdsEngine(init_params),
      thread_count_(getThreadCount(init_params)),
      vector_io_mode_(getVectorIoMode(init_params)),
      vector_max_entries_(getVectorMaxEntries(init_params)) {
    // Base ctor opened the cuFile driver; bail if that failed.
    if (this->initErr) {
        return;
    }
    if (vector_io_mode_ == nixlGdsMtVectorIoMode::ON && !isCuFileVectorIoCompiledIn()) {
        NIXL_ERROR << "GDS_MT: vector_io=on requires cuFile readv/writev support at build time";
        this->initErr = true;
        return;
    }
    if (vectorIoCanBeUsed(vector_io_mode_) && vector_max_entries_ > optimized_pooled_piov_limit) {
        NIXL_WARN << "GDS_MT: vector_max_entries=" << vector_max_entries_
                  << " exceeds the optimized pooled PIOV limit. Consider vector_max_entries <="
                  << optimized_pooled_piov_limit << " for better performance.";
    }

    try {
        executor_ = std::make_unique<tf::Executor>(thread_count_);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "GDS_MT: failed to create executor: " << e.what();
        this->initErr = true;
        return;
    }
    NIXL_DEBUG << "GDS_MT: thread count=" << thread_count_
               << " vector_io=" << vectorIoModeToString(vector_io_mode_)
               << " vector_compiled=" << isCuFileVectorIoCompiledIn()
               << " vector_max_entries=" << vector_max_entries_;
}

nixl_status_t
nixlGdsMtEngine::finalizePrep(std::vector<gdsXferReq> &&reqs, nixlBackendReqH *&handle) const {
    if (reqs.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto gds_handle = std::make_unique<nixlGdsMtReqH>();
    const bool use_vector_io = vectorIoCanBeUsed(vector_io_mode_);
    gds_handle->work_list = buildGdsMtWorkList(reqs, use_vector_io, vector_max_entries_);

    size_t scalar_work_items = 0;
    size_t vector_work_items = 0;
    size_t vector_entries = 0;
    for (nixlGdsMtWorkItem &work : gds_handle->work_list) {
        if (const auto *vector_req = std::get_if<nixlGdsMtVectorReq>(&work)) {
            ++vector_work_items;
            vector_entries += vector_req->entries.size();
        } else {
            ++scalar_work_items;
        }

        nixlGdsMtWorkItem *captured_work = &work;
        gds_handle->taskflow.emplace(
            [captured_work, overall_status = &gds_handle->overall_status]() {
                runWorkItem(captured_work, overall_status);
            });
    }

    NIXL_DEBUG << "GDS_MT: prepared " << scalar_work_items << " scalar work items and "
               << vector_work_items << " vector work items (" << vector_entries
               << " vector entries), vector_io=" << vectorIoModeToString(vector_io_mode_)
               << " vector_compiled=" << isCuFileVectorIoCompiledIn();

    handle = gds_handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsMtEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    auto *gds_handle = static_cast<nixlGdsMtReqH *>(handle);

    gds_handle->overall_status.store(NIXL_SUCCESS);
    gds_handle->running_transfer = executor_->run(gds_handle->taskflow);
    return NIXL_IN_PROG;
}

nixl_status_t
nixlGdsMtEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *gds_handle = static_cast<nixlGdsMtReqH *>(handle);
    if (gds_handle->running_transfer.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready) {
        return NIXL_IN_PROG;
    }
    gds_handle->running_transfer.get();
    return gds_handle->overall_status.load();
}

nixl_status_t
nixlGdsMtEngine::releaseReqH(nixlBackendReqH *handle) const {
    delete static_cast<nixlGdsMtReqH *>(handle);
    return NIXL_SUCCESS;
}
