/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <strings.h>

#include "ais_utils.h"
#include "common/nixl_log.h"

namespace {
// hipFile buffer registration fails on setups without the AIS fast path. Opting
// in lets the transfer fall back to the compatibility path instead of aborting
// registration outright.
bool
aisCompatModeAllowed() {
    const char *v = std::getenv("HIPFILE_ALLOW_COMPAT_MODE");
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0 || strcasecmp(v, "yes") == 0;
}
} // namespace

aisDriverHandle::aisDriverHandle() {
    const hipFileError_t status = hipFileDriverOpen();
    if (status.err != hipFileSuccess) {
        throw std::runtime_error("AIS: error initializing AMD Infinity Storage driver: error=" +
                                 std::to_string(status.err));
    }
}

aisDriverHandle::~aisDriverHandle() {
    (void)hipFileDriverClose();
}

aisMemBuf::aisMemBuf(void *ptr, size_t sz, int flags) : base_(ptr) {
    const hipFileError_t status = hipFileBufRegister(ptr, sz, flags);
    if (status.err != hipFileSuccess) {
        if (aisCompatModeAllowed()) {
            NIXL_WARN << "AIS: buffer registration failed - compat mode: err=" << status.err;
            return;
        }
        throw std::runtime_error("AIS: hipFileBufRegister failed (err=" +
                                 std::to_string(status.err) +
                                 "); set HIPFILE_ALLOW_COMPAT_MODE=true to allow fallback");
    }
    registered_ = true;
}

aisMemBuf::~aisMemBuf() {
    if (registered_) {
        const hipFileError_t status = hipFileBufDeregister(base_);
        if (status.err != hipFileSuccess) {
            NIXL_WARN << "AIS: warning: deregistering buffer: error=" << status.err
                      << " ptr=" << base_;
        }
    }
}

aisFileHandle::aisFileHandle(nixl::FileFd &&fd) : file_fd(std::move(fd)) {
    hipFileDescr_t descr = {};
    descr.handle.fd = file_fd.fd();
    descr.type = hipFileHandleTypeOpaqueFD;

    const hipFileError_t status = hipFileHandleRegister(&hip_fhandle, &descr);
    if (status.err != hipFileSuccess) {
        // ~FileFd as the exception unwinds closes the owned fd if any.
        throw std::runtime_error("AIS: file register error: error=" + std::to_string(status.err) +
                                 ", fd=" + std::to_string(file_fd.fd()));
    }
}

aisFileHandle::~aisFileHandle() {
    (void)hipFileHandleDeregister(hip_fhandle);
    // ~FileFd closes the fd if path-mode owned it.
}
