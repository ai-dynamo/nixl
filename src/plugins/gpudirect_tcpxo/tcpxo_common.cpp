/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#include "tcpxo_common.h"

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "common/nixl_log.h"

namespace tcpxo {

nixl_status_t
AbslStatusToNixlStatus(const absl::Status &status, const char *file, int line) {
    if (status.ok()) {
        return NIXL_SUCCESS;
    }

    NIXL_ERROR << file << ":" << line << ": " << status;

    switch (status.code()) {
    case absl::StatusCode::kInvalidArgument:
        return NIXL_ERR_INVALID_PARAM;
    case absl::StatusCode::kNotFound:
        return NIXL_ERR_NOT_FOUND;
    case absl::StatusCode::kUnimplemented:
        return NIXL_ERR_NOT_SUPPORTED;
    case absl::StatusCode::kCancelled:
        return NIXL_ERR_CANCELED;
    case absl::StatusCode::kPermissionDenied:
        return NIXL_ERR_NOT_ALLOWED;
    case absl::StatusCode::kAlreadyExists:
        return NIXL_IN_PROG;
    default:
        return NIXL_ERR_BACKEND;
    }
}

} // namespace tcpxo
