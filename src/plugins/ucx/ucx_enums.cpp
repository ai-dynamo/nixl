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

#include "ucx_enums.h"

#include "common/nixl_log.h"

namespace nixl::ucx {

std::ostream &
operator<<(std::ostream &os, const mt_mode_t t) {
    toStream(os, t);
    return os;
}

std::ostream &
operator<<(std::ostream &os, const ep_state_t t) {
    toStream(os, t);
    return os;
}

std::ostream &
operator<<(std::ostream &os, const am_cb_op_t t) {
    toStream(os, t);
    return os;
}

nixl_status_t
ucsToNixlStatus(const ucs_status_t t) {
    switch (__builtin_expect(t, UCS_OK)) {
    case UCS_OK:
        return NIXL_SUCCESS;
    case UCS_INPROGRESS:
    case UCS_ERR_BUSY:
        return NIXL_IN_PROG;
    case UCS_ERR_NOT_CONNECTED:
    case UCS_ERR_CONNECTION_RESET:
    case UCS_ERR_ENDPOINT_TIMEOUT:
    case UCS_ERR_CANCELED: // UCX >= 1.21: peer-failure cancels pending requests
        return NIXL_ERR_REMOTE_DISCONNECT;
    case UCS_ERR_INVALID_PARAM:
        return NIXL_ERR_INVALID_PARAM;
    default:
        NIXL_WARN << "Unexpected UCX error: " << ucs_status_string(t);
        return NIXL_ERR_BACKEND;
    }
}

} // namespace nixl::ucx
