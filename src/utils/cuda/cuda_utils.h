/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* forward declaration for the internal CUDA data structure */

#include <nixl.h>
#include <memory>

/****************************************
 * Pointer Context
*****************************************/

namespace nixlCuda {
    class memCtx {
    public:
        enum memory_t {
            MEM_NONE,
            MEM_HOST,
            MEM_DEV,
            MEM_VMM_DEV,
        } ;
    public:
        memCtx() = default;
        memCtx( memCtx& ) = delete;
        memCtx( const memCtx& ) = delete;
        memCtx( memCtx && ) = delete;
        memCtx( const memCtx && ) = delete;
        memCtx& operator=(const memCtx& ) = delete;

        virtual ~memCtx() = default;


        [[nodiscard]]
        virtual memory_t getMemType() const {
            return MEM_NONE;
        }

        [[nodiscard]]
        virtual nixl_status_t set() {
            // no-op for non-CUDA case
            return NIXL_SUCCESS;
        }

        [[nodiscard]]
        virtual nixl_status_t enableAddr(const void *address, uint64_t chkDevId) {
            return NIXL_SUCCESS;
        }
    };

    [[nodiscard]]
    std::unique_ptr<memCtx> makeMemCtx();

    [[nodiscard]]
    uint32_t numDevices();
}
