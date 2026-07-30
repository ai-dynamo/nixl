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
#ifndef MARSHAL_BACKEND_H
#define MARSHAL_BACKEND_H

#include "nixl_types.h"
#include "nixl_descriptors.h"
#include "absl/types/span.h"
#include "marshal_types.h"

#include <memory>
#include <optional>
#include <string>

namespace nixlMarshal {

class backend;

/**
 * @class asyncHandle
 * @brief Type-erased polymorphic base for an in-flight slot operation, parameterized by the
 *        completion-data type. Returned by backend::inbound/outboundProcessSlot.
 */
template<typename CompletionT> class asyncHandle {
protected:
    asyncHandle(std::weak_ptr<backend> backend) : backend_(std::move(backend)) {}

    std::weak_ptr<backend> backend_;

public:
    virtual ~asyncHandle() = default;

    /**
     * @brief  Poll for completion of the operation.
     * @return The completion data on success, std::nullopt while the operation is still pending.
     * @throw  std::runtime_error if the backend fails while checking for completion.
     */
    virtual std::optional<CompletionT>
    checkForCompletion() = 0;
};

/**
 * @class asyncHandleImpl
 * @brief CRTP layer that backends inherit from. Implements checkForCompletion() once as final and
 *        statically forwards to Derived::checkForCompletionImpl(), so the per-backend polling logic
 *        is resolved at compile time (no extra vtable lookup, fully inlinable).
 */
template<typename Derived, typename CompletionT>
class asyncHandleImpl : public asyncHandle<CompletionT> {
protected:
    using asyncHandle<CompletionT>::asyncHandle;

public:
    std::optional<CompletionT>
    checkForCompletion() final {
        return static_cast<Derived *>(this)->checkForCompletionImpl();
    }
};

using inbound_async_handle_t = asyncHandle<inboundSlotCompletionData>;
using outbound_async_handle_t = asyncHandle<outboundSlotCompletionData>;

/**
 * @class backend
 * @brief Abstract interface for the transformation layer of the marshal sub-service.
 *
 * @note The backend must provide a static createBackend function to force creation of a shared
 * pointer.
 * @note The backend can provide a sizing suggestion for the total buffer size per descriptor
 * given the chunked payload size the service agent has picked (see
 * recommendServiceMemSize in nixl_service.h).
 */
class backend : public std::enable_shared_from_this<backend> {
protected:
    backend() = default;

public:
    virtual ~backend() = default;

    /**
     * @brief  Returns the memory spaces supported by this backend.
     *
     * @return std::vector<mem_space_t> A vector (by reference) containing the
     *         supported mem_space_t values for this backend.
     */
    virtual const std::vector<mem_space_t> &
    getSupportedMemSpaces() const = 0;

    /**
     * @brief  Asynchronously drain a staging slot to destination descriptors (inbound).
     *         Submits the GPU operation and returns immediately. The backend must consume or copy
     *         `metadata` during the call (no retention guarantees).
     *
     * @param  buffers   Common per-slot buffers and stream.
     * @param  metadata  Marshalling metadata received from the remote agent; used to unmarshall
     *                   the slot data on the destination side.
     * @param  opts      Optional process inputs. Inbound chunk division is provided as a vector
     *                   of adjacent piece sizes via option_t::CHUNK_DIVISION.
     * @return Inbound async handle for the operation.
     * @throw  std::runtime_error if the backend fails to process the slot.
     */
    virtual std::unique_ptr<inbound_async_handle_t>
    inboundProcessSlot(const slotBuffers &buffers,
                       const std::string &metadata,
                       const process_slot_input_options_t &opts = {}) = 0;

    /**
     * @brief  Asynchronously fill a staging slot from source descriptors (outbound).
     *         Submits the GPU operation and returns immediately.
     *
     * @param  buffers Common per-slot buffers and stream.
     * @return Outbound async handle for the operation; checkForCompletion yields
     *         outboundSlotCompletionData (size/output options + metadata produced for the wire).
     * @throw  std::runtime_error if the backend fails to process the slot.
     */
    virtual std::unique_ptr<outbound_async_handle_t>
    outboundProcessSlot(const slotBuffers &buffers,
                        const process_slot_input_options_t &opts = {}) = 0;

    /**
     *  @brief  Calculate the memory requirements for a single slot of the backend from
     * configuration.
     *  @note The chunked payload size is the size of the payload that will be transferred in a
     * single operation. For inbound operations, the backend will not exceed this size in the
     * destination descriptor. For outbound operations, the nixlServiceAgent will not exceed this
     * size in the source descriptor.
     *
     *  @return memoryRequirements    The memory requirements for a single slot of the backend.
     */
    virtual memoryRequirements
    getSlotMemoryRequirements() const = 0;
};

} // namespace nixlMarshal

#endif // MARSHAL_BACKEND_H
