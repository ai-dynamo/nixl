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

#ifndef MARSHAL_TYPES_H
#define MARSHAL_TYPES_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <cuda_runtime.h>

#include "absl/types/span.h"

enum class nixl_marshal_compress_data_type_t : uint8_t;

namespace nixlMarshal {

/**
 * @enum mem_space_t
 * @brief The memory space of the staging slot.
 */
enum class mem_space_t {
    INVALID,
    HOST,
    DEVICE,
};

struct runtimeBuffer {
    std::byte *data;
    size_t size;
    mem_space_t space;

    runtimeBuffer(absl::Span<std::byte> span, mem_space_t sp)
        : data(span.data()),
          size(span.size()),
          space(sp) {}

    runtimeBuffer() : data(nullptr), size(0), space(mem_space_t::INVALID) {}
};

struct inboundSlotCompletionData {
    size_t size;
};

struct outboundSlotCompletionData {
    size_t size;
    // Carried in the wire protocol and consumed by the inbound side to unmarshall.
    std::string metadata = {};
};

// TODO-Eyal: remove enum and maps, change to set.
enum class option_t {
    READ_ONLY_REFERENCE_STRUCTURED_MEMORY,
    SLOT_OVERHEAD,
    WRITEABLE_WORKSPACE_MEMORY,
    USER_CUDA_STREAM,
    ANS_DATA_TYPE,
};

namespace ReadOnlyReferenceStructuredMemory {

    struct processSlotInput {
        runtimeBuffer ref;
        size_t elementSize;
    };

} // namespace ReadOnlyReferenceStructuredMemory

namespace SlotOverhead {

    struct memoryRequirements {
        size_t slotOverheadSize;
    };

} // namespace SlotOverhead

namespace WriteableWorkspaceMemory {

    struct memoryRequirements {
        size_t slotWorkspaceSize;
    };

    struct processSlotInput {
        runtimeBuffer workspace;
    };

} // namespace WriteableWorkspaceMemory

namespace UserCudaStream {

    struct processSlotInput {
        cudaStream_t stream;
    };

} // namespace UserCudaStream

namespace AnsDataType {

    struct processSlotInput {
        nixl_marshal_compress_data_type_t dataType;
    };

} // namespace AnsDataType

using process_slot_input_value_t = std::variant<ReadOnlyReferenceStructuredMemory::processSlotInput,
                                                WriteableWorkspaceMemory::processSlotInput,
                                                UserCudaStream::processSlotInput,
                                                AnsDataType::processSlotInput>;
using process_slot_input_options_t = std::unordered_map<option_t, process_slot_input_value_t>;

using memory_requirement_value_t =
    std::variant<SlotOverhead::memoryRequirements, WriteableWorkspaceMemory::memoryRequirements>;
using memory_requirements_options_t = std::unordered_map<option_t, memory_requirement_value_t>;

/**
 * @struct memoryRequirements
 * @brief  The backend's per-slot size requirements.
 */
struct memoryRequirements {
    memory_requirements_options_t opts = {};
};

struct slotBuffers {
    runtimeBuffer src;
    runtimeBuffer dst;
};

} // namespace nixlMarshal

#endif // MARSHAL_TYPES_H
