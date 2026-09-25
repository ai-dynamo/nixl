/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_DMABUF_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_DMABUF_H

#include <cstddef>
#include <cstdint>

// CUDA dmabuf export for FI_MR_DMABUF memory registration
// A dmabuf fd describes GPU memory through one address window. On a platform whose GPU
// memory carries several windows, such as GB200, the window a NIC reaches depends on its
// route to the GPU, so the export selects the mapping type.
namespace LibfabricUtils {

/**
 * @brief One exported CUDA dmabuf for an FI_MR_DMABUF registration
 *
 * The fields map onto struct fi_mr_dmabuf: the registered region is [base_addr + offset,
 * base_addr + offset + len). The fd holds -1 until an export succeeds.
 */
struct CudaDmabufExport {
    int fd = -1;
    uint64_t offset = 0;
    void *base_addr = nullptr;
    bool pcie_mapping = false; ///< True when the export carries the PCIe (BAR1) mapping
};

/**
 * @brief Widens [buffer, length) to the host page boundaries cuMemGetHandleForAddressRange
 * requires.
 * @param buffer The start of the region being registered.
 * @param length The length of the region in bytes; 0 is rounded up to one page.
 * @param page_size The host page size, a power of two.
 * @param[out] aligned_base The page-aligned base to pass to CUDA.
 * @param[out] aligned_length The page-aligned length to pass to CUDA.
 * @param[out] offset The offset of buffer inside the exported range.
 */
inline void
cudaDmabufAlignRange(const void *buffer,
                     size_t length,
                     size_t page_size,
                     uintptr_t &aligned_base,
                     size_t &aligned_length,
                     uint64_t &offset) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(buffer);
    const uintptr_t page_mask = static_cast<uintptr_t>(page_size) - 1;
    const size_t span = (length == 0) ? 1 : length;

    aligned_base = addr & ~page_mask;
    // Round the last touched byte up to its page end, matching libfabric's
    // ofi_get_page_end(addr + size - 1) + 1.
    const uintptr_t aligned_end = ((addr + span - 1) | page_mask) + 1;
    aligned_length = static_cast<size_t>(aligned_end - aligned_base);
    offset = static_cast<uint64_t>(addr - aligned_base);
}

/**
 * @brief Queries whether a CUDA device supports a dmabuf export that can be registered.
 * @note Requires both device dmabuf support and the RDMA dmabuf import ioctls the kernel
 * provides from 5.12 onwards.
 * @param device_id The CUDA device ordinal holding the memory.
 * @return True if succeeded, otherwise false.
 */
bool
cudaDmabufExportSupported(int device_id);

/**
 * @brief Exports a dmabuf covering the page-aligned range around [buffer, length).
 * @note On a platform that offers a single mapping, CUDA rejects the PCIe flag and the
 * export retries with the platform default.
 * @param buffer The start of the region being registered.
 * @param length The length of the region in bytes.
 * @param device_id The CUDA device ordinal holding the memory.
 * @param pcie_mapping Whether to request the PCIe (BAR1) mapping.
 * @param[out] out The resulting export; out.pcie_mapping reports the mapping it carries
 * (valid only if call succeeds).
 * @return True if succeeded, otherwise (out.fd remains -1) false.
 */
bool
cudaDmabufExportRange(void *buffer,
                      size_t length,
                      int device_id,
                      bool pcie_mapping,
                      CudaDmabufExport &out);

/**
 * @brief Closes the exported fd and resets the export.
 * @note fi_mr_regattr() takes its own reference to the dma_buf, so the fd can be closed
 * once registration returns. The call is idempotent.
 * @param to_close The export to close.
 */
void
cudaDmabufExportClose(CudaDmabufExport &to_close);

} // namespace LibfabricUtils

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_DMABUF_H
