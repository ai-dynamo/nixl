/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_IPC_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_IPC_H

#include <cstdint>
#include <memory>

// CUDA IPC export and import, which give a peer process on the same host a direct pointer
// to registered GPU memory
namespace LibfabricUtils {

/** CUDA IPC handle of a registered region, sent to peers with the memory keys */
struct CudaIpcExport {
    char handle[64]; ///< CUipcMemHandle of the allocation that holds the region
    uint64_t offset; ///< Offset of the region in the allocation
    char host_id[40]; ///< boot_id of the exporting host
};

/** Exports the allocation that holds buffer; false if it cannot be exported */
bool
cudaIpcExportRange(const void *buffer, CudaIpcExport &out);

/** Maps a peer allocation until released; nullptr if on another host or not mappable */
std::shared_ptr<void>
cudaIpcImport(const CudaIpcExport &in);

} // namespace LibfabricUtils

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_IPC_H
