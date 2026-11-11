/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_CAPS_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_CAPS_H

/**
 * @brief Whether an accelerator is attached to the CPU coherently.
 *
 * A coherent accelerator accesses pageable host memory through the host's page tables, and
 * a NIC that meets it only at the CPU reaches its memory through the CPU. On a host without
 * coherence, such a NIC crosses the PCIe root complex for every access.
 */
enum class nixlAccelCoherence {
    UNKNOWN, ///< The query is unavailable (no CUDA driver API, ROCm build) or failed
    COHERENT, ///< The accelerator uses the host's page tables for pageable memory
    NOT_COHERENT, ///< The accelerator reaches host memory only over PCIe
};

/**
 * @brief Queries the coherence of the CUDA device at a PCI address.
 *
 * Looks the device up by PCI address, so it needs no current CUDA context and can run at
 * plugin init, before the application selects a device. Reads
 * CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES.
 *
 * Declared extern "C" so unit tests can substitute it with the linker's --wrap.
 *
 * @param pci_bus_id PCI address in "dddd:bb:dd.f" form.
 * @return The device's coherence, or UNKNOWN when it cannot be determined.
 */
extern "C" nixlAccelCoherence
nixlCudaAccelCoherence(const char *pci_bus_id);

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_CUDA_CAPS_H
