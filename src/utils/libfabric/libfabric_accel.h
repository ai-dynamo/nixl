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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_H

#include "nfi_hmem.h"

#include <hwloc.h>
#include <string>
#include <vector>

/**
 * @brief nixl's side of the accelerator boundary: the part @ref nfi_hmem.h deliberately excludes.
 *
 * The shim answers libfabric-shaped questions -- what owns this pointer, which fi_mr_attr arm, is
 * this runtime usable. Three kinds of question are left out of it on purpose, and they live here:
 *
 *   1. **hwloc.** Accelerator discovery in nixl is hwloc-driven, because rail selection walks PCIe
 *      ancestry. libfabric does not depend on hwloc and never should, so the shim reports PCI
 *      addresses as integers and the matching against an hwloc topology happens here.
 *   2. **Policy.** Whether a vendor's devices join PCIe-proximity NIC grouping, and whether an
 *      unattributable VRAM_SEG pointer is a caller error, are nixl rail-selection decisions. They
 *      would be meaningless inside libfabric.
 *   3. **C++ ergonomics.** The shim is C so it can be donated upstream; the scope guard that makes
 *      it pleasant to use is not part of that contract.
 *
 * Adding an accelerator vendor: the shim gets a table entry, and this file gets a `groupable`
 * decision plus, if hwloc cannot identify the hardware from PCI attributes alone, a branch in
 * @ref accelIfaceOfHwlocObj.
 */

/****************************************
 * Diagnostics
 *****************************************/

/**
 * @brief Routes the shim's diagnostics into NIXL_LOG. Idempotent, safe to call from anywhere.
 *
 * Call before the first shim query if you want to see vendor initialisation messages, since that
 * happens lazily on first use and is the most informative thing it logs.
 */
void
nixlAccelInstallLogBridge();

/****************************************
 * Vendor enumeration
 *****************************************/

/**
 * @brief Every interface the shim knows about, in probe order, as a range-for-able container.
 *
 * A convenience over @ref nfi_hmem_ifaces so callers can iterate vendors without naming any. Says
 * nothing about whether a runtime is usable; ask @ref nfi_hmem_iface_available for that.
 */
const std::vector<enum fi_hmem_iface> &
nixlAccelKnownIfaces();

/****************************************
 * PCI addresses
 *****************************************/

/**
 * @brief Renders a shim PCI address the way hwloc does, so both sides key alike.
 *
 * hwloc writes "%x:%02x:%02x.%x" -- an unpadded domain, everything else padded. Vendor SDKs
 * generally zero-pad the domain ("0000:59:00.0"), which is why nixlLibfabricTopology renormalises
 * anything arriving as a string. Values from the shim are integers and go straight through here.
 *
 * @return Empty string when @p kind is NFI_HMEM_BUS_NONE, so callers can test for "no address"
 *         without inspecting the kind separately.
 */
std::string
accelBusIdString(const struct nfi_hmem_bus_id &bus, enum nfi_hmem_bus_kind kind);

/** @brief @ref accelBusIdString for the address inside an info block. */
std::string
accelBusIdString(const struct nfi_hmem_info &info);

/****************************************
 * hwloc matching
 *****************************************/

/**
 * @brief The interface whose vendor owns @p obj, or FI_HMEM_SYSTEM if none does.
 *
 * The single place hwloc objects are attributed to a vendor. Null and non-PCI objects are rejected
 * rather than dereferenced.
 *
 * Two different strategies, per vendor, and the difference is deliberate:
 *
 *   - **NVIDIA and Neuron are matched from hwloc alone** (vendor ID, plus PCI class or a device-ID
 *     allowlist). That keeps them discoverable in a build without the vendor SDK, and -- more
 *     importantly for CUDA -- keeps discovery independent of CUDA_VISIBLE_DEVICES. Accelerator
 *     discovery describes the machine; masking it to the process's view would report one GPU on a
 *     masked 8-GPU host and repartition every NIC.
 *   - **Intel is matched against the Level Zero driver's own device list** via
 *     @ref nfi_hmem_device_bdfs. There is no hwloc-only alternative: a discrete Intel GPU and Intel
 *     integrated graphics share vendor ID 0x8086 and PCI class 0x0300, so nothing in the PCI
 *     attributes separates them, and compute-only parts carry no display class at all. Answers
 *     false when Level Zero is unavailable.
 */
enum fi_hmem_iface
accelIfaceOfHwlocObj(hwloc_obj_t obj);

/****************************************
 * Rail-selection policy
 *****************************************/

/**
 * @brief Whether @p iface's devices take part in PCIe-proximity NIC grouping.
 *
 * Distinct from being an accelerator at all. Neuron devices are accelerators for counting purposes
 * but must stay out of grouping: a Trainium card carries its EFA device onboard, so a Neuron pointer
 * reports a NIC address (NFI_HMEM_BUS_NIC) and resolves through the NIC map instead of through an
 * accelerator-to-NIC map.
 */
bool
accelIfaceIsGroupable(enum fi_hmem_iface iface);

/**
 * @brief The runtime a VRAM_SEG pointer no vendor claimed should be registered under.
 *
 * FI_HMEM_SYSTEM, meaning the unclaimed pointer is a caller error, whenever CUDA is usable: device
 * memory cuPointerGetAttributes() does not recognise is a bug in the caller, and guessing another
 * runtime on such a host -- one with an Intel iGPU visible to Level Zero, say -- would register
 * memory that is most likely not that runtime's either, under a CUDA device index it does not
 * share. Otherwise the first usable runtime whose query can miss its own memory, in probe order:
 * Level Zero sees allocations from driver 0's context only, and Neuron only pointers libnrt
 * attributes to an EFA device. FI_HMEM_SYSTEM again if there is none.
 */
enum fi_hmem_iface
accelUnclaimedVramIface();

/****************************************
 * Context scope
 *****************************************/

/**
 * @brief RAII wrapper over @ref nfi_hmem_scope_push / @ref nfi_hmem_scope_pop.
 *
 * Bracket the operations that need a current accelerator context. Which those are is not obvious, so
 * to be explicit: **registration needs one, detection does not.** libfabric exports a dmabuf handle
 * from inside fi_mr_regattr() for providers configured that way, and that export requires a context
 * on the buffer's device; attributing a pointer requires nothing, because unified virtual addressing
 * lets the driver identify any allocation in the process. UCX draws the line in exactly the same
 * place, and for the same stated reason.
 *
 * Posting is bracketed too, because a provider may take a host-copy path for small transfers.
 *
 * A no-op for every runtime with no thread-current state (Level Zero, Neuron) and for host memory,
 * so callers can construct one unconditionally rather than testing first.
 */
class nixlAccelScope {
public:
    /** @brief Binds the device that owns the buffer @p info describes. */
    explicit nixlAccelScope(const struct nfi_hmem_info &info)
        : nixlAccelScope(info.iface, info.device) {}

    /** @brief Binds @p device of @p iface, for a caller holding the pair rather than an info block. */
    nixlAccelScope(enum fi_hmem_iface iface, uint64_t device)
        : scope_{},
          iface_(iface),
          device_(device),
          ok_(false) {
        ok_ = (nfi_hmem_scope_push(iface, device, &scope_) == 0);
    }

    ~nixlAccelScope() {
        nfi_hmem_scope_pop(&scope_);
    }

    nixlAccelScope(const nixlAccelScope &) = delete;
    nixlAccelScope &
    operator=(const nixlAccelScope &) = delete;

    /**
     * @brief Whether the binding succeeded.
     *
     * A runtime that has nothing to bind (or is unavailable) reports success here, so false means
     * the runtime actively refused, and the operation the scope brackets should fail rather than
     * run with no context on the buffer's device.
     */
    bool
    ok() const {
        return ok_;
    }

    /**
     * @brief Whether this scope already binds what @p info needs, so a caller posting several
     * buffers can keep one scope across them and rebind only when the device changes. False for a
     * scope whose binding failed, so a failure is retried rather than reused.
     */
    bool
    holds(const struct nfi_hmem_info &info) const {
        return ok_ && iface_ == info.iface && device_ == info.device;
    }

private:
    struct nfi_hmem_scope scope_;
    enum fi_hmem_iface iface_;
    uint64_t device_;
    bool ok_;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_H
