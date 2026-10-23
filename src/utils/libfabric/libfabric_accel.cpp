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

#include "libfabric_accel.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <set>
#include <vector>

namespace {

/****************************************
 * hwloc predicates
 *****************************************/

/** @brief Whether @p obj is a PCI device from @p vendor_id in the display/3D class range. */
bool
isDisplayClassPciDev(hwloc_obj_t obj, uint16_t vendor_id) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }
    if (obj->attr->pcidev.vendor_id != vendor_id) {
        return false;
    }
    // 0x300-0x3ff are display controllers; 0x302 is "3D controller", which is what a
    // compute-oriented NVIDIA board reports.
    const uint16_t class_id = obj->attr->pcidev.class_id;
    return (class_id >= 0x300 && class_id < 0x400);
}

/**
 * @brief Whether @p obj is an NVIDIA GPU. Pure hwloc; see accelIfaceOfHwlocObj for why.
 */
bool
isNvidiaAccel(hwloc_obj_t obj) {
    return isDisplayClassPciDev(obj, 0x10de);
}

/**
 * @brief Whether @p obj is an AMD GPU. Pure hwloc; see accelIfaceOfHwlocObj for why.
 *
 * Wider than the display class the other GPUs use: an Instinct accelerator such as MI300X reports
 * class 0x1200 (processing accelerator) rather than a display controller class.
 */
bool
isAmdAccel(hwloc_obj_t obj) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE || obj->attr->pcidev.vendor_id != 0x1002) {
        return false;
    }
    const uint16_t class_id = obj->attr->pcidev.class_id;
    return (class_id >= 0x300 && class_id < 0x400) || (class_id >= 0x1200 && class_id < 0x1300);
}

/**
 * @brief Whether @p obj is a Neuron accelerator, by vendor ID and device ID.
 *
 * Pure hwloc, so Trainium cards are counted in a build without libnrt. The device-ID allowlist is
 * the cost of that: unlike a GPU, a Neuron card has no PCI class separating it from the other
 * Annapurna devices sharing vendor 0x1d0f, so each generation has to be added here. libnrt exposes
 * no device enumeration to replace it with -- nrt_get_attached_efa_bdf() answers only about a
 * pointer -- which is why this is the one vendor with no SDK-side alternative.
 */
bool
isNeuronAccel(hwloc_obj_t obj) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }
    // Amazon vendor ID is 0x1d0f
    if (obj->attr->pcidev.vendor_id != 0x1d0f) {
        return false;
    }
    static const uint16_t NEURON_DEVICE_IDS[] = {
        0x7264, // INF2
        0x7164, // TRN1
        0x7364, // TRN2
        0x7564, // TRN3_DEVICE_0
        0x7565, // TRN3_DEVICE_1
    };
    return std::find(std::begin(NEURON_DEVICE_IDS),
                     std::end(NEURON_DEVICE_IDS),
                     obj->attr->pcidev.device_id) != std::end(NEURON_DEVICE_IDS);
}

/**
 * @brief The Level Zero driver's discrete-GPU addresses, in hwloc's normalisation.
 *
 * Cached because the only caller walks every PCI device in the topology, and re-asking the shim per
 * object would re-enter it thousands of times for an answer that cannot change: the shim's device
 * list is built once, during its one-time initialisation.
 */
const std::set<std::string> &
zeAccelBdfs() {
    static const std::set<std::string> bdfs = []() {
        std::set<std::string> out;

        const size_t count = nfi_hmem_device_bdfs(FI_HMEM_ZE, nullptr, 0);
        if (count == 0) {
            return out;
        }

        std::vector<struct nfi_hmem_bus_id> ids(count);
        const size_t written = nfi_hmem_device_bdfs(FI_HMEM_ZE, ids.data(), ids.size());
        for (size_t i = 0; i < std::min(written, ids.size()); ++i) {
            out.insert(accelBusIdString(ids[i], NFI_HMEM_BUS_ACCELERATOR));
        }
        return out;
    }();
    return bdfs;
}

/**
 * @brief Whether @p obj is a Level-Zero-capable discrete Intel accelerator.
 *
 * Membership in the driver's own list is the entire test -- no PCI class range, no per-generation
 * device-ID allowlist. Both alternatives are wrong in one direction or the other: a discrete Intel
 * GPU and Intel integrated graphics share class 0x0300 so the class cannot separate them, while
 * compute-only parts carry no display class at all and a class check silently drops them.
 */
bool
isIntelAccel(hwloc_obj_t obj) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }

    const std::set<std::string> &enumerated = zeAccelBdfs();
    if (enumerated.empty()) {
        return false;
    }

    struct nfi_hmem_bus_id bus;
    bus.domain = static_cast<uint16_t>(obj->attr->pcidev.domain);
    bus.bus = static_cast<uint8_t>(obj->attr->pcidev.bus);
    bus.slot = static_cast<uint8_t>(obj->attr->pcidev.dev);
    bus.func = static_cast<uint8_t>(obj->attr->pcidev.func);

    return enumerated.count(accelBusIdString(bus, NFI_HMEM_BUS_ACCELERATOR)) > 0;
}

/****************************************
 * Diagnostics bridge
 *****************************************/

void
logBridge(enum nfi_hmem_log_level level, const char *msg) {
    switch (level) {
    case NFI_HMEM_LOG_ERROR:
        NIXL_ERROR << "hmem: " << msg;
        break;
    case NFI_HMEM_LOG_WARN:
        NIXL_WARN << "hmem: " << msg;
        break;
    case NFI_HMEM_LOG_INFO:
        NIXL_INFO << "hmem: " << msg;
        break;
    case NFI_HMEM_LOG_DEBUG:
    default:
        NIXL_DEBUG << "hmem: " << msg;
        break;
    }
}

} // namespace

/****************************************
 * Public interface
 *****************************************/

void
nixlAccelInstallLogBridge() {
    static std::once_flag once;
    std::call_once(once, []() { nfi_hmem_set_log_fn(logBridge); });
}

const std::vector<enum fi_hmem_iface> &
nixlAccelKnownIfaces() {
    static const std::vector<enum fi_hmem_iface> ifaces = []() {
        size_t count = 0;
        const enum fi_hmem_iface *raw = nfi_hmem_ifaces(&count);
        return std::vector<enum fi_hmem_iface>(raw, raw + count);
    }();
    return ifaces;
}

std::string
accelBusIdString(const struct nfi_hmem_bus_id &bus, enum nfi_hmem_bus_kind kind) {
    if (kind == NFI_HMEM_BUS_NONE) {
        return "";
    }

    char buf[32];
    snprintf(buf,
             sizeof(buf),
             "%x:%02x:%02x.%x",
             static_cast<unsigned>(bus.domain),
             static_cast<unsigned>(bus.bus),
             static_cast<unsigned>(bus.slot),
             static_cast<unsigned>(bus.func));
    return std::string(buf);
}

std::string
accelBusIdString(const struct nfi_hmem_info &info) {
    return accelBusIdString(info.bus, info.bus_kind);
}

enum fi_hmem_iface
accelIfaceOfHwlocObj(hwloc_obj_t obj) {
    if (isNvidiaAccel(obj)) {
        return FI_HMEM_CUDA;
    }
    if (isAmdAccel(obj)) {
        return FI_HMEM_ROCR;
    }
    if (isNeuronAccel(obj)) {
        return FI_HMEM_NEURON;
    }
    if (isIntelAccel(obj)) {
        return FI_HMEM_ZE;
    }
    return FI_HMEM_SYSTEM;
}

bool
accelIfaceIsGroupable(enum fi_hmem_iface iface) {
    switch (iface) {
    case FI_HMEM_CUDA:
    case FI_HMEM_ROCR:
    case FI_HMEM_ZE:
        return true;
    case FI_HMEM_NEURON:
        // A Trainium card carries its EFA device onboard, so a Neuron pointer reports the NIC's
        // address and resolves through the NIC map. Admitting it to grouping would look up an
        // accelerator that is not there.
        return false;
    default:
        return false;
    }
}

enum fi_hmem_iface
accelUnclaimedVramIface() {
    // Strict wherever CUDA is usable; see the header. Asked of the runtimes rather than the
    // topology so the answer does not depend on which libfabric provider was selected.
    if (nfi_hmem_iface_available(FI_HMEM_CUDA)) {
        return FI_HMEM_SYSTEM;
    }
    // Probe order (see nfi_vendors[]), minus CUDA.
    for (const auto iface : {FI_HMEM_NEURON, FI_HMEM_ZE}) {
        if (nfi_hmem_iface_available(iface)) {
            return iface;
        }
    }
    return FI_HMEM_SYSTEM;
}
