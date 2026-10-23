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

/*
 * Tests for the libfabric HMEM shim (src/utils/libfabric/nfi_hmem.c) and nixl's side of the
 * accelerator boundary (src/utils/libfabric/libfabric_accel.cpp).
 *
 * Written to pass on any host: with no accelerator, with an NVIDIA GPU, with a Neuron device, and
 * with an Intel Xe GPU. Assertions that need a particular vendor are gated on
 * nfi_hmem_iface_available() and skipped with a visible log line, so a skip never reads as a pass.
 *
 * Two things are worth testing here that were not testable before the split:
 *
 *   - the shim's contract on its own, with no nixl types involved, which is what makes it
 *     donatable to libfabric;
 *   - that a context scope *restores* the caller's binding rather than leaving the thread bound,
 *     which is the defect the push/pop design fixes.
 */

#include "libfabric/libfabric_accel.h"
#include "libfabric/nfi_hmem.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

namespace {

int failures = 0;

void
check(bool cond, const std::string &what) {
    if (cond) {
        NIXL_INFO << "  PASS: " << what;
    } else {
        NIXL_ERROR << "  FAIL: " << what;
        ++failures;
    }
}

/****************************************
 * 0. The header's C-compatible surface
 *****************************************/

/*
 * The shim's implementation is ordinary C++, but its *header* is deliberately C-compatible: POD
 * structs, plain functions, extern "C". That is what keeps donating it to libfabric -- a C project --
 * a contained change, and it is the sort of property that rots silently the first time someone adds a
 * std::string field or a default argument.
 *
 * A compile-time check is the only kind that can catch that, so these are static_asserts rather than
 * runtime checks. The C99 compile of the header itself is exercised separately by the build.
 */
static_assert(std::is_standard_layout_v<nfi_hmem_bus_id>,
              "nfi_hmem_bus_id must stay a C-compatible POD");
static_assert(std::is_standard_layout_v<nfi_hmem_info>,
              "nfi_hmem_info must stay a C-compatible POD");
static_assert(std::is_standard_layout_v<nfi_hmem_scope>,
              "nfi_hmem_scope must stay a C-compatible POD");
static_assert(std::is_trivially_copyable_v<nfi_hmem_info>,
              "nfi_hmem_info must be memcpy-able, as a C caller would expect");
static_assert(std::is_trivially_destructible_v<nfi_hmem_scope>,
              "nfi_hmem_scope must not need a destructor; C callers pair push/pop by hand");
// Every entry point must have C linkage, or a C consumer cannot link against it.
static_assert(std::is_same_v<decltype(nfi_hmem_query_ptr), int(const void *, nfi_hmem_info *)>,
              "nfi_hmem_query_ptr must keep its C signature");
static_assert(std::is_same_v<decltype(nfi_hmem_iface_name), const char *(enum fi_hmem_iface)>,
              "nfi_hmem_iface_name must keep its C signature");

/****************************************
 * 1. The vendor list
 *****************************************/

void
testIfaceList() {
    NIXL_INFO << "1. Testing the shim's vendor list";

    size_t count = 0;
    const enum fi_hmem_iface *raw = nfi_hmem_ifaces(&count);
    check(raw != nullptr, "nfi_hmem_ifaces() never returns null");
    check(count > 0, "the shim knows at least one interface");

    const std::vector<enum fi_hmem_iface> &ifaces = nixlAccelKnownIfaces();
    check(ifaces.size() == count, "the C++ view has the same size as the C array");
    check(std::equal(ifaces.begin(), ifaces.end(), raw),
          "the C++ view preserves the C array's order, which is the probe order");

    // FI_HMEM_SYSTEM is not a vendor and must never appear, or the probe loop would try to claim
    // host memory as device memory.
    check(std::find(ifaces.begin(), ifaces.end(), FI_HMEM_SYSTEM) == ifaces.end(),
          "FI_HMEM_SYSTEM is not listed as a vendor");

    check(std::set<enum fi_hmem_iface>(ifaces.begin(), ifaces.end()).size() == ifaces.size(),
          "no interface is listed twice");

    /*
     * Probe order must descend, matching libfabric's ofi_get_hmem_iface(), which walks hmem_ops[]
     * from the highest interface down. Several providers -- cxi, rxm, opx -- call that function for
     * themselves, and rxm is what verbs runs under, so a different order here could make the shim
     * and the provider disagree about one pointer on a host with two runtimes.
     */
    bool descending = true;
    for (size_t i = 1; i < ifaces.size(); ++i) {
        descending = descending && (ifaces[i] < ifaces[i - 1]);
    }
    check(descending, "probe order descends, matching libfabric's own ofi_get_hmem_iface()");

    std::string names;
    for (const enum fi_hmem_iface iface : ifaces) {
        names += std::string(nfi_hmem_iface_name(iface)) + " ";
    }
    NIXL_INFO << "   probe order: " << names;
}

/****************************************
 * 2. Interface names
 *****************************************/

void
testIfaceNames() {
    NIXL_INFO << "2. Testing interface names";

    check(strcmp(nfi_hmem_iface_name(FI_HMEM_SYSTEM), "SYSTEM") == 0,
          "FI_HMEM_SYSTEM is named SYSTEM");
    /*
     * Names come from libfabric's fi_tostr(), not from the shim's vendor table, so an interface the
     * shim has no entry for is still named correctly. That is the point of not keeping a second
     * list: ROCR and SynapseAI exist in libfabric and would read as "Unknown" from a hand-written
     * table.
     */
    check(strcmp(nfi_hmem_iface_name(FI_HMEM_ROCR), "ROCR") == 0,
          "an interface with no vendor entry is still named from libfabric");
    check(strcmp(nfi_hmem_iface_name(FI_HMEM_SYNAPSEAI), "SYNAPSEAI") == 0,
          "SynapseAI is named too, though the shim has no entry for it");
    check(strcmp(nfi_hmem_iface_name(static_cast<enum fi_hmem_iface>(99)), "Unknown") == 0,
          "a value libfabric does not recognise is reported as Unknown");
    // The prefix strip is what keeps log lines like "CUDA=0 NEURON=0 ZE=1" short.
    check(strncmp(nfi_hmem_iface_name(FI_HMEM_CUDA), "FI_HMEM_", 8) != 0,
          "the redundant FI_HMEM_ prefix is stripped");

    bool all_named = true;
    for (const enum fi_hmem_iface iface : nixlAccelKnownIfaces()) {
        const char *name = nfi_hmem_iface_name(iface);
        if (name == nullptr || strcmp(name, "Unknown") == 0 || strcmp(name, "SYSTEM") == 0) {
            NIXL_ERROR << "  iface " << iface << " has no usable name";
            all_named = false;
        }
    }
    check(all_named, "every known interface has a name of its own");
}

/****************************************
 * 3. Availability consistency
 *****************************************/

void
testAvailability() {
    NIXL_INFO << "3. Testing availability";

    // Consistency, not a specific answer: whichever runtimes this host has, these have to agree, or
    // getSupportedMems() and the registerMemory() fallback would disagree about whether VRAM works.
    bool any = false;
    for (const enum fi_hmem_iface iface : nixlAccelKnownIfaces()) {
        const bool avail = nfi_hmem_iface_available(iface);
        NIXL_INFO << "   " << nfi_hmem_iface_name(iface) << ": "
                  << (avail ? "available" : "unavailable");
        any = any || avail;
    }
    check(any == nfi_hmem_available(), "nfi_hmem_available() agrees with the per-iface answers");

    const enum fi_hmem_iface primary = nfi_hmem_primary_iface();
    if (any) {
        check(primary != FI_HMEM_SYSTEM, "a host with a runtime reports a non-SYSTEM primary");
        check(nfi_hmem_iface_available(primary), "the primary interface is itself available");

        // The primary must be the *first available* in probe order, since that is what makes it the
        // documented tie-break rather than an arbitrary pick.
        enum fi_hmem_iface expected = FI_HMEM_SYSTEM;
        for (const enum fi_hmem_iface iface : nixlAccelKnownIfaces()) {
            if (nfi_hmem_iface_available(iface)) {
                expected = iface;
                break;
            }
        }
        check(primary == expected, "the primary is the first available interface in probe order");
    } else {
        check(primary == FI_HMEM_SYSTEM, "a host with no runtime reports SYSTEM as primary");
        NIXL_INFO << "   SKIP: no accelerator runtime on this host";
    }

    // Repeated calls must not flip: initialisation is once per process, and a consumer caches this.
    check(nfi_hmem_available() == any, "nfi_hmem_available() is stable across calls");
    check(nfi_hmem_primary_iface() == primary, "nfi_hmem_primary_iface() is stable across calls");
}

/****************************************
 * 4. Pointer attribution
 *****************************************/

void
testQueryPtr() {
    NIXL_INFO << "4. Testing pointer attribution";

    struct nfi_hmem_info info;

    // Argument checking, on a path that would otherwise dereference.
    check(nfi_hmem_query_ptr(nullptr, &info) == -EINVAL, "a null pointer is rejected");
    check(nfi_hmem_query_ptr(&info, nullptr) == -EINVAL, "a null output block is rejected");

    // Host memory is a successful answer of FI_HMEM_SYSTEM, not an error. Getting this wrong would
    // make every DRAM registration look like a failure.
    std::vector<char> host(4096, 0);
    memset(&info, 0xAB, sizeof(info));
    check(nfi_hmem_query_ptr(host.data(), &info) == 0, "host memory queries successfully");
    check(info.iface == FI_HMEM_SYSTEM, "host memory is reported as FI_HMEM_SYSTEM");
    check(info.bus_kind == NFI_HMEM_BUS_NONE, "host memory reports no PCI address");
    check(info.device == 0, "host memory reports device 0, not the 0xAB fill");

    // A stack address is host memory too, and is the case a vendor probe is most likely to
    // mis-claim, since it is not in any runtime's allocation table.
    int on_stack = 0;
    check(nfi_hmem_query_ptr(&on_stack, &info) == 0 && info.iface == FI_HMEM_SYSTEM,
          "a stack address is host memory");
}

/****************************************
 * 5. fi_mr_attr population
 *****************************************/

void
testFillMrAttr() {
    NIXL_INFO << "5. Testing fi_mr_attr population";

    struct nfi_hmem_info info;
    struct fi_mr_attr attr;

    // Fields the shim must not touch. Registration sets these and would break if they were reset.
    memset(&attr, 0, sizeof(attr));
    attr.access = FI_REMOTE_WRITE | FI_REMOTE_READ;
    attr.requested_key = 0x1234;
    attr.offset = 0;

    memset(&info, 0, sizeof(info));
    info.iface = FI_HMEM_SYSTEM;
    nfi_hmem_fill_mr_attr(&info, &attr);
    check(attr.iface == FI_HMEM_SYSTEM, "FI_HMEM_SYSTEM is written through");
    check(attr.access == (FI_REMOTE_WRITE | FI_REMOTE_READ) && attr.requested_key == 0x1234,
          "access and requested_key are left alone");

    // Each vendor's ordinal must land in its own union arm, unmodified. Re-encoding here would
    // double-encode the Level Zero ordinal, which is the arm most easily got wrong.
    info.iface = FI_HMEM_CUDA;
    info.device = 3;
    nfi_hmem_fill_mr_attr(&info, &attr);
    check(attr.iface == FI_HMEM_CUDA && attr.device.cuda == 3,
          "a CUDA ordinal lands in device.cuda unchanged");

    info.iface = FI_HMEM_ZE;
    info.device = static_cast<uint64_t>(fi_hmem_ze_device(0, 2));
    nfi_hmem_fill_mr_attr(&info, &attr);
    check(attr.iface == FI_HMEM_ZE && attr.device.ze == fi_hmem_ze_device(0, 2),
          "a Level Zero ordinal lands in device.ze already encoded, not re-encoded");

    info.iface = FI_HMEM_NEURON;
    info.device = static_cast<uint64_t>(-1);
    nfi_hmem_fill_mr_attr(&info, &attr);
    check(attr.iface == FI_HMEM_NEURON && attr.device.neuron == -1,
          "the Neuron sentinel survives the round trip to device.neuron");

    // Null arguments must be tolerated rather than dereferenced: this runs on the registration path.
    nfi_hmem_fill_mr_attr(nullptr, &attr);
    nfi_hmem_fill_mr_attr(&info, nullptr);
    check(true, "null arguments are ignored rather than dereferenced");
}

/****************************************
 * 6. Device inventory
 *****************************************/

void
testDeviceBdfs() {
    NIXL_INFO << "6. Testing device inventory";

    for (const enum fi_hmem_iface iface : nixlAccelKnownIfaces()) {
        const size_t count = nfi_hmem_device_bdfs(iface, nullptr, 0);
        NIXL_INFO << "   " << nfi_hmem_iface_name(iface) << ": " << count << " device(s)";

        if (count == 0) {
            continue;
        }

        // The count-only form and the filling form must agree, or a caller sizing a buffer from the
        // first would truncate silently.
        std::vector<struct nfi_hmem_bus_id> ids(count);
        const size_t written = nfi_hmem_device_bdfs(iface, ids.data(), ids.size());
        check(written == count, "the count-only and filling forms agree");

        // A short buffer must be honoured, not overrun.
        std::vector<struct nfi_hmem_bus_id> one(1);
        check(nfi_hmem_device_bdfs(iface, one.data(), 1) == count,
              "a short buffer still returns the true total");

        // Addresses have to render in hwloc's normalisation, or the topology lookup that matches
        // them against hwloc objects silently finds nothing and rail selection quietly falls back
        // to all rails.
        bool all_ok = true;
        for (const auto &id : ids) {
            const std::string rendered = accelBusIdString(id, NFI_HMEM_BUS_ACCELERATOR);
            NIXL_INFO << "     " << rendered;
            unsigned d, b, s, f;
            if (sscanf(rendered.c_str(), "%x:%x:%x.%x", &d, &b, &s, &f) != 4 || d != id.domain ||
                b != id.bus || s != id.slot || f != id.func) {
                all_ok = false;
            }
        }
        check(all_ok, "every reported address round-trips through hwloc's normalisation");
    }

    check(nfi_hmem_device_bdfs(FI_HMEM_SYNAPSEAI, nullptr, 0) == 0,
          "an interface with no table entry reports no devices");
}

/****************************************
 * 7. PCI address rendering
 *****************************************/

void
testBusIdString() {
    NIXL_INFO << "7. Testing PCI address rendering";

    struct nfi_hmem_bus_id bus;
    bus.domain = 0;
    bus.bus = 0xad;
    bus.slot = 0;
    bus.func = 0;

    // hwloc leaves the domain unpadded and pads the rest. A zero-padded domain here would fail
    // every topology lookup.
    check(accelBusIdString(bus, NFI_HMEM_BUS_ACCELERATOR) == "0:ad:00.0",
          "the domain is unpadded and the rest padded, as hwloc writes it");

    bus.domain = 0x10;
    bus.bus = 0x3a;
    bus.slot = 0x1f;
    bus.func = 3;
    check(accelBusIdString(bus, NFI_HMEM_BUS_NIC) == "10:3a:1f.3",
          "a non-zero domain and a non-zero function render correctly");

    check(accelBusIdString(bus, NFI_HMEM_BUS_NONE).empty(),
          "NFI_HMEM_BUS_NONE renders empty, so callers need not inspect the kind");

    struct nfi_hmem_info info;
    memset(&info, 0, sizeof(info));
    info.bus_kind = NFI_HMEM_BUS_NONE;
    check(accelBusIdString(info).empty(), "an info block with no address renders empty");
}

/****************************************
 * 8. Policy
 *****************************************/

void
testPolicy() {
    NIXL_INFO << "8. Testing rail-selection policy";

    check(accelIfaceIsGroupable(FI_HMEM_CUDA), "NVIDIA GPUs join PCIe-proximity grouping");
    check(accelIfaceIsGroupable(FI_HMEM_ZE), "Intel GPUs join PCIe-proximity grouping");
    // A Trainium card carries its EFA device onboard, so a Neuron pointer reports the NIC's address
    // and resolves through the NIC map. Admitting it to grouping would look for an accelerator that
    // is not in the map.
    check(!accelIfaceIsGroupable(FI_HMEM_NEURON), "Neuron devices stay out of grouping");
    check(!accelIfaceIsGroupable(FI_HMEM_SYSTEM), "host memory is not groupable");
    check(accelIfaceIsGroupable(FI_HMEM_ROCR), "AMD GPUs take part in grouping");
    check(!accelIfaceIsGroupable(FI_HMEM_SYNAPSEAI), "an unknown interface is not groupable");

    // The rule, stated case by case rather than re-derived: wherever CUDA is usable an unclaimed
    // VRAM pointer is a caller error, even with Level Zero or Neuron alongside (an NVIDIA box with
    // an Intel iGPU); elsewhere it falls back to Neuron, then Level Zero; with neither it is an
    // error too. CUDA is never the answer.
    const enum fi_hmem_iface fallback = accelUnclaimedVramIface();
    if (nfi_hmem_iface_available(FI_HMEM_CUDA)) {
        check(fallback == FI_HMEM_SYSTEM, "an unclaimed VRAM pointer is an error where CUDA is usable");
    } else if (nfi_hmem_iface_available(FI_HMEM_NEURON)) {
        check(fallback == FI_HMEM_NEURON, "without CUDA, unclaimed VRAM falls back to Neuron first");
    } else if (nfi_hmem_iface_available(FI_HMEM_ZE)) {
        check(fallback == FI_HMEM_ZE, "without CUDA or Neuron, unclaimed VRAM falls back to ZE");
    } else {
        check(fallback == FI_HMEM_SYSTEM, "with no fallback runtime, unclaimed VRAM is an error");
    }
    check(fallback != FI_HMEM_CUDA, "CUDA is never the unclaimed-VRAM fallback");

    // hwloc predicates must reject rather than dereference.
    check(accelIfaceOfHwlocObj(nullptr) == FI_HMEM_SYSTEM,
          "a null hwloc object is rejected, not dereferenced");
}

/****************************************
 * 9. Context scope
 *****************************************/

void
testScope() {
    NIXL_INFO << "9. Testing context scope";

    check(!nfi_hmem_needs_scope(FI_HMEM_SYSTEM), "host memory needs no context binding");
    check(!nfi_hmem_needs_scope(FI_HMEM_NEURON), "Neuron needs no context binding");
    check(!nfi_hmem_needs_scope(FI_HMEM_ZE), "Level Zero needs no context binding");

    struct nfi_hmem_scope scope;
    check(nfi_hmem_scope_push(FI_HMEM_CUDA, 0, nullptr) == -EINVAL,
          "a null scope is rejected");

    // A no-op runtime must still produce a scope that is safe to pop, and safe to pop twice, since
    // the RAII wrapper pops unconditionally in its destructor.
    memset(&scope, 0xCD, sizeof(scope));
    check(nfi_hmem_scope_push(FI_HMEM_SYSTEM, 0, &scope) == 0,
          "pushing a scope for host memory succeeds as a no-op");
    nfi_hmem_scope_pop(&scope);
    nfi_hmem_scope_pop(&scope);
    check(true, "popping a no-op scope twice is safe");

    {
        // The RAII wrapper must report success for a runtime with nothing to bind, so callers can
        // construct one unconditionally without treating "nothing to do" as a failure.
        const nixlAccelScope guard(FI_HMEM_SYSTEM, 0);
        check(guard.ok(), "nixlAccelScope reports success when there is nothing to bind");
    }

    if (!nfi_hmem_iface_available(FI_HMEM_CUDA)) {
        NIXL_INFO << "   SKIP: CUDA unavailable, cannot test real context push/pop";
        return;
    }
    check(nfi_hmem_needs_scope(FI_HMEM_CUDA), "CUDA needs context binding");

#ifdef HAVE_CUDA
    /*
     * The defect this design exists to fix: the previous implementation used cuCtxSetCurrent and
     * cudaSetDevice with no restore, so registering or posting on an application thread left that
     * thread bound to nixl's device. The application's next unqualified CUDA call then ran on the
     * wrong device. Push/pop has to leave the thread exactly as it found it.
     */
    CUcontext before = nullptr;
    const bool have_before = (cuCtxGetCurrent(&before) == CUDA_SUCCESS);

    {
        const nixlAccelScope guard(FI_HMEM_CUDA, 0);
        check(guard.ok(), "a CUDA scope pushes successfully");

        CUcontext inside = nullptr;
        check(cuCtxGetCurrent(&inside) == CUDA_SUCCESS && inside != nullptr,
              "a context is current inside the scope");
    }

    CUcontext after = nullptr;
    const bool have_after = (cuCtxGetCurrent(&after) == CUDA_SUCCESS);
    check(have_before == have_after && before == after,
          "the thread's context is exactly as it was before the scope");

    // Nesting must work too, because registration can be bracketed inside a caller's own scope.
    {
        const nixlAccelScope outer(FI_HMEM_CUDA, 0);
        {
            const nixlAccelScope inner(FI_HMEM_CUDA, 0);
            check(inner.ok(), "a nested CUDA scope pushes successfully");
        }
        CUcontext still = nullptr;
        check(cuCtxGetCurrent(&still) == CUDA_SUCCESS && still != nullptr,
              "the outer scope's context survives the inner scope's pop");
    }

    CUcontext final_ctx = nullptr;
    (void)cuCtxGetCurrent(&final_ctx);
    check(final_ctx == after, "nested scopes unwind to the original binding");
#endif
}

} // namespace

int
main() {
    NIXL_INFO << "=== Testing the libfabric HMEM shim ===";

    // Before anything provokes vendor initialisation, so its log lines are visible in this output.
    nixlAccelInstallLogBridge();

    // Section 0 is compile-time (static_asserts above); nothing to run.
    testIfaceList();
    testIfaceNames();
    testAvailability();
    testQueryPtr();
    testFillMrAttr();
    testDeviceBdfs();
    testBusIdString();
    testPolicy();
    testScope();

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
