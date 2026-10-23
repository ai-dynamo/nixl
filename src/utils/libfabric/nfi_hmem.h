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

/**
 * @file nfi_hmem.h
 * @brief Heterogeneous-memory helpers that libfabric has internally but does not export.
 *
 * libfabric requires callers of fi_mr_regattr() to declare fi_mr_attr::iface and the correct arm of
 * the fi_mr_attr::device union. It knows perfectly well how to derive both from a pointer --
 * ofi_get_hmem_iface() in src/hmem.c does exactly that, and prov/hook/hook_hmem even auto-fills
 * fi_mr_attr from it -- but neither is reachable by an application: include/ofi_hmem.h is not
 * installed and libfabric.so exports only the fi_* API. Every consumer that wants to register
 * device memory therefore reimplements libfabric's vendor probing. This is that reimplementation,
 * factored so it can eventually be deleted.
 *
 * Three deliberate constraints on the *interface*, so that moving it into libfabric later is a
 * contained change rather than a redesign:
 *
 *   1. A C-compatible surface -- POD structs, plain functions, `extern "C"` -- with no dependency
 *      beyond libfabric's own public headers, dlopen and the vendor SDK headers. In particular no
 *      hwloc: PCI addresses come back as integers and the caller formats them. The *implementation*
 *      is ordinary C++ like the rest of nixl; an earlier revision was plain C, but that cost the
 *      whole project a second build language for one file, which is not a trade worth making for a
 *      donation that has not happened yet.
 *   2. Nothing nixl-specific. No nixl types, no nixl logging, no nixl error codes. Diagnostics go
 *      through a caller-installed callback (@ref nfi_hmem_set_log_fn); inside libfabric that
 *      becomes FI_WARN.
 *   3. The `nfi_` prefix is deliberately *not* `fi_`. `fi_*` is libfabric's public namespace, and
 *      defining a symbol there would collide the day libfabric adds its own. Renaming at donation
 *      time is mechanical.
 *
 * What is intentionally *not* here, because libfabric would not want it:
 *
 *   - hwloc predicates ("is this hwloc object an accelerator"). Use @ref nfi_hmem_device_bdfs and
 *     match against your own topology.
 *   - policy. Whether a vendor's devices participate in NIC proximity grouping, or whether an
 *     unattributable device pointer is a caller error, are consumer decisions.
 *
 * @section threading Threading
 *
 * All functions are safe to call concurrently. Vendor initialisation happens once per process, on
 * first use, and is serialised internally. The @ref nfi_hmem_scope calls operate on thread-local
 * accelerator state and must be paired on the same thread.
 *
 * @section perf Fast path vs slow path
 *
 * @ref nfi_hmem_query_ptr talks to the vendor runtime and is a slow path -- call it once per
 * registration and keep the result. @ref nfi_hmem_scope_push is a fast path and takes the already
 * known (iface, device), so it never re-probes. Do not reach for query_ptr in a per-descriptor loop.
 */

#ifndef NFI_HMEM_H
#define NFI_HMEM_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/****************************************
 * Types
 *****************************************/

/** @brief A PCI address, unformatted. Callers render it in whatever notation they key on. */
struct nfi_hmem_bus_id {
	uint16_t domain;
	uint8_t bus;
	uint8_t slot;
	uint8_t func;
};

/**
 * @brief Whose PCI address @ref nfi_hmem_info::bus holds.
 *
 * Not decoration: the answer is vendor-dependent and a consumer that assumes "the accelerator"
 * will silently mis-key its topology lookups. A CUDA or Level Zero pointer reports the GPU's own
 * address; a Neuron pointer reports the address of the NIC attached to that accelerator, because
 * that is what libnrt offers and what is actually useful -- a Trainium card carries its EFA device
 * onboard.
 */
enum nfi_hmem_bus_kind {
	/** No PCI address available. @ref nfi_hmem_info::bus is zeroed. */
	NFI_HMEM_BUS_NONE = 0,
	/** The accelerator's own PCI address. */
	NFI_HMEM_BUS_ACCELERATOR,
	/** The PCI address of a NIC attached to the accelerator. */
	NFI_HMEM_BUS_NIC
};

/** @brief What the vendor runtimes know about one pointer. */
struct nfi_hmem_info {
	/**
	 * Detected interface. FI_HMEM_SYSTEM means "no vendor claimed it", which for a pointer the
	 * caller believes to be device memory is informative rather than an error -- see
	 * @ref nfi_hmem_query_ptr.
	 */
	enum fi_hmem_iface iface;

	/**
	 * Interface-specific device ordinal, already encoded for the fi_mr_attr::device arm that
	 * @p iface selects. Opaque: a CUdevice for CUDA, fi_hmem_ze_device(0, index) for Level Zero,
	 * the -1 sentinel for Neuron. Never treat this as an index into the caller's own device
	 * numbering.
	 */
	uint64_t device;

	/** Which device @ref bus describes. */
	enum nfi_hmem_bus_kind bus_kind;

	/** PCI address, valid only when @ref bus_kind is not NFI_HMEM_BUS_NONE. */
	struct nfi_hmem_bus_id bus;
};

/**
 * @brief A pushed accelerator context, to be popped on the same thread.
 *
 * Opaque in practice; declared here so callers can put one on the stack. Zero-initialise before
 * the first push.
 */
struct nfi_hmem_scope {
	enum fi_hmem_iface iface;
	uint64_t saved;
	int pushed;
};

/** @brief Diagnostic severity, matching the usual four levels. */
enum nfi_hmem_log_level {
	NFI_HMEM_LOG_ERROR = 0,
	NFI_HMEM_LOG_WARN,
	NFI_HMEM_LOG_INFO,
	NFI_HMEM_LOG_DEBUG
};

/**
 * @brief Diagnostic sink. @p msg is a NUL-terminated single line, not owned by the callee.
 *
 * Keeps this file free of any particular logging framework. May be called from any thread.
 */
typedef void (*nfi_hmem_log_fn)(enum nfi_hmem_log_level level, const char *msg);

/****************************************
 * Setup
 *****************************************/

/**
 * @brief Installs the diagnostic sink. Pass NULL to silence output, which is the default.
 *
 * Call before any other function here if you want to see initialisation messages, since vendor
 * setup happens on first use and is the most interesting thing to log.
 */
void
nfi_hmem_set_log_fn(nfi_hmem_log_fn fn);

/****************************************
 * Detection
 *****************************************/

/**
 * @brief Identifies which accelerator runtime owns @p addr.
 *
 * Probes each usable vendor and stops at the first that claims the pointer -- the same
 * first-hit-wins loop as libfabric's ofi_get_hmem_iface(), and in the same order, so that this and
 * a provider deriving the iface for itself cannot reach different conclusions about one pointer.
 * That matters in practice: several providers (cxi, rxm, opx) call ofi_get_hmem_iface() internally,
 * and rxm is what verbs runs under.
 *
 * Host memory is reported as FI_HMEM_SYSTEM with a success return; it is not an error. A caller
 * registering something it believes is device memory should treat FI_HMEM_SYSTEM as "no runtime
 * recognised this" and apply its own policy -- for some vendors that is a caller bug, for others a
 * reason to fall back.
 *
 * Slow path. Call once per registration and keep @p info.
 *
 * @param addr Pointer to identify.
 * @param[out] info Detected interface, ordinal and PCI address. Fully overwritten.
 * @return 0 on success, -EINVAL if @p addr or @p info is NULL.
 */
int
nfi_hmem_query_ptr(const void *addr, struct nfi_hmem_info *info);

/**
 * @brief Populates fi_mr_attr::iface and the fi_mr_attr::device arm that @p info selects.
 *
 * The reason this file exists. libfabric's ABI makes fi_mr_attr::device a union whose applicable
 * member is chosen by iface at compile time, so it cannot be written generically by a caller --
 * every consumer ends up with the same switch statement. This is that switch, once.
 *
 * Only iface and device are touched; access, requested_key, offset, context, auth_key and the iov
 * remain the caller's business.
 *
 * An interface whose provider ignores the device value still has it initialised, because libfabric
 * requires that. FI_HMEM_SYSTEM leaves device alone, which is what libfabric expects for host
 * memory.
 *
 * @param info Result of @ref nfi_hmem_query_ptr, or any (iface, device) pair the caller trusts.
 * @param[in,out] attr Attribute block to fill. Must be non-NULL.
 */
void
nfi_hmem_fill_mr_attr(const struct nfi_hmem_info *info, struct fi_mr_attr *attr);

/****************************************
 * Enablement
 *****************************************/

/**
 * @brief Whether any accelerator runtime is usable in this process.
 *
 * The question "should I advertise device-memory support" turns on. Deliberately not "which vendor
 * is this host": a host can have two, and @ref nfi_hmem_query_ptr decides per pointer.
 *
 * Answers the vendor runtimes directly, so it is independent of which libfabric provider was
 * selected and of whether that provider advertises FI_HMEM. Check the provider separately.
 */
bool
nfi_hmem_available(void);

/**
 * @brief Whether one specific runtime is usable in this process.
 *
 * Forces that vendor's one-time initialisation on first call.
 */
bool
nfi_hmem_iface_available(enum fi_hmem_iface iface);

/**
 * @brief First usable runtime in probe order, or FI_HMEM_SYSTEM if there is none.
 *
 * For logging, and for a caller that needs a single answer to "what kind of machine is this" --
 * a last-resort fallback when a device pointer could not be attributed. Not a substitute for
 * per-pointer detection: on a host with two runtimes it names only one.
 */
enum fi_hmem_iface
nfi_hmem_primary_iface(void);

/**
 * @brief Every interface this file knows about, in probe order.
 *
 * Lets a consumer iterate vendors instead of naming them. Appearing here says nothing about
 * whether the runtime is usable; ask @ref nfi_hmem_iface_available for that.
 *
 * @param[out] count Number of entries. Must be non-NULL.
 * @return Pointer to a static array of @p count entries. Never NULL, never freed.
 */
const enum fi_hmem_iface *
nfi_hmem_ifaces(size_t *count);

/**
 * @brief Short name of @p iface ("CUDA", "ZE", "NEURON", "SYSTEM", ...). For diagnostics.
 *
 * Derived from libfabric's own fi_tostr(FI_TYPE_HMEM_IFACE) with the redundant "FI_HMEM_" prefix
 * stripped, not from a list maintained here -- so it cannot drift from libfabric, and it names every
 * interface libfabric knows rather than only the ones this file has a vendor entry for.
 * FI_HMEM_ROCR and FI_HMEM_SYNAPSEAI therefore come back correctly named, not as "Unknown".
 *
 * @return A borrowed, NUL-terminated string valid for the process lifetime. Never NULL.
 *         "Unknown" for a value libfabric does not recognise.
 */
const char *
nfi_hmem_iface_name(enum fi_hmem_iface iface);

/****************************************
 * Device inventory
 *****************************************/

/**
 * @brief PCI addresses of the accelerators @p iface's runtime enumerated.
 *
 * For a consumer doing its own topology work: this is the authoritative device list for a vendor
 * whose hardware cannot be identified from PCI attributes alone. Level Zero is the case that
 * exists -- a discrete Intel GPU and Intel integrated graphics share vendor ID 0x8086 and PCI
 * class 0x0300, so no class check separates them, while compute-only parts carry no display class
 * at all and a class check drops them. The runtime knows; ask it.
 *
 * Only devices usable as RDMA accelerators are reported. For Level Zero that means discrete GPUs:
 * an integrated GPU is excluded here even though memory on it registers perfectly well, because a
 * device on the CPU package has no meaningful PCIe distance to a NIC.
 *
 * Not every vendor has such a list. An absent or unusable runtime reports zero, which a caller
 * should read as "I cannot help you identify this vendor's hardware", not as "there is none".
 *
 * Forces the vendor's one-time initialisation on first call, which matters: the natural caller
 * runs during topology discovery, before any pointer has been probed.
 *
 * @param iface Vendor to enumerate.
 * @param[out] out Destination, or NULL to query the count only.
 * @param max Capacity of @p out in entries; ignored when @p out is NULL.
 * @return Total number of devices the runtime enumerated, which may exceed @p max. At most @p max
 *         entries are written.
 */
size_t
nfi_hmem_device_bdfs(enum fi_hmem_iface iface, struct nfi_hmem_bus_id *out, size_t max);

/****************************************
 * Thread context binding
 *****************************************/

/**
 * @brief Whether @p iface keeps thread-current state that must be established before use.
 *
 * True for CUDA, whose context currency is per-thread and absent on a freshly created thread, so
 * driver calls made by libfabric on a consumer-owned progress thread would fail. False for Level
 * Zero and Neuron, and for FI_HMEM_SYSTEM. When false, the scope calls below are no-ops and a
 * caller can skip them entirely.
 *
 * Note libfabric itself does no context management on the data path -- its only cuCtxCreate is a
 * throwaway context in EFA's start-up p2p probe -- so this obligation lands on whoever calls it.
 * UCX solves the same problem inside uct_cuda_ctx_*; this is the equivalent.
 */
bool
nfi_hmem_needs_scope(enum fi_hmem_iface iface);

/**
 * @brief Makes @p device of @p iface current on the calling thread until the matching pop.
 *
 * Push/pop rather than set: the caller's thread may be an application thread that already has its
 * own context, and leaving that thread bound to somebody else's device afterwards is a visible side
 * effect -- the application's next unqualified accelerator call would run on the wrong device.
 * @ref nfi_hmem_scope_pop restores exactly what was current before.
 *
 * Binds the device's *primary* context rather than any context the consumer may have observed on a
 * pointer. That is deliberate: a primary context always exists, needs no discovery, and cannot go
 * stale, so there is no mode to fall out of when a second device shows up.
 *
 * Cheap and repeatable -- safe in a per-descriptor loop. A no-op when
 * @ref nfi_hmem_needs_scope is false for @p iface, in which case @p scope records nothing and the
 * matching pop is also a no-op.
 *
 * Fast path: takes an already known (iface, device) and never probes a pointer.
 *
 * @param iface Interface owning @p device.
 * @param device Ordinal as reported in @ref nfi_hmem_info::device.
 * @param[out] scope Receives the state @ref nfi_hmem_scope_pop needs. Must be non-NULL.
 * @return 0 on success, -EINVAL if @p scope is NULL, or a negative errno if the runtime refused.
 *         On failure @p scope is left safe to pop.
 */
int
nfi_hmem_scope_push(enum fi_hmem_iface iface, uint64_t device, struct nfi_hmem_scope *scope);

/**
 * @brief Undoes @ref nfi_hmem_scope_push, restoring the thread's previous binding.
 *
 * Safe to call on a scope whose push failed, and safe to call twice. Must run on the thread that
 * pushed.
 */
void
nfi_hmem_scope_pop(struct nfi_hmem_scope *scope);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NFI_HMEM_H */
