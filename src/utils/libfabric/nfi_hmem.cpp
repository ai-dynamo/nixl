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
 * Vendor table for nfi_hmem.h. Adding an accelerator runtime should mean adding one entry to
 * nfi_vendors[] plus one arm in nfi_hmem_fill_mr_attr(), and touching nothing else in this file or
 * any consumer of it.
 *
 * C++, like the rest of nixl. An earlier revision of this file was plain C so that donating it to
 * libfabric would be a code move rather than a rewrite, but that cost the whole project a second
 * build language for one file, which is a poor trade for a donation that has not happened yet. The
 * shape that matters for donation is the *interface* -- see nfi_hmem.h -- and that is unchanged: no
 * hwloc, no nixl types, no nixl logging, and a C-compatible surface. Only the implementation is
 * idiomatic C++ now.
 *
 * Every vendor runtime is reached through dlopen rather than linked, so one binary runs on hosts
 * with and without each driver. Only the SDK *headers* are a build-time dependency, and each
 * vendor's whole section compiles out without them.
 */

#include "nfi_hmem.h"

#include <dlfcn.h>

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef NIXL_HAVE_CUDA_DRIVER_API
#include <cuda.h>
#endif

#ifdef HAVE_ZE
#include <level_zero/ze_api.h>
#endif

#ifdef NIXL_HAVE_ROCM
#include <hip/hip_runtime_api.h>
#endif

/****************************************
 * Diagnostics
 *****************************************/

static nfi_hmem_log_fn nfi_log_fn;

void
nfi_hmem_set_log_fn(nfi_hmem_log_fn fn)
{
	nfi_log_fn = fn;
}

__attribute__((format(printf, 2, 3))) static void
nfi_log(enum nfi_hmem_log_level level, const char *fmt, ...)
{
	/* Read once: a consumer may install or clear the sink concurrently. */
	const nfi_hmem_log_fn fn = nfi_log_fn;
	if (fn == nullptr) {
		return;
	}

	char buf[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	fn(level, buf);
}

/****************************************
 * Vendor table shape
 *****************************************/

struct nfi_vendor {
	/*
	 * No name field: interface names come from libfabric's own fi_tostr(), which is public for
	 * FI_TYPE_HMEM_IFACE. See nfi_hmem_iface_name().
	 */
	enum fi_hmem_iface iface;

	/*
	 * Once per process, soft-fail, serialised by the caller. nullptr when the vendor SDK was absent
	 * at build time, which makes the runtime permanently unavailable rather than a build break.
	 */
	bool (*init)(void);

	/* false means "not my pointer". Probed in table order, first hit wins. nullptr with init. */
	bool (*query)(const void *addr, struct nfi_hmem_info *info);

	/*
	 * Accelerator PCI addresses the runtime enumerated, for consumers doing their own topology.
	 * nullptr for a vendor that cannot enumerate.
	 */
	size_t (*device_bdfs)(struct nfi_hmem_bus_id *out, size_t max);

	/* Thread-current context management, or nullptr when the runtime has no such state. */
	int (*scope_push)(uint64_t device, uint64_t *saved);
	void (*scope_pop)(uint64_t saved);
};

/****************************************
 * Neuron
 *****************************************/

static int (*nfi_nrt_get_attached_efa_bdf)(const void *va, char *efa_bdf, size_t *len);

static bool
nfi_neuron_init(void)
{
	/* Runs once under the table's init lock, so a plain dlopen here needs no guard of its own. */
	void *handle = dlopen("libnrt.so.1", RTLD_NOW);

	if (!handle) {
		nfi_log(NFI_HMEM_LOG_DEBUG, "libnrt.so.1 not available, Neuron detection disabled");
		return false;
	}

	/*
	 * Gate on the symbol resolving, not just on the library loading. Without this the detector
	 * would probe Neuron on every CUDA or Intel host and log a failure per registration.
	 */
	nfi_nrt_get_attached_efa_bdf = (int (*)(const void *, char *, size_t *))dlsym(
		handle, "nrt_get_attached_efa_bdf");
	if (!nfi_nrt_get_attached_efa_bdf) {
		nfi_log(NFI_HMEM_LOG_WARN,
			"could not resolve libnrt symbol nrt_get_attached_efa_bdf");
		return false;
	}

	return true;
}

static bool
nfi_neuron_query(const void *addr, struct nfi_hmem_info *info)
{
	char buf[] = "0000:00:00.0";
	size_t buflen = sizeof(buf);
	unsigned domain, bus, slot, func;

	if (nfi_nrt_get_attached_efa_bdf(addr, buf, &buflen) != 0) {
		return false;
	}

	info->iface = FI_HMEM_NEURON;
	/*
	 * libfabric requires fi_mr_attr::device to be initialised for Neuron but the EFA provider
	 * never reads it. The -1 sentinel follows the spec and would fault loudly if some future
	 * provider did start reading it.
	 */
	info->device = static_cast<uint64_t>(-1);

	/*
	 * This is the attached EFA device's address, not the accelerator's -- a Trainium card carries
	 * its EFA device onboard, and libnrt offers no accelerator BDF. Reported as NIC so a consumer
	 * keys its lookup off the NIC map rather than an accelerator-to-NIC map.
	 */
	if (sscanf(buf, "%x:%x:%x.%x", &domain, &bus, &slot, &func) == 4) {
		info->bus_kind = NFI_HMEM_BUS_NIC;
		info->bus.domain = static_cast<uint16_t>(domain);
		info->bus.bus = static_cast<uint8_t>(bus);
		info->bus.slot = static_cast<uint8_t>(slot);
		info->bus.func = static_cast<uint8_t>(func);
	}

	return true;
}

/****************************************
 * CUDA
 *****************************************/

#ifdef NIXL_HAVE_CUDA_DRIVER_API

static bool
nfi_cuda_init(void)
{
	if (cuInit(0) != CUDA_SUCCESS) {
		nfi_log(NFI_HMEM_LOG_DEBUG, "cuInit failed, CUDA detection disabled");
		return false;
	}
	return true;
}

static void
nfi_cuda_fill_bus(struct nfi_hmem_info *info, CUdevice dev)
{
	int domain = 0, bus = 0, slot = 0;

	if (cuDeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev) != CUDA_SUCCESS ||
	    cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev) != CUDA_SUCCESS ||
	    cuDeviceGetAttribute(&slot, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev) != CUDA_SUCCESS)
		return;

	info->bus_kind = NFI_HMEM_BUS_ACCELERATOR;
	info->bus.domain = static_cast<uint16_t>(domain);
	info->bus.bus = static_cast<uint8_t>(bus);
	info->bus.slot = static_cast<uint8_t>(slot);
	/* CUDA exposes no function number; a GPU is always function 0. */
	info->bus.func = 0;
}

static bool
nfi_cuda_query(const void *addr, struct nfi_hmem_info *info)
{
	CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
	CUdevice dev = 0;
	CUpointer_attribute attrs[2];
	void *data[2];

	attrs[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
	data[0] = &mem_type;
	attrs[1] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
	data[1] = &dev;

	/*
	 * Works for any device allocation in the process, not only one whose context is current:
	 * unified virtual addressing gives every allocation a distinct range the driver can attribute.
	 * A pointer this does not recognise is host memory, or memory belonging to another vendor.
	 */
	if (cuPointerGetAttributes(2, attrs, data, (CUdeviceptr)addr) != CUDA_SUCCESS) {
		return false;
	}
	if (mem_type != CU_MEMORYTYPE_DEVICE) {
		return false;
	}

	info->iface = FI_HMEM_CUDA;
	info->device = (uint64_t)dev;
	nfi_cuda_fill_bus(info, dev);
	return true;
}

static size_t
nfi_cuda_device_bdfs(struct nfi_hmem_bus_id *out, size_t max)
{
	int count = 0, i;
	size_t found = 0;

	if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
		return 0;
	}

	for (i = 0; i < count; i++) {
		struct nfi_hmem_info info;
		CUdevice dev;

		if (cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
			continue;
		}

		info = {};
		nfi_cuda_fill_bus(&info, dev);
		if (info.bus_kind == NFI_HMEM_BUS_NONE) {
			continue;
		}

		if (out && found < max) {
			out[found] = info.bus;
		}
		found++;
	}

	return found;
}

/*
 * Primary contexts retained by this shim, one slot per CUDA device ordinal, filled on first use and
 * held for the life of the process. Retaining on every push and releasing on every pop put two
 * driver calls, each taking driver-internal locks, on every posted descriptor; the primary context
 * is a per-device singleton, so retaining it once is enough. The retain is never dropped -- the
 * process holds registrations that need the context for as long as it runs, as cudart's own
 * retain does. Ordinals past the table fall back to retain/release per scope.
 *
 * The cache is trusted on the fast path: a push is one cuCtxPushCurrent, with no state query. Only
 * when that push fails is the context asked whether it is still active. An application that reset
 * the device (cudaDeviceReset, cuDevicePrimaryCtxReset) leaves the cache pointing at an inactive
 * context, which retaining again re-creates -- what the old retain-per-push did implicitly -- and the
 * push is retried once. A state query that fails, or a context still active, is a real failure and
 * retains nothing, so a misbehaving driver cannot grow the retain count one push at a time. The
 * stale retain is not released either, since a reset may already have dropped it and releasing
 * could then take someone else's.
 *
 * This relies on pushing a reset primary context failing. If a driver accepted the push and failed
 * the operation instead, that operation would report the error and the cache would not refresh.
 */
#define NFI_CUDA_MAX_CACHED_DEVICES 64
static std::atomic<CUcontext> nfi_cuda_primary[NFI_CUDA_MAX_CACHED_DEVICES];

/*
 * Retained primary context for @p dev, or nullptr on failure; *@p cached says whether pop must not
 * release it. @p stale, when not nullptr, is a cached context found inactive, to be replaced.
 */
static CUcontext
nfi_cuda_primary_ctx(CUdevice dev, bool *cached, CUcontext stale)
{
	const bool cacheable = dev >= 0 && dev < NFI_CUDA_MAX_CACHED_DEVICES;
	*cached = cacheable;

	if (cacheable && !stale) {
		CUcontext ctx = nfi_cuda_primary[dev].load(std::memory_order_acquire);
		if (ctx) {
			return ctx;
		}
	}

	CUcontext primary = nullptr;
	if (cuDevicePrimaryCtxRetain(&primary, dev) != CUDA_SUCCESS) {
		nfi_log(NFI_HMEM_LOG_ERROR, "cuDevicePrimaryCtxRetain failed for CUDA device %d",
			static_cast<int>(dev));
		return nullptr;
	}
	if (!cacheable) {
		return primary;
	}

	/* Two threads can miss at once; the loser drops its extra retain and uses the winner's, which
	   is the same context -- the primary context is a per-device singleton. A refresh replaces the
	   stale entry the same way. */
	CUcontext expected = stale;
	if (!nfi_cuda_primary[dev].compare_exchange_strong(expected, primary,
							   std::memory_order_acq_rel)) {
		(void)cuDevicePrimaryCtxRelease(dev);
		return expected;
	}
	return primary;
}

/* True only when the driver positively reports @p dev's primary context inactive. */
static bool
nfi_cuda_primary_ctx_inactive(CUdevice dev)
{
	unsigned int flags = 0;
	int active = 0;
	return cuDevicePrimaryCtxGetState(dev, &flags, &active) == CUDA_SUCCESS && !active;
}

/*
 * Push/pop the device's primary context. Using the primary context rather than latching one
 * observed on a pointer means there is nothing to discover and no mode to fall out of when a second
 * device appears.
 *
 * Whatever the calling thread had current is restored by the context stack itself: push places the
 * primary context on top of it and pop removes exactly that. *saved records the device, with the
 * top bit set when its retain is cached rather than owned by this scope.
 */
#define NFI_CUDA_SAVED_CACHED (1ULL << 63)

static int
nfi_cuda_scope_push(uint64_t device, uint64_t *saved)
{
	CUdevice dev = (CUdevice)device;
	bool cached = false;
	CUcontext primary = nfi_cuda_primary_ctx(dev, &cached, nullptr);
	if (!primary) {
		return -EIO;
	}

	CUresult ret = cuCtxPushCurrent(primary);
	if (ret != CUDA_SUCCESS && cached && nfi_cuda_primary_ctx_inactive(dev)) {
		nfi_log(NFI_HMEM_LOG_INFO,
			"primary context of CUDA device %d is no longer active (device reset?), "
			"retaining it again", static_cast<int>(dev));
		primary = nfi_cuda_primary_ctx(dev, &cached, primary);
		ret = primary ? cuCtxPushCurrent(primary) : CUDA_ERROR_INVALID_CONTEXT;
	}
	if (ret != CUDA_SUCCESS) {
		if (primary && !cached) {
			(void)cuDevicePrimaryCtxRelease(dev);
		}
		nfi_log(NFI_HMEM_LOG_ERROR, "cuCtxPushCurrent failed for CUDA device %d", static_cast<int>(dev));
		return -EIO;
	}

	*saved = device | (cached ? NFI_CUDA_SAVED_CACHED : 0);
	return 0;
}

static void
nfi_cuda_scope_pop(uint64_t saved)
{
	CUcontext popped = nullptr;

	if (cuCtxPopCurrent(&popped) != CUDA_SUCCESS) {
		nfi_log(NFI_HMEM_LOG_WARN, "cuCtxPopCurrent failed");
	}

	if (saved & NFI_CUDA_SAVED_CACHED) {
		return; /* the cached retain stays for the process */
	}
	if (cuDevicePrimaryCtxRelease(static_cast<CUdevice>(saved)) != CUDA_SUCCESS) {
		nfi_log(NFI_HMEM_LOG_WARN, "cuDevicePrimaryCtxRelease failed for CUDA device %d",
			static_cast<int>(saved));
	}
}

#endif /* NIXL_HAVE_CUDA_DRIVER_API */

/****************************************
 * Level Zero
 *****************************************/

#ifdef HAVE_ZE

/*
 * Spelled out as function pointer types rather than recovered from &zeInit, because taking the
 * address of a ze_api.h declaration would create exactly the link-time dependency on libze_loader
 * that dlopen exists to avoid.
 */
struct nfi_ze_ops {
	ze_result_t (*init)(ze_init_flags_t);
	ze_result_t (*driver_get)(uint32_t *, ze_driver_handle_t *);
	ze_result_t (*device_get)(ze_driver_handle_t, uint32_t *, ze_device_handle_t *);
	ze_result_t (*device_get_sub)(ze_device_handle_t, uint32_t *, ze_device_handle_t *);
	ze_result_t (*context_create)(ze_driver_handle_t, const ze_context_desc_t *,
				      ze_context_handle_t *);
	ze_result_t (*device_pci_props)(ze_device_handle_t, ze_pci_ext_properties_t *);
	ze_result_t (*device_props)(ze_device_handle_t, ze_device_properties_t *);
	ze_result_t (*mem_alloc_props)(ze_context_handle_t, const void *,
				       ze_memory_allocation_properties_t *, ze_device_handle_t *);
};

/**
 * One Level Zero device, in zeDeviceGet() order -- which is what makes the index here equal the
 * ordinal libfabric expects in fi_mr_attr::device.
 */
struct nfi_ze_device {
	ze_device_handle_t handle = nullptr;
	struct nfi_hmem_bus_id bus = {};
	/** Whether @ref bus was readable. Separate from bus being zeroed, which is a valid address. */
	bool has_bus = false;
	/**
	 * Discrete GPU, per the driver's own ZE_DEVICE_PROPERTY_FLAG_INTEGRATED. Only these are
	 * reported by nfi_hmem_device_bdfs(): memory on an integrated GPU registers perfectly well,
	 * it just has no meaningful PCIe distance to a NIC and must not enter proximity grouping.
	 */
	bool discrete = false;
};

static struct nfi_ze_ops nfi_ze;
static ze_context_handle_t nfi_ze_context;
static std::vector<nfi_ze_device> nfi_ze_devices;

static bool
nfi_ze_load(void *handle)
{
	struct {
		const char *name;
		void **slot;
	} syms[] = {
		{"zeInit", (void **)&nfi_ze.init},
		{"zeDriverGet", (void **)&nfi_ze.driver_get},
		{"zeDeviceGet", (void **)&nfi_ze.device_get},
		{"zeDeviceGetSubDevices", (void **)&nfi_ze.device_get_sub},
		{"zeContextCreate", (void **)&nfi_ze.context_create},
		{"zeDevicePciGetPropertiesExt", (void **)&nfi_ze.device_pci_props},
		{"zeDeviceGetProperties", (void **)&nfi_ze.device_props},
		{"zeMemGetAllocProperties", (void **)&nfi_ze.mem_alloc_props},
	};
	for (const auto &sym : syms) {
		*sym.slot = dlsym(handle, sym.name);
		if (*sym.slot == nullptr) {
			nfi_log(NFI_HMEM_LOG_WARN, "could not resolve Level Zero symbol %s", sym.name);
			return false;
		}
	}
	return true;
}

static bool
nfi_ze_init(void)
{
	uint32_t driver_count = 0;
	uint32_t device_count = 0;

	void *handle = dlopen("libze_loader.so.1", RTLD_NOW);
	if (handle == nullptr) {
		nfi_log(NFI_HMEM_LOG_DEBUG,
			"libze_loader.so.1 not available, Level Zero detection disabled");
		return false;
	}

	if (!nfi_ze_load(handle)) {
		return false;
	}

	/*
	 * ZE_INIT_FLAG_GPU_ONLY, not 0, to match libfabric's hmem_ze.c. With flags=0 the loader also
	 * enumerates non-GPU drivers (Intel's NPU ships one), so driver 0 here could be a driver
	 * libfabric never looked at, and the ordinal correspondence below would be silently wrong.
	 */
	if (nfi_ze.init(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) {
		nfi_log(NFI_HMEM_LOG_DEBUG, "zeInit failed, Level Zero detection disabled");
		return false;
	}

	if (nfi_ze.driver_get(&driver_count, nullptr) != ZE_RESULT_SUCCESS || driver_count == 0) {
		nfi_log(NFI_HMEM_LOG_DEBUG, "no Level Zero drivers found");
		return false;
	}
	std::vector<ze_driver_handle_t> drivers(driver_count);
	if (nfi_ze.driver_get(&driver_count, drivers.data()) != ZE_RESULT_SUCCESS) {
		nfi_log(NFI_HMEM_LOG_WARN, "zeDriverGet failed to retrieve driver handles");
		return false;
	}

	/*
	 * Driver 0 only: libfabric's hmem_ze.c holds a single static driver handle and asserts
	 * !ze_get_driver_idx(device), so a device on any other driver has no representable ordinal.
	 */
	if (driver_count > 1) {
		nfi_log(NFI_HMEM_LOG_DEBUG,
			"found %u Level Zero drivers, using driver 0 only (libfabric is single-driver)",
			static_cast<unsigned>(driver_count));
	}
	ze_driver_handle_t driver = drivers[0];

	if (nfi_ze.device_get(driver, &device_count, nullptr) != ZE_RESULT_SUCCESS ||
	    device_count == 0) {
		nfi_log(NFI_HMEM_LOG_DEBUG, "no Level Zero devices on driver 0");
		return false;
	}
	std::vector<ze_device_handle_t> handles(device_count);
	if (nfi_ze.device_get(driver, &device_count, handles.data()) != ZE_RESULT_SUCCESS) {
		nfi_log(NFI_HMEM_LOG_WARN, "zeDeviceGet failed to retrieve device handles");
		return false;
	}
	ze_context_desc_t ctx_desc = {};
	ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
	if (nfi_ze.context_create(driver, &ctx_desc, &nfi_ze_context) != ZE_RESULT_SUCCESS) {
		nfi_log(NFI_HMEM_LOG_WARN, "zeContextCreate failed, Level Zero detection disabled");
		nfi_ze_context = nullptr;
		return false;
	}

	nfi_ze_devices.reserve(handles.size());
	for (size_t i = 0; i < handles.size(); i++) {
		nfi_ze_device dev;
		dev.handle = handles[i];

		ze_pci_ext_properties_t pci = {};
		pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
		if (nfi_ze.device_pci_props(dev.handle, &pci) == ZE_RESULT_SUCCESS) {
			dev.bus.domain = static_cast<uint16_t>(pci.address.domain);
			dev.bus.bus = static_cast<uint8_t>(pci.address.bus);
			dev.bus.slot = static_cast<uint8_t>(pci.address.device);
			dev.bus.func = static_cast<uint8_t>(pci.address.function);
			dev.has_bus = true;
		} else {
			nfi_log(NFI_HMEM_LOG_DEBUG,
				"zeDevicePciGetPropertiesExt failed for Level Zero device %u",
				static_cast<unsigned>(i));
		}

		/*
		 * Asking the driver whether a device is integrated is what replaces a PCI class check.
		 * A class check cannot answer it -- a discrete Intel GPU and Intel integrated graphics
		 * both report 0x0300 -- and it wrongly drops compute-only parts, which carry no display
		 * class at all.
		 */
		ze_device_properties_t props = {};
		props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
		if (nfi_ze.device_props(dev.handle, &props) == ZE_RESULT_SUCCESS) {
			dev.discrete = (props.type == ZE_DEVICE_TYPE_GPU) &&
				((props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) == 0);
		} else {
			nfi_log(NFI_HMEM_LOG_DEBUG,
				"zeDeviceGetProperties failed for Level Zero device %u, not treating it as discrete",
				static_cast<unsigned>(i));
		}

		nfi_ze_devices.push_back(dev);
	}

	nfi_log(NFI_HMEM_LOG_INFO, "Level Zero detection enabled with %u device(s) on driver 0",
		static_cast<unsigned>(nfi_ze_devices.size()));
	return true;
}

/*
 * zeMemGetAllocProperties can hand back a sub-device handle for memory allocated against a tile,
 * while zeDeviceGet only ever returns root devices. libfabric hits the same case and lets it fall
 * through to a no-op assert, silently reporting ordinal 0; resolving to the root gives the correct
 * ordinal.
 */
static int
nfi_ze_device_index(ze_device_handle_t handle)
{
	for (size_t i = 0; i < nfi_ze_devices.size(); i++) {
		if (nfi_ze_devices[i].handle == handle) {
			return static_cast<int>(i);
		}
	}

	for (size_t i = 0; i < nfi_ze_devices.size(); i++) {
		uint32_t sub_count = 0;
		if (nfi_ze.device_get_sub(nfi_ze_devices[i].handle, &sub_count, nullptr) !=
			    ZE_RESULT_SUCCESS ||
		    sub_count == 0) {
			continue;
		}

		std::vector<ze_device_handle_t> subs(sub_count);
		if (nfi_ze.device_get_sub(nfi_ze_devices[i].handle, &sub_count, subs.data()) !=
		    ZE_RESULT_SUCCESS) {
			continue;
		}

		for (uint32_t sub = 0; sub < sub_count; sub++) {
			if (subs[sub] == handle) {
				nfi_log(NFI_HMEM_LOG_DEBUG,
					"resolved Level Zero sub-device %u to root device %u",
					static_cast<unsigned>(sub), static_cast<unsigned>(i));
				return static_cast<int>(i);
			}
		}
	}

	return -1;
}

static bool
nfi_ze_query(const void *addr, struct nfi_hmem_info *info)
{
	ze_memory_allocation_properties_t props = {};
	ze_device_handle_t device = nullptr;
	int index = 0;

	props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;

	if (nfi_ze.mem_alloc_props(nfi_ze_context, addr, &props, &device) != ZE_RESULT_SUCCESS) {
		return false;
	}

	/*
	 * Matches ze_hmem_is_addr_valid(), which rejects only this case: DEVICE, SHARED and HOST all
	 * need FI_HMEM_ZE registration.
	 */
	if (props.type == ZE_MEMORY_TYPE_UNKNOWN) {
		return false;
	}

	if (device == nullptr) {
		/* ZE_MEMORY_TYPE_HOST belongs to no device, so there is no ordinal or address. */
		nfi_log(NFI_HMEM_LOG_DEBUG,
			"Level Zero allocation %p has no owning device (type %d), reporting ordinal 0",
			addr, static_cast<int>(props.type));
	} else {
		index = nfi_ze_device_index(device);
		if (index < 0) {
			/*
			 * Should not happen now that sub-devices resolve to their root, but a
			 * mis-ordinaled registration is recoverable where a misattributed iface is
			 * not, so claim the pointer anyway.
			 */
			nfi_log(NFI_HMEM_LOG_WARN,
				"Level Zero device for %p is not in the driver 0 device list, using ordinal 0",
				addr);
			index = 0;
		} else if (nfi_ze_devices[index].has_bus) {
			info->bus_kind = NFI_HMEM_BUS_ACCELERATOR;
			info->bus = nfi_ze_devices[index].bus;
		}
	}

	info->iface = FI_HMEM_ZE;
	/* Driver index is 0: libfabric's Level Zero support is single-driver. */
	info->device = static_cast<uint64_t>(fi_hmem_ze_device(0, index));
	return true;
}

static size_t
nfi_ze_device_bdfs(struct nfi_hmem_bus_id *out, size_t max)
{
	size_t found = 0;

	for (const nfi_ze_device &dev : nfi_ze_devices) {
		if (!dev.discrete || !dev.has_bus) {
			continue;
		}
		if (out != nullptr && found < max) {
			out[found] = dev.bus;
		}
		found++;
	}

	return found;
}

#endif /* HAVE_ZE */

/****************************************
 * AMD ROCr (HIP)
 *****************************************/

#ifdef NIXL_HAVE_ROCM

/*
 * libamdhip64 is dlopen'd like libze_loader and libnrt, so a ROCm build still runs on a host
 * without the HIP runtime. The function pointer types are spelled out rather than recovered from
 * &hipGetDeviceCount, which would bring back the link-time dependency.
 *
 * No scope_push/scope_pop: libfabric's ROCr support works through HSA (hsa_amd_pointer_info,
 * hsa_amd_memory_lock), which keeps no thread-current device, so there is nothing to bind around
 * registration or posting -- the same position as Level Zero.
 */
struct nfi_rocr_ops {
	hipError_t (*get_device_count)(int *);
	hipError_t (*pointer_get_attributes)(hipPointerAttribute_t *, const void *);
	hipError_t (*device_get_pci_bus_id)(char *, int, int);
};

static struct nfi_rocr_ops nfi_rocr;

static bool
nfi_rocr_init(void)
{
	void *handle = dlopen("libamdhip64.so", RTLD_NOW);
	if (handle == nullptr) {
		nfi_log(NFI_HMEM_LOG_DEBUG, "libamdhip64.so not available, ROCr detection disabled");
		return false;
	}

	struct {
		const char *name;
		void **slot;
	} syms[] = {
		{"hipGetDeviceCount", (void **)&nfi_rocr.get_device_count},
		{"hipPointerGetAttributes", (void **)&nfi_rocr.pointer_get_attributes},
		{"hipDeviceGetPCIBusId", (void **)&nfi_rocr.device_get_pci_bus_id},
	};
	for (const auto &sym : syms) {
		*sym.slot = dlsym(handle, sym.name);
		if (*sym.slot == nullptr) {
			nfi_log(NFI_HMEM_LOG_WARN, "could not resolve HIP symbol %s", sym.name);
			return false;
		}
	}

	int count = 0;
	if (nfi_rocr.get_device_count(&count) != hipSuccess || count <= 0) {
		nfi_log(NFI_HMEM_LOG_DEBUG, "no HIP devices found, ROCr detection disabled");
		return false;
	}
	nfi_log(NFI_HMEM_LOG_INFO, "ROCr detection enabled with %d device(s)", count);
	return true;
}

/* Fill @p info's bus from HIP @p dev's PCI address; left NFI_HMEM_BUS_NONE when unreadable. */
static void
nfi_rocr_fill_bus(struct nfi_hmem_info *info, int dev)
{
	char buf[32];
	unsigned domain, bus, slot, func;

	if (nfi_rocr.device_get_pci_bus_id(buf, sizeof(buf), dev) != hipSuccess) {
		return;
	}
	if (sscanf(buf, "%x:%x:%x.%x", &domain, &bus, &slot, &func) == 4) {
		info->bus_kind = NFI_HMEM_BUS_ACCELERATOR;
		info->bus.domain = static_cast<uint16_t>(domain);
		info->bus.bus = static_cast<uint8_t>(bus);
		info->bus.slot = static_cast<uint8_t>(slot);
		info->bus.func = static_cast<uint8_t>(func);
	}
}

static bool
nfi_rocr_query(const void *addr, struct nfi_hmem_info *info)
{
	hipPointerAttribute_t attr = {};

	/*
	 * Like cuPointerGetAttributes(), works for any allocation in the process. Device memory only,
	 * matching what the plugin accepted for AMD before this shim: a pointer this does not
	 * recognise is host memory, or memory belonging to another vendor.
	 */
	if (nfi_rocr.pointer_get_attributes(&attr, addr) != hipSuccess) {
		return false;
	}
#if HIP_VERSION_MAJOR >= 6
	const hipMemoryType mem_type = attr.type;
#else
	const hipMemoryType mem_type = attr.memoryType;
#endif
	if (mem_type != hipMemoryTypeDevice) {
		return false;
	}

	info->iface = FI_HMEM_ROCR;
	/* The HIP device index, which is what fi_mr_attr::device.rocr carries. */
	info->device = static_cast<uint64_t>(attr.device);
	nfi_rocr_fill_bus(info, attr.device);
	return true;
}

static size_t
nfi_rocr_device_bdfs(struct nfi_hmem_bus_id *out, size_t max)
{
	int count = 0;
	size_t found = 0;

	if (nfi_rocr.get_device_count(&count) != hipSuccess) {
		return 0;
	}
	for (int i = 0; i < count; i++) {
		struct nfi_hmem_info info = {};
		nfi_rocr_fill_bus(&info, i);
		if (info.bus_kind == NFI_HMEM_BUS_NONE) {
			continue;
		}
		if (out && found < max) {
			out[found] = info.bus;
		}
		found++;
	}
	return found;
}

#endif /* NIXL_HAVE_ROCM */

/****************************************
 * Table
 *****************************************/

/*
 * Probe order. Descending fi_hmem_iface, matching libfabric's ofi_get_hmem_iface(), which iterates
 * hmem_ops[] from the highest interface down. Several providers -- cxi, rxm, opx -- call that
 * function for themselves, and rxm is what verbs runs under, so a different order here could make
 * this file and the provider disagree about the same pointer on a host with two runtimes.
 */
static const struct nfi_vendor nfi_vendors[] = {
	{
		FI_HMEM_NEURON,
		nfi_neuron_init,
		nfi_neuron_query,
		nullptr, /* libnrt offers no device enumeration */
		nullptr, /* no thread-current state */
		nullptr,
	},
#ifdef HAVE_ZE
	{
		FI_HMEM_ZE,
		nfi_ze_init,
		nfi_ze_query,
		nfi_ze_device_bdfs,
		nullptr, /* Level Zero contexts are not thread-current */
		nullptr,
	},
#else
	{FI_HMEM_ZE, nullptr, nullptr, nullptr, nullptr, nullptr},
#endif
#ifdef NIXL_HAVE_ROCM
	{
		FI_HMEM_ROCR,
		nfi_rocr_init,
		nfi_rocr_query,
		nfi_rocr_device_bdfs,
		nullptr, /* HSA keeps no thread-current state */
		nullptr,
	},
#else
	{FI_HMEM_ROCR, nullptr, nullptr, nullptr, nullptr, nullptr},
#endif
#ifdef NIXL_HAVE_CUDA_DRIVER_API
	{
		FI_HMEM_CUDA,
		nfi_cuda_init,
		nfi_cuda_query,
		nfi_cuda_device_bdfs,
		nfi_cuda_scope_push,
		nfi_cuda_scope_pop,
	},
#else
	{FI_HMEM_CUDA, nullptr, nullptr, nullptr, nullptr, nullptr},
#endif
};

#define NFI_NUM_VENDORS (sizeof(nfi_vendors) / sizeof(nfi_vendors[0]))

/*
 * One-time init state, parallel to nfi_vendors[]. One shared mutex rather than a std::once_flag per
 * vendor: the flags would have to be a parallel array kept in step with the table by hand, which is
 * exactly the kind of thing that silently breaks when a vendor is added.
 *
 * The mutex serialises init only. Once a vendor's state is settled it never changes, so every later
 * call -- one per posted descriptor, from the post thread pool and the progress threads at once --
 * reads it with a single acquire load and takes no lock.
 */
enum nfi_vendor_state : unsigned char {
	NFI_VENDOR_UNTRIED = 0,
	NFI_VENDOR_OK,
	NFI_VENDOR_UNAVAILABLE,
};

static std::mutex nfi_vendor_lock;
static std::atomic<unsigned char> nfi_vendor_state[NFI_NUM_VENDORS];

static bool
nfi_vendor_available(size_t index)
{
	if (index >= NFI_NUM_VENDORS) {
		return false;
	}

	unsigned char state = nfi_vendor_state[index].load(std::memory_order_acquire);
	if (state != NFI_VENDOR_UNTRIED) {
		return state == NFI_VENDOR_OK;
	}

	const std::lock_guard<std::mutex> lock(nfi_vendor_lock);
	state = nfi_vendor_state[index].load(std::memory_order_relaxed);
	if (state == NFI_VENDOR_UNTRIED) {
		/*
		 * A nullptr init or query means the SDK was absent at build time, which makes the
		 * runtime permanently unavailable rather than a build break.
		 */
		const bool ok = nfi_vendors[index].init && nfi_vendors[index].query &&
			nfi_vendors[index].init();
		state = ok ? NFI_VENDOR_OK : NFI_VENDOR_UNAVAILABLE;
		nfi_vendor_state[index].store(state, std::memory_order_release);
	}
	return state == NFI_VENDOR_OK;
}

static size_t
nfi_vendor_index(enum fi_hmem_iface iface)
{
	for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
		if (nfi_vendors[i].iface == iface) {
			return i;
		}
	}
	return NFI_NUM_VENDORS;
}

/****************************************
 * Public interface
 *****************************************/

int
nfi_hmem_query_ptr(const void *addr, struct nfi_hmem_info *info)
{
	if (!addr || !info) {
		return -EINVAL;
	}

	*info = {};
	info->iface = FI_HMEM_SYSTEM;

	for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
		if (!nfi_vendor_available(i)) {
			continue;
		}
		if (nfi_vendors[i].query(addr, info)) {
			nfi_log(NFI_HMEM_LOG_DEBUG, "%s claims %p (device %llu)",
				nfi_hmem_iface_name(nfi_vendors[i].iface), addr,
				(unsigned long long)info->device);
			return 0;
		}
		/* A failed probe must not leave partial output behind for the next one. */
		*info = {};
		info->iface = FI_HMEM_SYSTEM;
	}

	nfi_log(NFI_HMEM_LOG_DEBUG, "no accelerator runtime claimed %p, treating as system memory",
		addr);
	return 0;
}

void
nfi_hmem_fill_mr_attr(const struct nfi_hmem_info *info, struct fi_mr_attr *attr)
{
	if (!info || !attr) {
		return;
	}

	attr->iface = info->iface;

	switch (info->iface) {
	case FI_HMEM_CUDA:
		attr->device.cuda = static_cast<int>(info->device);
		break;
	case FI_HMEM_ZE:
		/* Already encoded as fi_hmem_ze_device(driver, device) by the detector. */
		attr->device.ze = static_cast<int>(info->device);
		break;
	case FI_HMEM_NEURON:
		attr->device.neuron = static_cast<int>(info->device);
		break;
	case FI_HMEM_ROCR:
		/*
		 * The device.rocr arm arrived in libfabric 2.3. Older libfabric has no such member, and
		 * the EFA provider ignores the device for ROCr, so the zero-initialised union stands in.
		 */
#if FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION) >= FI_VERSION(2, 3)
		attr->device.rocr = static_cast<int>(info->device);
#endif
		break;
	case FI_HMEM_SYSTEM:
		/* Leaving device unset is what libfabric expects for host memory. */
		break;
	default:
		nfi_log(NFI_HMEM_LOG_WARN,
			"no fi_mr_attr::device arm for hmem iface %d, registering with device unset",
			static_cast<int>(info->iface));
		break;
	}
}

bool
nfi_hmem_available(void)
{
	for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
		if (nfi_vendor_available(i)) {
			return true;
		}
	}
	return false;
}

bool
nfi_hmem_iface_available(enum fi_hmem_iface iface)
{
	return nfi_vendor_available(nfi_vendor_index(iface));
}

enum fi_hmem_iface
nfi_hmem_primary_iface(void)
{
	for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
		if (nfi_vendor_available(i)) {
			return nfi_vendors[i].iface;
		}
	}
	return FI_HMEM_SYSTEM;
}

const enum fi_hmem_iface *
nfi_hmem_ifaces(size_t *count)
{
	static enum fi_hmem_iface ifaces[NFI_NUM_VENDORS];
	/*
	 * Rewritten on every call rather than guarded by a once-flag: the values are compile-time
	 * constants copied out of a const table, so concurrent writers store identical bytes and
	 * there is nothing for a reader to observe torn.
	 */
	for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
		ifaces[i] = nfi_vendors[i].iface;
	}

	if (count) {
		*count = NFI_NUM_VENDORS;
	}
	return ifaces;
}

/*
 * Names come from libfabric rather than a table of our own. FI_TYPE_HMEM_IFACE is public, so a
 * second list here would only be something that can drift -- and libfabric's covers FI_HMEM_ROCR and
 * FI_HMEM_SYNAPSEAI, which this file has no vendor entry for but can still be asked to name.
 *
 * The "FI_HMEM_" prefix is stripped: everything named here is an hmem interface, and repeating that
 * in a line like "CUDA=0 NEURON=0 ZE=1" only makes it longer.
 *
 * Cached because the public signature returns a borrowed pointer while fi_tostr_r() wants a caller
 * buffer. The whole table is filled once, so a reader can never observe a half-written slot.
 */
#define NFI_IFACE_NAME_SLOTS 16
#define NFI_IFACE_NAME_MAX 32

static char nfi_iface_names[NFI_IFACE_NAME_SLOTS][NFI_IFACE_NAME_MAX];
static std::once_flag nfi_iface_names_once;

static void
nfi_init_iface_names(void)
{
	static const char prefix[] = "FI_HMEM_";
	const size_t prefix_len = sizeof(prefix) - 1;
	int slot;

	for (slot = 0; slot < NFI_IFACE_NAME_SLOTS; slot++) {
		enum fi_hmem_iface iface = (enum fi_hmem_iface)slot;
		char buf[NFI_IFACE_NAME_MAX];
		const char *name = buf;

		fi_tostr_r(buf, sizeof(buf), &iface, FI_TYPE_HMEM_IFACE);
		if (!strncmp(name, prefix, prefix_len)) {
			name += prefix_len;
		}

		/* snprintf, not strcpy: a longer name in a future libfabric must truncate, not overrun. */
		snprintf(nfi_iface_names[slot], NFI_IFACE_NAME_MAX, "%s", name);
	}
}

const char *
nfi_hmem_iface_name(enum fi_hmem_iface iface)
{
	const int slot = static_cast<int>(iface);

	std::call_once(nfi_iface_names_once, nfi_init_iface_names);

	if (slot < 0 || slot >= NFI_IFACE_NAME_SLOTS) {
		return "Unknown";
	}
	return nfi_iface_names[slot];
}

size_t
nfi_hmem_device_bdfs(enum fi_hmem_iface iface, struct nfi_hmem_bus_id *out, size_t max)
{
	size_t index = nfi_vendor_index(iface);

	if (index >= NFI_NUM_VENDORS || !nfi_vendors[index].device_bdfs) {
		return 0;
	}
	if (!nfi_vendor_available(index)) {
		return 0;
	}

	return nfi_vendors[index].device_bdfs(out, out ? max : 0);
}

bool
nfi_hmem_needs_scope(enum fi_hmem_iface iface)
{
	size_t index = nfi_vendor_index(iface);

	if (index >= NFI_NUM_VENDORS || !nfi_vendors[index].scope_push) {
		return false;
	}
	return nfi_vendor_available(index);
}

int
nfi_hmem_scope_push(enum fi_hmem_iface iface, uint64_t device, struct nfi_hmem_scope *scope)
{
	size_t index;

	if (!scope) {
		return -EINVAL;
	}

	scope->iface = FI_HMEM_SYSTEM;
	scope->saved = 0;
	scope->pushed = 0;

	index = nfi_vendor_index(iface);
	if (index >= NFI_NUM_VENDORS || !nfi_vendors[index].scope_push) {
		return 0;
	}
	if (!nfi_vendor_available(index)) {
		return 0;
	}

	int ret = nfi_vendors[index].scope_push(device, &scope->saved);
	if (ret) {
		return ret;
	}

	scope->iface = iface;
	scope->pushed = 1;
	return 0;
}

void
nfi_hmem_scope_pop(struct nfi_hmem_scope *scope)
{
	size_t index;

	if (!scope || !scope->pushed) {
		return;
	}

	index = nfi_vendor_index(scope->iface);
	if (index < NFI_NUM_VENDORS && nfi_vendors[index].scope_pop) {
		nfi_vendors[index].scope_pop(scope->saved);
	}

	scope->pushed = 0;
}
