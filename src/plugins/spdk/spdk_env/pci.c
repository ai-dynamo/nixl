/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "env_internal.h"
#include "spdk/log.h"
#include "spdk_env_compat.h"
#include "spdk/string.h"



struct spdk_pci_driver {
	const char *name;
	struct spdk_pci_id *id_table;
	uint32_t flags;
	TAILQ_ENTRY(spdk_pci_driver) tailq;
};

static TAILQ_HEAD(, spdk_pci_driver) g_pci_drivers = TAILQ_HEAD_INITIALIZER(g_pci_drivers);
static TAILQ_HEAD(, spdk_pci_device) g_pci_devices = TAILQ_HEAD_INITIALIZER(g_pci_devices);
static TAILQ_HEAD(, spdk_pci_device_provider) g_pci_device_providers = TAILQ_HEAD_INITIALIZER(g_pci_device_providers);

int pci_env_init(void)
{
	return 0;
}

void pci_env_reinit(void)
{
}

void pci_env_fini(void)
{
}

void spdk_pci_driver_register(const char *name, struct spdk_pci_id *id_table, uint32_t flags)
{
	struct spdk_pci_driver *driver;

	driver = calloc(1, sizeof(*driver));
	if (!driver) {
		/* Without this the driver simply goes missing and the caller sees
		 * an unrelated "no such driver" failure later.
		 */
		ENV_ERRLOG("Failed to allocate PCI driver %s", name);
		return;
	}

	driver->name = name;
	driver->id_table = id_table;
	driver->flags = flags;
	TAILQ_INSERT_TAIL(&g_pci_drivers, driver, tailq);
}

struct spdk_pci_driver *spdk_pci_get_driver(const char *name)
{
	struct spdk_pci_driver *driver;

	TAILQ_FOREACH(driver, &g_pci_drivers, tailq)
	{
		if (strcmp(driver->name, name) == 0) {
			return driver;
		}
	}

	return NULL;
}

struct spdk_pci_driver *spdk_pci_nvme_get_driver(void)
{
	return spdk_pci_get_driver("nvme");
}

int spdk_pci_enumerate(struct spdk_pci_driver *driver, spdk_pci_enum_cb enum_cb, void *enum_ctx)
{
	return -ENOTSUP;
}

void spdk_pci_for_each_device(void *ctx, void (*fn)(void *ctx, struct spdk_pci_device *dev))
{
	struct spdk_pci_device *dev;

	TAILQ_FOREACH(dev, &g_pci_devices, internal.tailq)
	{
		fn(ctx, dev);
	}
}

int spdk_pci_device_map_bar(struct spdk_pci_device *dev,
			    uint32_t bar,
			    void **mapped_addr,
			    uint64_t *phys_addr,
			    uint64_t *size)
{
	if (dev->map_bar) {
		return dev->map_bar(dev, bar, mapped_addr, phys_addr, size);
	}
	return -ENOTSUP;
}

int spdk_pci_device_unmap_bar(struct spdk_pci_device *dev, uint32_t bar, void *mapped_addr)
{
	if (dev->unmap_bar) {
		return dev->unmap_bar(dev, bar, mapped_addr);
	}
	return -ENOTSUP;
}

int spdk_pci_device_enable_interrupt(struct spdk_pci_device *dev)
{
	return -ENOTSUP;
}

int spdk_pci_device_disable_interrupt(struct spdk_pci_device *dev)
{
	return -ENOTSUP;
}

int spdk_pci_device_get_interrupt_efd(struct spdk_pci_device *dev)
{
	return -ENOTSUP;
}

int spdk_pci_device_enable_interrupts(struct spdk_pci_device *dev, uint32_t efd_count)
{
	return -ENOTSUP;
}

int spdk_pci_device_disable_interrupts(struct spdk_pci_device *dev)
{
	return -ENOTSUP;
}

int spdk_pci_device_get_interrupt_efd_by_index(struct spdk_pci_device *dev, uint32_t index)
{
	return -ENOTSUP;
}

uint32_t spdk_pci_device_get_domain(struct spdk_pci_device *dev)
{
	return dev->addr.domain;
}

uint8_t spdk_pci_device_get_bus(struct spdk_pci_device *dev)
{
	return dev->addr.bus;
}

uint8_t spdk_pci_device_get_dev(struct spdk_pci_device *dev)
{
	return dev->addr.dev;
}

uint8_t spdk_pci_device_get_func(struct spdk_pci_device *dev)
{
	return dev->addr.func;
}

struct spdk_pci_addr spdk_pci_device_get_addr(struct spdk_pci_device *dev)
{
	return dev->addr;
}

uint16_t spdk_pci_device_get_vendor_id(struct spdk_pci_device *dev)
{
	return dev->id.vendor_id;
}

uint16_t spdk_pci_device_get_device_id(struct spdk_pci_device *dev)
{
	return dev->id.device_id;
}

uint16_t spdk_pci_device_get_subvendor_id(struct spdk_pci_device *dev)
{
	return dev->id.subvendor_id;
}

uint16_t spdk_pci_device_get_subdevice_id(struct spdk_pci_device *dev)
{
	return dev->id.subdevice_id;
}

struct spdk_pci_id spdk_pci_device_get_id(struct spdk_pci_device *dev)
{
	return dev->id;
}

int spdk_pci_device_get_numa_id(struct spdk_pci_device *dev)
{
	return dev->numa_id;
}

int spdk_pci_device_get_serial_number(struct spdk_pci_device *dev, char *sn, size_t len)
{
	return -ENOTSUP;
}

int spdk_pci_device_claim(struct spdk_pci_device *dev)
{
	return -ENOTSUP;
}

void spdk_pci_device_unclaim(struct spdk_pci_device *dev)
{
}

void spdk_pci_device_detach(struct spdk_pci_device *device)
{
}

int spdk_pci_device_attach(struct spdk_pci_driver *driver,
			   spdk_pci_enum_cb enum_cb,
			   void *enum_ctx,
			   struct spdk_pci_addr *pci_address)
{
	return -ENOTSUP;
}

int spdk_pci_device_allow(struct spdk_pci_addr *pci_addr)
{
	return -ENOTSUP;
}

int spdk_pci_device_cfg_read(struct spdk_pci_device *dev, void *buf, uint32_t len, uint32_t offset)
{
	if (dev->cfg_read) {
		return dev->cfg_read(dev, buf, len, offset);
	}
	return -ENOTSUP;
}

int spdk_pci_device_cfg_write(struct spdk_pci_device *dev, void *buf, uint32_t len, uint32_t offset)
{
	if (dev->cfg_write) {
		return dev->cfg_write(dev, buf, len, offset);
	}
	return -ENOTSUP;
}

int spdk_pci_device_cfg_read8(struct spdk_pci_device *dev, uint8_t *value, uint32_t offset)
{
	return spdk_pci_device_cfg_read(dev, value, 1, offset);
}

int spdk_pci_device_cfg_write8(struct spdk_pci_device *dev, uint8_t value, uint32_t offset)
{
	return spdk_pci_device_cfg_write(dev, &value, 1, offset);
}

int spdk_pci_device_cfg_read16(struct spdk_pci_device *dev, uint16_t *value, uint32_t offset)
{
	return spdk_pci_device_cfg_read(dev, value, 2, offset);
}

int spdk_pci_device_cfg_write16(struct spdk_pci_device *dev, uint16_t value, uint32_t offset)
{
	return spdk_pci_device_cfg_write(dev, &value, 2, offset);
}

int spdk_pci_device_cfg_read32(struct spdk_pci_device *dev, uint32_t *value, uint32_t offset)
{
	return spdk_pci_device_cfg_read(dev, value, 4, offset);
}

int spdk_pci_device_cfg_write32(struct spdk_pci_device *dev, uint32_t value, uint32_t offset)
{
	return spdk_pci_device_cfg_write(dev, &value, 4, offset);
}

bool spdk_pci_device_is_removed(struct spdk_pci_device *dev)
{
	return dev->internal.pending_removal;
}

int spdk_pci_addr_compare(const struct spdk_pci_addr *a1, const struct spdk_pci_addr *a2)
{
	if (a1->domain > a2->domain) {
		return 1;
	} else if (a1->domain < a2->domain) {
		return -1;
	} else if (a1->bus > a2->bus) {
		return 1;
	} else if (a1->bus < a2->bus) {
		return -1;
	} else if (a1->dev > a2->dev) {
		return 1;
	} else if (a1->dev < a2->dev) {
		return -1;
	} else if (a1->func > a2->func) {
		return 1;
	} else if (a1->func < a2->func) {
		return -1;
	}
	return 0;
}

int spdk_pci_addr_parse(struct spdk_pci_addr *addr, const char *bdf)
{
	unsigned domain, bus, dev, func;
	size_t len;
	int n = -1;

	if (addr == NULL || bdf == NULL) {
		return -EINVAL;
	}

	len = strlen(bdf);

	/* %n records how much each pattern consumed. sscanf() ignores whatever
	 * follows, so without this "0000:00:04.0garbage" would parse cleanly and
	 * select a device the caller never named.
	 */
	if (strchr(bdf, ':') != NULL) {
		if (sscanf(bdf, "%x:%x:%x.%x%n", &domain, &bus, &dev, &func, &n) == 4 &&
		    n >= 0 && (size_t)n == len) {
			goto ok;
		}
		n = -1;
		if (sscanf(bdf, "%x:%x.%x%n", &bus, &dev, &func, &n) == 3 &&
		    n >= 0 && (size_t)n == len) {
			domain = 0;
			goto ok;
		}
		return -EINVAL;
	}

	if (sscanf(bdf, "%x.%x.%x.%x%n", &domain, &bus, &dev, &func, &n) == 4 &&
	    n >= 0 && (size_t)n == len) {
		goto ok;
	}

	return -EINVAL;

ok:
	/* The fields are narrower than unsigned, so an out-of-range value would
	 * otherwise be silently truncated into a valid-looking address.
	 */
	if (bus > 0xFF || dev > 0x1F || func > 0x7) {
		return -EINVAL;
	}

	addr->domain = domain;
	addr->bus = (uint8_t)bus;
	addr->dev = (uint8_t)dev;
	addr->func = (uint8_t)func;
	return 0;
}

int spdk_pci_addr_fmt(char *bdf, size_t sz, const struct spdk_pci_addr *addr)
{
	int rc;

	rc = snprintf(bdf, sz, "%04x:%02x:%02x.%x", addr->domain, addr->bus, addr->dev, addr->func);
	if (rc < 0 || (size_t)rc >= sz) {
		return -EINVAL;
	}
	return 0;
}

/* Hooked devices go on g_pci_devices, the same list spdk_pci_for_each_device()
 * walks, so a hooked device is visible to enumeration. A separate list would
 * not work anyway: both would thread through the same internal.tailq link.
 */
int spdk_pci_hook_device(struct spdk_pci_driver *drv, struct spdk_pci_device *dev)
{
	dev->internal.driver = drv;
	dev->internal.attached = false;
	TAILQ_INSERT_TAIL(&g_pci_devices, dev, internal.tailq);
	return 0;
}

void spdk_pci_unhook_device(struct spdk_pci_device *dev)
{
	TAILQ_REMOVE(&g_pci_devices, dev, internal.tailq);
}

const char *spdk_pci_device_get_type(const struct spdk_pci_device *dev)
{
	return dev->type;
}

void spdk_pci_register_device_provider(struct spdk_pci_device_provider *provider)
{
	TAILQ_INSERT_TAIL(&g_pci_device_providers, provider, tailq);
}

struct spdk_pci_driver *spdk_pci_vmd_get_driver(void)
{
	return spdk_pci_get_driver("vmd");
}

struct spdk_pci_driver *spdk_pci_idxd_get_driver(void)
{
	return spdk_pci_get_driver("idxd");
}

struct spdk_pci_driver *spdk_pci_ioat_get_driver(void)
{
	return spdk_pci_get_driver("ioat");
}

struct spdk_pci_driver *spdk_pci_ae4dma_get_driver(void)
{
	return spdk_pci_get_driver("ae4dma");
}

struct spdk_pci_driver *spdk_pci_virtio_get_driver(void)
{
	return spdk_pci_get_driver("virtio");
}

int spdk_pci_event_listen(void)
{
	return -ENOTSUP;
}

int spdk_pci_get_event(int fd, struct spdk_pci_event *event)
{
	return -ENOTSUP;
}

int spdk_pci_register_error_handler(spdk_pci_error_handler sighandler, void *ctx)
{
	return -ENOTSUP;
}

void spdk_pci_unregister_error_handler(spdk_pci_error_handler sighandler)
{
}
