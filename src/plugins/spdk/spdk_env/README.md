# DPDK-free SPDK environment

This private library implements SPDK's environment interface without DPDK or
local PCI device support. DMA allocations are FD-backed shared mappings so
vfio-user can export them to another process. Memory maps publish backing FDs
before notifying consumers and retain them through unregister notifications.

Application memory must already be writable, shared, page-aligned and contained
in one VMA. The private registration helper discovers and retains its backing
FD; it never replaces memory or copies data. Keep the original FD open for
unprivileged discovery when procfs map_files access is unavailable. Mapping
lifetime must be serialized against registration, I/O and deregistration.

Private mappings are rejected. Local PCI DMA is unsupported: physical-address
translation fails rather than presenting virtual addresses as physical ones.
Set `NIXL_SPDK_REQUIRE_HUGEPAGES=1` when the consuming vfio-user server requires
hugepage-backed allocations. See LICENSE.spdk-env for adapted-code attribution.

With a built upstream SPDK tree, run the independent memory/FD lifetime test:

```sh
bash test/unit/plugins/spdk/test_env.sh /path/to/spdk
```

The plugin adds Meson integration and runtime tests separately.
