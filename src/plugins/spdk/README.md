# SPDK block backend

The SPDK plugin provides local DRAM ↔ BLK_SEG transfers using SPDK bdevs.
It embeds the DPDK-free environment in spdk_env; PCI probing and local PCIe
devices are unsupported. Only statically linked bdev modules are available.

## Build

Build upstream SPDK first. Configure NIXL with:

```sh
meson setup build --reconfigure -Denable_plugins=SPDK -Dbuildtype=debug \
    -Dspdk_include_path=/path/to/spdk/include \
    -Dspdk_library_path=/path/to/spdk/build/lib
ninja -C build
meson test -C build spdk_plugin_memory_test spdk_plugin_test spdk_plugin_runtime_test
```

Both shared and static plugin builds resolve their SPDK archives and transitive
system libraries from this prefix. Use `-Dstatic_plugins=SPDK` for the latter.

## Runtime and registration

The first backend initializes the process-wide SPDK runtime; the last tears it
down. Concurrent backends must configure distinct bdev names. Each backend owns
its SPDK thread and I/O channels. The control plane remains serviced while idle,
including in caller-driven modes; completion stays with the selected execution
mode. Memory and request handles must remain valid until I/O completes.

Applications use ordinary `agent.registerMem()`. DRAM must already be a writable
shared mapping, with 4 KiB-aligned address and length, wholly inside one VMA.
Keep the backing FD open through registration for unprivileged FD discovery.
The backend retains its own descriptor through deregistration; private memory,
including `posix_memalign()` allocations, is rejected. See
[environment documentation](spdk_env/README.md) for ownership and restrictions.

BLK descriptors use `metaInfo` as the bdev name, `addr` as the byte offset and
`len` as the accessible range. Transfers must respect the device block size,
write-unit size and capacity. WRITE moves DRAM to the bdev; READ reverses it.
Transfers use the agent's own name as the remote agent; notifications and
inter-agent connections are not supported.

Configure devices using `json_config` or `json_config_file`, or convenience
`bdev_type`/`bdev_name` parameters. Transport-attached NVMe uses
`bdev_trtype`, `bdev_traddr`, `bdev_subnqn` and applicable address-family/port
fields. Local PCIe attachment is rejected.

For vfio-user, build SPDK with `--with-vfio-user` and pass the socket directory
as the transport address. Servers using physical-address DMA may require
hugetlb-backed shared payloads and `NIXL_SPDK_REQUIRE_HUGEPAGES=1`.
The runtime test accepts `NIXL_SPDK_TEST_VFIO_SOCKET` for an explicit block
transport test; by default it tests a malloc bdev without root or hugepages.
