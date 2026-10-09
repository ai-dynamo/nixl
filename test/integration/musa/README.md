# MUSA_UCX Hardware Smoke Test

Requires a real MUSA runtime and a UCX provider advertising memory type `musa`.
Missing hardware/provider is a **failure**, never a successful skip.
This local two-process test does not certify cross-node RDMA or GPU-direct operation.

Configure with the already documented SDK/UCX/plugin options and additionally
`-Dmusa_hardware_tests=true`. Then:

```bash
meson compile -C build-musa musa_ucx_e2e
meson test -C build-musa --suite musa-hardware --print-errorlogs
```

The registered test runs device/device READ and WRITE with a completion notification.
Each process uses device 0 and has a 90-second watchdog.

To extend the matrix, use the same built binary. Ensure `NIXL_PLUGIN_DIR` includes the
directory containing `libplugin_MUSA_UCX.so` (Meson sets it for the registered test):

```bash
build-musa/test/unit/plugins/musa/musa_ucx_e2e host device 65536 0 0 0
build-musa/test/unit/plugins/musa/musa_ucx_e2e device host 65536 2 0 0
build-musa/test/unit/plugins/musa/musa_ucx_e2e device device 1048576 2 0 1
```

Arguments: memory A, memory B, byte count, UCX post-thread count, device A, device B.
The control channel uses a local socketpair; NIXL transports the actual registered
payload. Both processes initialize SDK/UCX only after fork and keep buffers alive
until transfers finish and registrations are removed.

The test uses background progress. Caller-only progress, repeated reuse, fault
injection, and cross-node coverage remain separate items in
[the hardware acceptance matrix](../../../docs/musa_hardware_acceptance.md).
