# C++ Examples

This directory contains C++ examples showing how to use NIXL agents and perform transfers.

Available examples:
- `nixl_example.cpp`: Basic agent usage and transfers
- `nixl_etcd_example.cpp`: Metadata exchange example using etcd
- `telemetry_reader.cpp`: Reading transfer telemetry
- `local_vram_staging_probe.cpp`: CUDA shared-mmap staging probe for single-node no-P2P GPU pairs

Run:
- Binaries are generated under `build/examples/` (depending on your Meson setup).
- See comments in each example source file for invocation details.
- Example local staging probe:
  `local_vram_staging_probe --mode single --source-gpu 0 --target-gpu 1 --bytes 256M --iters 4`
