# NULL Plugin

## Overview

The NULL plugin is a minimal backend implementation that completes all transfer operations instantly without performing any actual data transfer. Similar to `/dev/null` in Linux, this plugin discards all data and immediately reports success.

## Purpose

This plugin is designed for performance testing and benchmarking the NIXL library overhead without the influence of actual I/O operations. It allows you to:

- Measure pure library overhead
- Test the plugin infrastructure
- Validate the control flow and API usage
- Benchmark maximum theoretical throughput

## Supported Features

- **Memory Types**: All memory types (DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG)
- **Operations**: Read and Write (both complete instantly)
- **Local Operations**: Supported
- **Remote Operations**: Not supported
- **Notifications**: Not supported

