<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

All notable changes to the NIXL service (marshal-based compression utilities) are documented in this file.

## [0.1.0] - 2026-08-10

Initial experimental release of the NIXL service layer on top of `nixlAgent`, with pluggable marshal backends including nvCOMP-based compression.

### Added

- Marshalled **WRITE** and **READ** transfers with transparent GPU compression and decompression via nvCOMP (`nixlMarshalCompressConfig`), plus pass-through mode (`nixlMarshalDirectConfig`).
- Python bindings for the service API and an example (`examples/python/service_api_example.py`) covering marshalled WRITE (default) and READ (`--direction read`).

## [0.2.0] - 2026-09-17

Version 0.2 release including a new mode, memory and performance optimizations, and bug fixes.

### Added

- Stored Compressed Data: We’ve added support for storing data in its compressed form, featuring an on-demand "pull and decompress" execution mode. This is exposed through a per-transfer service phase configuration (Pre-Transfer / Post-Transfer / Pre-And-Post-Transfer):
  - WRITE + Pre-Transfer: Compresses the data, writes it to the destination buffer, and keeps it compressed.
  - READ + Post-Transfer: Reads the compressed data and decompresses it into the destination buffer (decompression is handled on the reader’s side).
  - Try the new mode: python3 examples/python/service_storage_example.py --memory-type vram|dram|file
- Memory Footprint Optimization: We significantly reduced the memory footprint. The previous requirement of 1 GB per concurrent transfer has been optimized to a ~700 MB baseline, plus ~350 MB for each additional concurrent transfer.
  - Base footprint (no concurrent transfers): ~700 MB
  - 2 concurrent transfers: ~1 GB
  - 3 concurrent transfers: ~1.35 GB
- Datatype Input for Compression: You can now provide the input datatype as a per-transfer parameter to ensure optimal compression.
- NIXL Bench Integration: nixlServiceAgent is now integrated into NIXL Bench (note: functionality is currently limited).