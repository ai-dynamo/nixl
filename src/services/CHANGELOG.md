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