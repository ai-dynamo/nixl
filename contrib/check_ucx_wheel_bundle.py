#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
import zipfile


def require_matches(names, pattern, description):
    matches = [name for name in names if pattern.search(name)]
    if not matches:
        raise RuntimeError(f"Missing {description}")
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Check that a NIXL wheel contains private UCX libraries and UCX modules."
    )
    parser.add_argument("wheel", help="Path to a repaired NIXL wheel")
    parser.add_argument(
        "--soname-suffix",
        default="nixl",
        help="Expected private UCX SONAME suffix, for example 'nixl'",
    )
    args = parser.parse_args()

    with zipfile.ZipFile(args.wheel) as wheel:
        names = wheel.namelist()

    escaped_suffix = re.escape(args.soname_suffix)
    for lib in ("ucp", "uct", "ucs", "ucm"):
        pattern = re.compile(
            rf"(^|/)[^/]*\.libs/lib{lib}-{escaped_suffix}(-[0-9a-f]{{8}})?\.so"
        )
        matches = require_matches(names, pattern, f"private lib{lib}-{args.soname_suffix}")
        print(f"found private lib{lib}: {matches[0]}")

    ucx_modules = require_matches(
        names,
        re.compile(r"(^|/)[^/]*\.libs/ucx/libuc[stm]_[^/]+\.so"),
        "bundled UCX modules under *.libs/ucx",
    )
    print(f"found {len(ucx_modules)} UCX module(s)")

    nixl_plugins = require_matches(
        names,
        re.compile(r"(^|/)[^/]*\.libs/(?:nixl/|plugins/)?libplugin_UCX\.so$"),
        "bundled NIXL UCX plugin",
    )
    print(f"found NIXL UCX plugin: {nixl_plugins[0]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
