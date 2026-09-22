#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import subprocess
import sys

CHILD_SCRIPT = r"""
import ctypes
import os
import sys

preload = os.environ.get("CHECK_UCX_PRELOAD")
library = os.environ["CHECK_UCX_LIBRARY"]
deepbind = os.environ.get("CHECK_UCX_DEEPBIND") == "1"
call_plugin_init = os.environ.get("CHECK_UCX_CALL_PLUGIN_INIT") == "1"

if preload:
    ctypes.CDLL(preload, mode=os.RTLD_NOW | os.RTLD_GLOBAL)

mode = os.RTLD_NOW | os.RTLD_LOCAL
if deepbind:
    mode |= getattr(os, "RTLD_DEEPBIND", 0x8)

lib = ctypes.CDLL(library, mode=mode)
if call_plugin_init:
    init = lib.nixl_plugin_init
    init.restype = ctypes.c_void_p
    plugin = init()
    if not plugin:
        raise RuntimeError("nixl_plugin_init returned null")
    try:
        fini = lib.nixl_plugin_fini
    except AttributeError:
        fini = None
    if fini is not None:
        fini()

sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""


def run_checked(cmd, **kwargs):
    completed = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout, completed.stderr


def check_needed(library, expected_needed):
    if not expected_needed:
        return

    stdout, _ = run_checked(["readelf", "-d", library])
    for needed in expected_needed:
        needle = f"Shared library: [{needed}]"
        if needle not in stdout:
            raise RuntimeError(f"{library} does not NEEDED {needed}")
        print(f"needed ok: {needed}")


def parse_expectations(values):
    expectations = {}
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"expected SYMBOL=TARGET_SUBSTRING, got {value!r}")
        symbol, target = value.split("=", 1)
        if not symbol or not target:
            raise RuntimeError(f"expected SYMBOL=TARGET_SUBSTRING, got {value!r}")
        expectations[symbol] = target
    return expectations


def find_bindings(debug_output, source_name):
    binding_re = re.compile(
        r"binding file (?P<source>.+?) \[[^\]]+\] to (?P<target>.+?) "
        r"\[[^\]]+\]: .* symbol [`'](?P<symbol>[^`']+)[`']"
    )
    bindings = {}
    for line in debug_output.splitlines():
        match = binding_re.search(line)
        if not match:
            continue
        if source_name not in match.group("source"):
            continue
        bindings[match.group("symbol")] = match.group("target")
    return bindings


def check_bindings(args):
    expectations = parse_expectations(args.expect_binding)
    if args.expected_soname and not args.call_nixl_plugin_init:
        raise RuntimeError("--expected-soname requires --call-nixl-plugin-init")

    if not expectations and not args.call_nixl_plugin_init:
        return

    env = os.environ.copy()
    if expectations:
        env["LD_DEBUG"] = "bindings"
    env["CHECK_UCX_LIBRARY"] = args.library
    env["CHECK_UCX_DEEPBIND"] = "1" if args.deepbind else "0"
    if args.preload_ucx:
        env["CHECK_UCX_PRELOAD"] = args.preload_ucx
    if args.call_nixl_plugin_init:
        env["CHECK_UCX_CALL_PLUGIN_INIT"] = "1"
    if args.expected_soname:
        env["NIXL_UCX_EXPECTED_SONAME"] = args.expected_soname

    completed = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        text=True,
        capture_output=True,
        env=env,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"binding probe failed with exit code {completed.returncode}\n{output}"
        )

    if expectations:
        source_name = os.path.basename(args.library)
        bindings = find_bindings(output, source_name)
        for symbol, expected_target in expectations.items():
            target = bindings.get(symbol)
            if target is None:
                raise RuntimeError(
                    f"no LD_DEBUG binding found for {source_name}:{symbol}"
                )
            if expected_target not in target:
                raise RuntimeError(
                    f"{source_name}:{symbol} bound to {target}, expected {expected_target}"
                )
            print(f"binding ok: {symbol} -> {target}")


def main():
    parser = argparse.ArgumentParser(
        description="Check UCX dynamic linkage and glibc symbol binding for a NIXL artifact."
    )
    parser.add_argument("--library", required=True, help="Library or plugin to dlopen")
    parser.add_argument(
        "--preload-ucx",
        help="External libucp.so.0 to load with RTLD_GLOBAL before loading --library",
    )
    parser.add_argument(
        "--deepbind",
        action="store_true",
        help="Load --library with RTLD_DEEPBIND",
    )
    parser.add_argument(
        "--call-nixl-plugin-init",
        action="store_true",
        help="Call nixl_plugin_init after dlopen; useful for libplugin_UCX.so",
    )
    parser.add_argument(
        "--expected-soname",
        help="Set NIXL_UCX_EXPECTED_SONAME while calling nixl_plugin_init",
    )
    parser.add_argument(
        "--expect-needed",
        action="append",
        default=[],
        help="Require a DT_NEEDED entry, for example libucp-nixl.so.0",
    )
    parser.add_argument(
        "--expect-binding",
        action="append",
        default=[],
        help="Require SYMBOL to bind to a target substring, for example ucp_init_version=libucp-nixl",
    )
    args = parser.parse_args()

    check_needed(args.library, args.expect_needed)
    check_bindings(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
