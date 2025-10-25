#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import csv
import hashlib
import logging
import os
import shutil
import tempfile
import zipfile

logger = logging.getLogger(__name__)


def extract_wheel(wheel_path):
    """
    Extract the wheel to a temporary directory.
    Returns:
        Path to the temporary directory. The caller is responsible for cleaning up the directory.
    """
    temp_dir = tempfile.mkdtemp()
    logger.info("Extracting wheel %s to %s", wheel_path, temp_dir)
    with zipfile.ZipFile(wheel_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir


def update_wheel_record_file(temp_dir):
    """
    Update the RECORD file in the wheel to include the hashes and sizes of all files.
    """
    dist_info_dir = None
    for entry in os.listdir(temp_dir):
        if entry.endswith(".dist-info"):
            dist_info_dir = entry
            break
    if dist_info_dir is None:
        raise RuntimeError("No .dist-info directory found in wheel")

    record_path = os.path.join(temp_dir, dist_info_dir, "RECORD")

    def hash_and_size(file_path):
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        digest = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode("ascii")
        size = os.path.getsize(file_path)
        return f"sha256={digest}", str(size)

    entries = []
    for root, _, files in os.walk(temp_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, temp_dir).replace(os.sep, "/")
            if rel_path == f"{dist_info_dir}/RECORD":
                # RECORD file itself: no hash or size
                entries.append((rel_path, "", ""))
            else:
                file_hash, file_size = hash_and_size(full_path)
                entries.append((rel_path, file_hash, file_size))

    with open(record_path, "w", newline="") as rec_file:
        writer = csv.writer(rec_file)
        writer.writerows(entries)


def create_wheel(wheel_path, temp_dir):
    """
    Create a wheel from a temporary directory.
    """
    logger.info("Creating wheel %s from %s", wheel_path, temp_dir)
    update_wheel_record_file(temp_dir)
    with zipfile.ZipFile(
        wheel_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zip_ref:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, start=temp_dir)
                zip_ref.write(abs_path, arcname=rel_path)


def get_repaired_lib_name_map(libs_dir):
    """
    auditwheel repair renames all libs to include a hash of the library name
    e.g. "nixl.libs/libboost_atomic-fb1368c6.so.1.66.0"
    Extract mapping from base name (like "libboost_atomic") to full file name
    (like "libboost_atomic-fb1368c6.so.1.66.0").
    """
    name_map = {}
    for fname in sorted(os.listdir(libs_dir)):
        if (
            os.path.isfile(os.path.join(libs_dir, fname))
            and ".so" in fname
            and "-" in fname
        ):
            base_name = fname.split("-")[0]
            name_map[base_name] = fname
            print(f"Found already bundled lib: {base_name} -> {fname}")
    return name_map


def get_lib_deps(lib_path):
    """
    Get the dependencies of a library, as a map from library name to path.

    Example:

    ldd nixl.libs/ucx/libuct_cma.so
        linux-vdso.so.1 (0x00007fff007f3000)
        libuct.so.0 => /lib64/libuct.so.0 (0x00007f8683abb000)
        libucs.so.0 => /lib64/libucs.so.0 (0x00007f8683a37000)
        libm.so.6 => /lib64/libm.so.6 (0x00007f8683571000)
        libucm.so.0 => /lib64/libucm.so.0 (0x00007f8683a18000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007f868336d000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f868314d000)
        librt.so.1 => /lib64/librt.so.1 (0x00007f8682f45000)
        libc.so.6 => /lib64/libc.so.6 (0x00007f8682b6e000)
        libucs-311e600f.so.0.0.0 => /workspace/nixl/dist/xuz/nixl.libs/ucx/../libucs-311e600f.so.0.0.0 (0x00007f868398b000)
        libucm-e091ff91.so.0.0.0 => /workspace/nixl/dist/xuz/nixl.libs/ucx/../libucm-e091ff91.so.0.0.0 (0x00007f868396a000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f86838f3000)

    Returns:
        {
            "libuct.so.0": "/lib64/libuct.so.0",
            "libucs.so.0": "/lib64/libucs.so.0",
            ...
        }
    """
    deps = os.popen(f"ldd {lib_path}").read().strip().split("\n")
    ret = {}
    for dep in deps:
        if "=>" in dep:
            left, right = dep.split("=>", 1)
            dep_name = left.strip()
            right = right.strip()
            if right == "not found":
                ret[dep_name] = None
            else:
                # Remove address " (0x00007f8683abb000)"
                dep_path = right.split(" ")[0].strip()
                ret[dep_name] = dep_path
    return ret


def copytree(src, dst):
    """
    Copy a tree of files from @src directory to @dst directory.
    Similar to shutil.copytree, but returns a list of all files copied.
    Returns:
        List of files copied.
    """
    copied_files = []
    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dst_dir = os.path.join(dst, rel_path)
        os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)
            copied_files.append(dst_file)
    return copied_files


def add_plugins(wheel_path, sys_plugins_dir, install_dirname):
    """
    Adds the plugins from @sys_dir to the wheel.
    The plugins are copied to a subdirectory @install_dir relative to the wheel's nixl.libs.
    The plugins are patched to load their dependencies from the wheel.
    The wheel file is then recreated.
    """
    temp_dir = extract_wheel(wheel_path)

    pkg_name = wheel_path.split("/")[-1].split("-")[0]
    pkg_libs_dir = os.path.join(temp_dir, f"{pkg_name}.libs")
    if not os.path.exists(pkg_libs_dir):
        raise FileNotFoundError(
            f"{pkg_name}.libs directory not found in wheel: {wheel_path}"
        )

    logger.debug("Listing existing libs:")
    name_map = get_repaired_lib_name_map(pkg_libs_dir)

    # Ensure that all of them in name_map have RPATH set to $ORIGIN
    for fname in name_map.values():
        fpath = os.path.join(pkg_libs_dir, fname)
        rpath = os.popen(f"patchelf --print-rpath {fpath}").read().strip()
        if "$ORIGIN" in rpath.split(":"):
            continue
        if not rpath:
            rpath = "$ORIGIN"
        else:
            rpath = "$ORIGIN:" + rpath
        logger.debug("Setting rpath for %s to %s", fpath, rpath)
        ret = os.system(f"patchelf --set-rpath '{rpath}' {fpath}")
        if ret != 0:
            raise RuntimeError(f"Failed to set rpath for {fpath}")

    pkg_plugins_dir = os.path.join(pkg_libs_dir, install_dirname)
    logger.debug("Copying plugins from %s to %s", sys_plugins_dir, pkg_plugins_dir)
    copied_files = copytree(sys_plugins_dir, pkg_plugins_dir)
    if not copied_files:
        raise RuntimeError(f"No plugins found in {sys_plugins_dir}")

    # Patch all libs to load plugin deps from the wheel
    for fname in copied_files:
        logger.debug("Patching %s", fname)
        fpath = os.path.join(pkg_plugins_dir, fname)
        if os.path.isfile(fpath) and ".so" in fname:
            rpath = os.popen(f"patchelf --print-rpath {fpath}").read().strip()
            if not rpath:
                rpath = "$ORIGIN/..:$ORIGIN"
            else:
                rpath = "$ORIGIN/..:$ORIGIN:" + rpath
            logger.debug("Setting rpath for %s to %s", fpath, rpath)
            ret = os.system(f"patchelf --set-rpath '{rpath}' {fpath}")
            if ret != 0:
                raise RuntimeError(f"Failed to set rpath for {fpath}")
            # Replace the original libs with the patched one
            for libname, _ in get_lib_deps(fpath).items():
                # "libuct.so.0" -> "libuct"
                base_name = libname.split(".")[0]
                if base_name in name_map:
                    packaged_name = name_map[base_name]
                    logger.debug(
                        "Replacing %s with %s in %s", libname, packaged_name, fpath
                    )
                    ret = os.system(
                        f"patchelf --replace-needed {libname} {packaged_name} {fpath}"
                    )
                    if ret != 0:
                        raise RuntimeError(
                            f"Failed to replace {libname} with {packaged_name} in {fpath}"
                        )
            # Check that there is no breakage introduced in the patched lib
            logger.debug("Checking that %s loads", fpath)
            original_deps = get_lib_deps(os.path.join(sys_plugins_dir, fname))
            for libname, libpath in get_lib_deps(fpath).items():
                if libpath is None:
                    if (
                        libname not in original_deps
                        or original_deps[libname] is not None
                    ):
                        raise RuntimeError(f"Library {libname} not loaded by {fpath}")

    create_wheel(wheel_path, temp_dir)
    shutil.rmtree(temp_dir)
    logger.info("Added plugins to wheel: %s", wheel_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ucx-plugins-dir",
        type=str,
        help="Path to the UCX plugins directory",
        default="/usr/lib64/ucx",
    )
    parser.add_argument(
        "--nixl-plugins-dir",
        type=str,
        help="Path to the NIXL plugins directory",
        default="/usr/local/nixl/lib/$ARCH-linux-gnu/plugins",
    )
    parser.add_argument(
        "wheel", type=str, nargs="+", help="Path to one or more wheel files"
    )
    args = parser.parse_args()
    if "$ARCH" in args.nixl_plugins_dir:
        arch = os.getenv("ARCH", os.uname().machine)
        args.nixl_plugins_dir = args.nixl_plugins_dir.replace("$ARCH", arch)

    for wheel_path in args.wheel:
        add_plugins(wheel_path, args.ucx_plugins_dir, "ucx")
        add_plugins(wheel_path, args.nixl_plugins_dir, "nixl")


if __name__ == "__main__":
    main()
