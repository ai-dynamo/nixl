# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared helpers for the wheel_inspect tests: synthetic wheels and ELF files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Union

SYMLINK_ATTR = 0o120777 << 16

MemberSpec = Union[bytes, tuple[str, str]]


def make_wheel(path: Path, members: dict[str, MemberSpec]) -> Path:
    """Write a zip at ``path``; a ``("symlink", target)`` value becomes a symlink member."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, spec in members.items():
            if isinstance(spec, tuple):
                info = zipfile.ZipInfo(name)
                info.external_attr = SYMLINK_ATTR
                zf.writestr(info, spec[1])
            else:
                zf.writestr(name, spec)
    return path


def make_elf(
    needed: list[str],
    runpath: str | None,
    sections: dict[str, bytes] | None = None,
    build_id: bytes | None = None,
) -> bytes:
    """Minimal ELF64 LE ET_DYN: PT_LOAD + PT_DYNAMIC, .dynstr, optional build-id note and extra sections."""
    import struct

    sections = dict(sections or {})
    strtab = b"\0"
    offsets: dict[str, int] = {}
    for name in needed + ([runpath] if runpath is not None else []):
        offsets[name] = len(strtab)
        strtab += name.encode() + b"\0"
    dyn = b"".join(struct.pack("<qQ", 1, offsets[n]) for n in needed)
    if runpath is not None:
        dyn += struct.pack("<qQ", 29, offsets[runpath])
    note = b""
    if build_id is not None:
        note_name = b"GNU\0"
        note = (
            struct.pack("<III", len(note_name), len(build_id), 3)
            + note_name
            + build_id
            + b"\0" * (-len(build_id) % 4)
        )

    # layout: ehdr | phdrs | dynstr | dynamic | note | extra sections | shstrtab | shdrs
    ehdr_size, phdr_size, shdr_size = 64, 56, 64
    n_ph = 2
    blobs: list[tuple[str, bytes, int]] = [
        (".dynstr", strtab, 3),
        (".dynamic", b"", 6),
    ]  # dynamic filled below
    if note:
        blobs.append((".note.gnu.build-id", note, 7))
    for name, data in sections.items():
        blobs.append((name, data, 1))
    off = ehdr_size + n_ph * phdr_size
    placed: list[tuple[str, int, int, int]] = []  # name, offset, size, type
    body = b""
    dynstr_off = off
    for name, data, stype in blobs:
        if name == ".dynamic":
            dyn_off = off + len(body)
            data = dyn + struct.pack("<qQ", 5, dynstr_off) + struct.pack("<qQ", 0, 0)
        placed.append((name, off + len(body), len(data), stype))
        body += data
    shstr = b"\0"
    sh_name_off: dict[str, int] = {}
    for name, _, _, _ in placed + [(".shstrtab", 0, 0, 3)]:
        sh_name_off[name] = len(shstr)
        shstr += name.encode() + b"\0"
    shstr_off = off + len(body)
    body += shstr
    shoff = off + len(body)
    n_sh = len(placed) + 2  # null + shstrtab
    file_size = shoff + n_sh * shdr_size

    ehdr = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    ehdr += struct.pack(
        "<HHIQQQIHHHHHH",
        3,
        0xB7,
        1,
        0,
        ehdr_size,
        shoff,
        0,
        ehdr_size,
        phdr_size,
        n_ph,
        shdr_size,
        n_sh,
        n_sh - 1,
    )
    # PT_LOAD covering the whole file at vaddr == offset, PT_DYNAMIC pointing at the dynamic blob
    dyn_size = next(size for name, _, size, _ in placed if name == ".dynamic")
    phdrs = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, file_size, file_size, 0x1000)
    phdrs += struct.pack(
        "<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, dyn_size, dyn_size, 8
    )
    shdrs = b"\0" * shdr_size
    for name, offset, size, stype in placed:
        shdrs += struct.pack(
            "<IIQQQQIIQQ", sh_name_off[name], stype, 0, offset, offset, size, 0, 0, 1, 0
        )
    shdrs += struct.pack(
        "<IIQQQQIIQQ",
        sh_name_off[".shstrtab"],
        3,
        0,
        0,
        shstr_off,
        len(shstr),
        0,
        0,
        1,
        0,
    )
    out = ehdr + phdrs + body + shdrs
    assert len(out) == file_size
    return out


DEFAULT_UCX = (
    "libucm_cuda",
    "libuct_cma",
    "libuct_cuda",
    "libuct_ib",
    "libuct_ib_mlx5",
    "libuct_ib_mlx5_gda",
    "libuct_ib_efa",
    "libucx_perftest_cuda",
    "libucx_perftest_mad",
    "libuct_ib_mlx5_ext",
    "libuct_ib_mlx5_gdp",
)
DEFAULT_NIXL = (
    "libplugin_UCX.so",
    "libplugin_GDS.so",
    "libplugin_GDS_MT.so",
    "libplugin_POSIX.so",
    "libplugin_OBJ.so",
    "libplugin_AZURE_BLOB.so",
    "libplugin_GPUNETIO.so",
    "libplugin_LIBFABRIC.so",
    "libplugin_GUSLI.so",
    "libplugin_INFINIA.so",
    "libtelemetry_exporter_prometheus.so",
    "libtelemetry_exporter_doca.so",
)
CU13_ONLY_NIXL = ("libtrace_backend_nvtx.so",)
INFINIA_NEEDED = ["libred_client.so.2", "libred_async.so", "libc.so.6"]
INFINIA_RUNPATH = "$ORIGIN/..:$ORIGIN:/opt/ddn/red//lib"


def _cell_label(cu: str, py: str, arch: str) -> str:
    return f"{cu}/{py}/{arch}"


def _selected(selector: str, cu: str, py: str, arch: str) -> str | None:
    """Selector forms: '<cu>/<py>/<arch>:<path>', '<cu>:*:<path>', '*:<path>'. Returns the path or None."""
    if selector.startswith("*:"):
        return selector[2:]
    head, sep, path = selector.partition(":")
    if not sep:
        return None
    if head == _cell_label(cu, py, arch):
        return path
    if path.startswith("*:") and head == cu:
        return path[2:]
    return None


def make_set(
    root: Path,
    version: str,
    *,
    cudas=("cu12", "cu13"),
    pys=("cp310", "cp311"),
    arches=("x86_64", "aarch64"),
    extra: dict[str, bytes] | None = None,
    drop: set[str] | None = None,
    ucx_modules=DEFAULT_UCX,
    nixl_plugins=DEFAULT_NIXL,
    vendored: dict[str, bytes] | None = None,
    vendored_by_cell: dict[str, dict[str, bytes]] | None = None,
    readme_block: str = "",
    ucx_release: str = "1.23.0",
) -> Path:
    """A complete synthetic wheel set with the real path shapes (see the design §3)."""
    root.mkdir(parents=True, exist_ok=True)
    for cu in cudas:
        for py in pys:
            for arch in arches:
                label = _cell_label(cu, py, arch)
                pkg = f"nixl_{cu}"
                members: dict[str, MemberSpec] = {}
                for m in ucx_modules:
                    members[f"{pkg}.libs/ucx/{m}.so.0.0.0"] = f"{m} {arch}".encode()
                    members[f"{pkg}.libs/ucx/{m}.la"] = b"# libtool\n"
                plugins = tuple(nixl_plugins) + (CU13_ONLY_NIXL if cu == "cu13" else ())
                for p in plugins:
                    if p == "libplugin_INFINIA.so":
                        members[f"{pkg}.libs/nixl/{p}"] = make_elf(
                            INFINIA_NEEDED, INFINIA_RUNPATH
                        )
                    else:
                        members[f"{pkg}.libs/nixl/{p}"] = f"{p} {arch}".encode()
                    members[f".{pkg}.mesonpy.libs/plugins/{p}"] = (
                        f"{p} build-tree".encode()
                    )
                members[f".{pkg}.mesonpy.libs/libnixl.so"] = b"libnixl build-tree"
                members[
                    f"{pkg}.libs/libucp-{abs(hash((cu, arch))) % 10**8:08x}.so.0.0.0"
                ] = (b"\0ucp\0" + ucx_release.encode() + b"\0")
                members[f"{pkg}.libs/libucs-aaaaaaaa.so.0.0.0"] = b"ucs"
                for name, data in (vendored or {}).items():
                    members[f"{pkg}.libs/{name}"] = data
                for name, data in ((vendored_by_cell or {}).get(label) or {}).items():
                    if name.startswith("libucp-"):
                        for existing in [
                            k for k in members if k.startswith(f"{pkg}.libs/libucp-")
                        ]:
                            del members[existing]
                    members[f"{pkg}.libs/{name}"] = data
                members[
                    f"nixl_ep_{cu}/nixl_ep_cpp_torch213.{py}-{arch}-linux-gnu.so"
                ] = (b"\0" * 2048)
                members[f"{pkg}/__init__.py"] = b"# nixl\n"
                members[f"{pkg}-{version}.data/headers/asio.hpp"] = b"// asio\n"
                members[f"{pkg}-{version}.dist-info/METADATA"] = (
                    f"Metadata-Version: 2.1\nName: nixl-{cu}\nVersion: {version}\n\n# NIXL\n{readme_block}"
                ).encode()
                members[f"{pkg}-{version}.dist-info/WHEEL"] = (
                    b"Wheel-Version: 1.0\nGenerator: meson\n"
                )
                members[f"{pkg}-{version}.dist-info/RECORD"] = version.encode()
                for selector, data in (extra or {}).items():
                    path = _selected(selector, cu, py, arch)
                    if path is not None:
                        members[path.replace("nixl_cuXX", pkg)] = data
                for selector in drop or set():
                    path = _selected(selector, cu, py, arch)
                    if path is not None:
                        members.pop(path.replace("nixl_cuXX", pkg), None)
                make_wheel(
                    root / f"{pkg}-{version}-{py}-{py}-manylinux_2_28_{arch}.whl",
                    members,
                )
    make_wheel(
        root / f"nixl-{version}-py3-none-any.whl",
        {
            f"nixl-{version}.dist-info/METADATA": (
                f"Metadata-Version: 2.1\nName: nixl\nVersion: {version}\n"
                f"Requires-Dist: nixl-cu12=={version}\nRequires-Dist: nixl-cu13=={version}\n"
            ).encode(),
            f"nixl-{version}.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: setuptools (80.9.0)\n",
            f"nixl-{version}.dist-info/RECORD": version.encode(),
        },
    )
    return root
