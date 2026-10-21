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
