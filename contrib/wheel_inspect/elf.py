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

"""Read what the checks need out of an ELF file without binutils or loading it.

64-bit little-endian only (the x86_64 and aarch64 wheels). Every read is
bounds-checked; anything malformed yields None rather than an exception.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

_PT_LOAD, _PT_DYNAMIC = 1, 2
_DT_NULL, _DT_NEEDED, _DT_STRTAB, _DT_RPATH, _DT_RUNPATH = 0, 1, 5, 15, 29
_NT_GNU_BUILD_ID = 3


@dataclass
class ElfInfo:
    needed: list[str] = field(default_factory=list)
    runpath: Optional[str] = None
    rpath: Optional[str] = None
    build_id: Optional[str] = None
    sections: dict[str, int] = field(default_factory=dict)


class _Reader:
    def __init__(self, data: bytes):
        self.data = data

    def unpack(self, fmt: str, offset: int) -> tuple:
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(self.data):
            raise ValueError("read past end of file")
        return struct.unpack_from(fmt, self.data, offset)

    def cstring(self, offset: int) -> str:
        if offset < 0 or offset >= len(self.data):
            raise ValueError("string offset past end of file")
        end = self.data.find(b"\0", offset)
        if end < 0:
            raise ValueError("unterminated string")
        return self.data[offset:end].decode("utf-8", "replace")

    def slice(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("slice past end of file")
        return self.data[offset : offset + size]


def is_elf(data: bytes) -> bool:
    return data[:4] == b"\x7fELF"


def parse_elf(data: bytes) -> Optional[ElfInfo]:
    try:
        return _parse(data)
    except (ValueError, struct.error, IndexError, UnicodeDecodeError):
        return None


def _parse(data: bytes) -> Optional[ElfInfo]:
    if not is_elf(data) or len(data) < 64 or data[4] != 2 or data[5] != 1:
        return None
    r = _Reader(data)
    (
        _type,
        _machine,
        _version,
        _entry,
        phoff,
        shoff,
        _flags,
        _ehsize,
        phentsize,
        phnum,
        shentsize,
        shnum,
        shstrndx,
    ) = r.unpack("<HHIQQQIHHHHHH", 16)
    info = ElfInfo()

    loads: list[tuple[int, int, int]] = []
    dynamic: Optional[tuple[int, int]] = None
    for i in range(phnum):
        p_type, _p_flags, p_offset, p_vaddr, _p_paddr, p_filesz, _p_memsz, _p_align = (
            r.unpack("<IIQQQQQQ", phoff + i * phentsize)
        )
        if p_type == _PT_LOAD:
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == _PT_DYNAMIC:
            dynamic = (p_offset, p_filesz)

    def vaddr_to_offset(vaddr: int) -> int:
        for seg_vaddr, seg_offset, seg_filesz in loads:
            if seg_vaddr <= vaddr < seg_vaddr + seg_filesz:
                return seg_offset + (vaddr - seg_vaddr)
        raise ValueError("virtual address not in any PT_LOAD")

    if dynamic is not None:
        dyn_offset, dyn_size = dynamic
        r.slice(dyn_offset, dyn_size)
        needed_offsets: list[int] = []
        strtab_vaddr: Optional[int] = None
        rpath_off: Optional[int] = None
        runpath_off: Optional[int] = None
        for pos in range(dyn_offset, dyn_offset + dyn_size - 15, 16):
            tag, val = r.unpack("<qQ", pos)
            if tag == _DT_NULL:
                break
            if tag == _DT_NEEDED:
                needed_offsets.append(val)
            elif tag == _DT_STRTAB:
                strtab_vaddr = val
            elif tag == _DT_RPATH:
                rpath_off = val
            elif tag == _DT_RUNPATH:
                runpath_off = val
        if strtab_vaddr is None and (
            needed_offsets or rpath_off is not None or runpath_off is not None
        ):
            raise ValueError("dynamic section without DT_STRTAB")
        if strtab_vaddr is not None:
            strtab = vaddr_to_offset(strtab_vaddr)
            info.needed = [r.cstring(strtab + off) for off in needed_offsets]
            info.rpath = (
                r.cstring(strtab + rpath_off) if rpath_off is not None else None
            )
            info.runpath = (
                r.cstring(strtab + runpath_off) if runpath_off is not None else None
            )

    if shoff and shnum and shstrndx < shnum:
        _n, _t, _f, _a, shstr_offset, _s = r.unpack(
            "<IIQQQQ", shoff + shstrndx * shentsize
        )
        for i in range(shnum):
            sh_name, sh_type, _sh_flags, _sh_addr, sh_offset, sh_size = r.unpack(
                "<IIQQQQ", shoff + i * shentsize
            )
            if sh_type == 0:
                continue
            name = r.cstring(shstr_offset + sh_name)
            if not name:
                continue
            info.sections[name] = sh_size
            if name == ".note.gnu.build-id":
                namesz, descsz, note_type = r.unpack("<III", sh_offset)
                desc_offset = sh_offset + 12 + (namesz + 3) // 4 * 4
                if note_type == _NT_GNU_BUILD_ID:
                    info.build_id = r.slice(desc_offset, descsz).hex()
    return info
