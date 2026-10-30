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

from wheel_inspect import elf
from wheel_inspect_fixtures import make_elf


def test_parse_elf_needed_and_runpath():
    info = elf.parse_elf(
        make_elf(
            ["libred_client.so.2", "libred_async.so"], "$ORIGIN/..:/opt/ddn/red//lib"
        )
    )
    assert info is not None
    assert info.needed == ["libred_client.so.2", "libred_async.so"]
    assert info.runpath == "$ORIGIN/..:/opt/ddn/red//lib" and info.rpath is None


def test_parse_elf_sections_and_build_id():
    info = elf.parse_elf(
        make_elf(
            [],
            None,
            sections={".nv_fatbin": b"\0" * 4096, ".rodata": b"UCX 1.23.0\0"},
            build_id=bytes(range(20)),
        )
    )
    assert info is not None
    assert info.sections[".nv_fatbin"] == 4096 and info.sections[".rodata"] == 11
    assert info.build_id == bytes(range(20)).hex()
    assert info.needed == [] and info.runpath is None


def test_parse_elf_rejects_non_elf_and_32bit():
    assert elf.parse_elf(b"not an elf at all" * 10) is None
    assert elf.is_elf(b"\x7fELF" + b"\0" * 60) and not elf.is_elf(b"PK\x03\x04")
    data = bytearray(make_elf([], None))
    data[4] = 1  # EI_CLASS = ELFCLASS32
    assert elf.parse_elf(bytes(data)) is None


def test_parse_elf_truncated_returns_none():
    data = make_elf(["libx.so"], "/x")
    assert elf.parse_elf(data[: len(data) // 2]) is None
    assert elf.parse_elf(data[:70]) is None
