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
