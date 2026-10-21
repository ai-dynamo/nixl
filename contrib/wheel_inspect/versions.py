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

"""A PEP 440 subset: plain numeric releases only, enough to pick the previous release."""

from __future__ import annotations

import re
from typing import Iterable, Optional

_PLAIN = re.compile(r"^\d+(\.\d+)*$")


def parse_version(text: str) -> Optional[tuple[int, ...]]:
    """(1, 6, 0) for "1.6.0"; None for pre/dev/post/local versions or anything else."""
    text = text.strip()
    if not _PLAIN.match(text):
        return None
    return tuple(int(part) for part in text.split("."))


def _canonical(parts: tuple[int, ...]) -> tuple[int, ...]:
    """Drop trailing zeros so that "1.6" and "1.6.0" compare equal, as PEP 440 says."""
    end = len(parts)
    while end > 0 and parts[end - 1] == 0:
        end -= 1
    return parts[:end]


def highest_below(candidates: Iterable[str], candidate: str) -> Optional[str]:
    parsed_target = parse_version(candidate)
    if parsed_target is None:
        return None
    target = _canonical(parsed_target)
    best: Optional[tuple[tuple[int, ...], str]] = None
    for text in candidates:
        raw = parse_version(text)
        parsed = _canonical(raw) if raw is not None else None
        if parsed is None or parsed >= target:
            continue
        if best is None or parsed > best[0]:
            best = (parsed, text)
    return best[1] if best else None
