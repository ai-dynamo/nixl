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

"""A directory of NIXL wheels as a validated set of (CUDA, Python, arch) cells."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable, Optional

CUDA_ORDER = ("cu12", "cu13")
ARCH_ORDER = ("x86_64", "aarch64")

_PACKAGE = re.compile(r"^nixl(?:_cu\d+)?$")
_PLATFORM = re.compile(r"^manylinux_[0-9_]+_(x86_64|aarch64)$")


def py_sort_key(py: str) -> int:
    digits = re.sub(r"\D", "", py)
    return int(digits) if digits else 0


@dataclass(frozen=True, order=True)
class Cell:
    cuda: str
    py: str
    arch: str

    META: ClassVar["Cell"]

    @property
    def is_meta(self) -> bool:
        return self.cuda == "meta"

    @property
    def label(self) -> str:
        return "meta" if self.is_meta else f"{self.cuda}/{self.py}/{self.arch}"


Cell.META = Cell("meta", "-", "any")


def _order_index(order: tuple[str, ...], value: str) -> int:
    return order.index(value) if value in order else len(order)


def sort_cells(cells: Iterable[Cell]) -> list[Cell]:
    return sorted(
        cells,
        key=lambda c: (
            c.is_meta,
            _order_index(CUDA_ORDER, c.cuda),
            c.cuda,
            py_sort_key(c.py),
            _order_index(ARCH_ORDER, c.arch),
            c.arch,
        ),
    )


def parse_wheel_filename(name: str) -> dict[str, Optional[str]]:
    if not name.endswith(".whl"):
        raise ValueError(f"not a wheel filename: {name}")
    parts = name[: -len(".whl")].split("-")
    if len(parts) == 5:
        package, version, python_tag, abi_tag, platform_tag = parts
        build_tag: Optional[str] = None
    elif len(parts) == 6:
        package, version, build_tag, python_tag, abi_tag, platform_tag = parts
    else:
        raise ValueError(f"not a wheel filename: {name}")
    return {
        "package": package,
        "version": version,
        "build_tag": build_tag,
        "python_tag": python_tag,
        "abi_tag": abi_tag,
        "platform_tag": platform_tag,
    }


def cell_of(package: str, python_tag: str, platform_tag: str) -> Cell:
    if not _PACKAGE.match(package):
        raise ValueError(f"not a NIXL wheel package: {package}")
    if package == "nixl":
        return Cell.META
    platform = _PLATFORM.match(platform_tag)
    if not platform:
        raise ValueError(f"unsupported platform tag: {platform_tag}")
    return Cell(package[len("nixl_") :], python_tag, platform.group(1))


@dataclass
class WheelFile:
    path: Path
    name: str
    package: str
    version: str
    cell: Cell
    size: int


class WheelSetError(Exception):
    pass


@dataclass
class WheelSet:
    dir: Path
    version: str
    wheels: dict[Cell, WheelFile]

    @property
    def cells(self) -> list[Cell]:
        return sort_cells(self.wheels)

    @property
    def binary(self) -> dict[Cell, WheelFile]:
        return {c: w for c, w in self.wheels.items() if not c.is_meta}

    @property
    def meta(self) -> WheelFile:
        return self.wheels[Cell.META]

    def pairs_with(
        self, other: "WheelSet"
    ) -> tuple[list[Cell], list[Cell], list[Cell]]:
        mine, theirs = set(self.binary), set(other.binary)
        return (
            sort_cells(mine & theirs),
            sort_cells(mine - theirs),
            sort_cells(theirs - mine),
        )


def load_wheelset(directory: Path) -> WheelSet:
    directory = Path(directory)
    files = sorted(directory.glob("*.whl"))
    if not files:
        raise WheelSetError(f"no wheels in {directory}")

    wheels: dict[Cell, WheelFile] = {}
    versions: set[str] = set()
    for path in files:
        try:
            parsed = parse_wheel_filename(path.name)
            cell = cell_of(
                parsed["package"] or "",
                parsed["python_tag"] or "",
                parsed["platform_tag"] or "",
            )
        except ValueError as exc:
            raise WheelSetError(f"{path.name}: {exc}") from exc
        version = parsed["version"] or ""
        if cell in wheels:
            what = (
                "more than one meta wheel"
                if cell.is_meta
                else f"two wheels for one cell ({cell.label})"
            )
            raise WheelSetError(
                f"{what}, versions {wheels[cell].version} and {version}: {wheels[cell].name}, {path.name}"
            )
        versions.add(version)
        wheels[cell] = WheelFile(
            path, path.name, parsed["package"] or "", version, cell, path.stat().st_size
        )

    if len(versions) != 1:
        raise WheelSetError(
            f"mixed versions in {directory}: {', '.join(sorted(versions))}"
        )
    if Cell.META not in wheels:
        raise WheelSetError(
            f"no meta wheel (nixl-<ver>-py3-none-any.whl) in {directory}"
        )

    binary = [c for c in wheels if not c.is_meta]
    if not binary:
        raise WheelSetError(f"no binary wheels in {directory}")
    cudas = sorted({c.cuda for c in binary})
    pys = sorted({c.py for c in binary}, key=py_sort_key)
    arches = sorted({c.arch for c in binary})
    expected = {
        Cell(cu, py, arch) for cu, py, arch in itertools.product(cudas, pys, arches)
    }
    missing = sort_cells(expected - set(binary))
    if missing:
        raise WheelSetError(
            f"{directory} is not a full product of {cudas} x {pys} x {arches}: missing "
            + ", ".join(c.label for c in missing)
        )
    return WheelSet(directory, versions.pop(), wheels)
