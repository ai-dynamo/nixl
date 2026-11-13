"""Per-wheel member inventory with the two path normalisations, and the pair diff."""

from __future__ import annotations

import hashlib
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

# nixl_cu12-1.4.1.dist-info/ and nixl_cu12-1.4.1.data/ -> nixl_cu12-VER.<kind>/
_RE_VER = re.compile(r"^(nixl(?:_cu\d+)?)-[^/]*\.(dist-info|data)/")
# nixl_cu12.libs/libucp-72ad654d.so.0.0.0 -> nixl_cu12.libs/libucp-HASH.so.0.0.0
_RE_AUDITHASH = re.compile(r"^(nixl_cu\d+\.libs/[^/]+?)-[0-9a-f]{8}(\.so)")


@dataclass(frozen=True)
class Member:
    key: str
    path: str
    digest: str
    size: int
    compressed: int


class InventoryError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_of(path: str) -> str:
    return _RE_AUDITHASH.sub(r"\1-HASH\2", _RE_VER.sub(r"\1-VER.\2/", path))


def inventory(wheel: Path) -> dict[str, Member]:
    members: dict[str, Member] = {}
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                digest = "symlink:" + zf.read(info).decode("utf-8", "replace")
            else:
                sha = hashlib.sha256()
                with zf.open(info) as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        sha.update(chunk)
                digest = sha.hexdigest()
            key = key_of(info.filename)
            if key in members:
                raise InventoryError(
                    f"{wheel.name}: {members[key].path} and {info.filename} both normalise to {key}"
                )
            members[key] = Member(
                key, info.filename, digest, info.file_size, info.compress_size
            )
    return members


@dataclass
class PairDiff:
    new: dict[str, Member]
    deleted: dict[str, Member]
    rehashed: dict[str, tuple[Member, Member]]
    changed: dict[str, tuple[Member, Member]]


def diff_pair(reference: dict[str, Member], candidate: dict[str, Member]) -> PairDiff:
    diff = PairDiff({}, {}, {}, {})
    for key in sorted(set(candidate) - set(reference)):
        diff.new[key] = candidate[key]
    for key in sorted(set(reference) - set(candidate)):
        diff.deleted[key] = reference[key]
    for key in sorted(set(reference) & set(candidate)):
        old, new = reference[key], candidate[key]
        if _RE_VER.sub(r"\1-VER.\2/", old.path) != _RE_VER.sub(r"\1-VER.\2/", new.path):
            diff.rehashed[key] = (old, new)
        elif old.digest != new.digest:
            diff.changed[key] = (old, new)
    return diff
