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

"""Resolve and download wheel sets from PyPI or an Artifactory folder, and publish the report."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from wheel_inspect.versions import highest_below, parse_version

ALLOWED_HOSTS = (
    "pypi.org",
    "files.pythonhosted.org",
    "artifactory.nvidia.com",
    "il.artifactory.nvidia.com",
)
PYPI_PACKAGES = ("nixl", "nixl-cu12", "nixl-cu13")
PYPI_INDEX = "https://pypi.org/pypi"
ARTIFACTORY_REPO_PATH = "/artifactory/sw-nbu-swx-nixl-pypi-local/"
REPORT_FILES = ("wheel-inspection.json", "wheel-inspection.md", "wheel-inspection.html")
DOWNLOAD_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 5.0

Opener = Callable[[urllib.request.Request], Any]
Auth = Optional[tuple[str, str]]


class FetchError(Exception):
    pass


@dataclass
class FileRef:
    url: str
    filename: str
    sha256: Optional[str]
    size: Optional[int]


def env_auth() -> Auth:
    user, token = os.environ.get("ARTIFACTORY_USER"), os.environ.get(
        "ARTIFACTORY_TOKEN"
    )
    return (user, token) if user and token else None


def _check_host(url: str) -> None:
    parts = urllib.parse.urlparse(url)
    if parts.scheme != "https" or (parts.hostname or "") not in ALLOWED_HOSTS:
        raise FetchError(
            f"host not allowed (https to {', '.join(ALLOWED_HOSTS)}): {url}"
        )


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to allowed hosts; never carry credentials to another host."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        _check_host(newurl)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and (
            urllib.parse.urlparse(newurl).netloc
            != urllib.parse.urlparse(req.full_url).netloc
        ):
            new.remove_header("Authorization")
        return new


def _request(
    url: str, auth: Auth = None, method: str = "GET", data: Optional[bytes] = None
) -> urllib.request.Request:
    _check_host(url)
    req = urllib.request.Request(url, data=data, method=method)
    if auth:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    return req


def _get(url: str, opener: Opener, auth: Auth = None) -> bytes:
    try:
        with opener(_request(url, auth)) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise FetchError(f"GET {url} returned {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise FetchError(f"GET {url} failed: {exc}") from exc


def _get_json(url: str, opener: Opener, auth: Auth = None) -> Any:
    try:
        return json.loads(_get(url, opener, auth))
    except ValueError as exc:
        raise FetchError(f"{url}: not JSON") from exc


def parse_source(text: str) -> tuple[str, str]:
    kind, sep, value = text.partition(":")
    if not sep or kind not in ("pypi", "artifactory", "dir") or not value:
        raise FetchError(
            f"source must be pypi:<version>, artifactory:<folder URL> or dir:<path>: {text!r}"
        )
    return kind, value


def pypi_versions(
    opener: Opener, index_url: str = PYPI_INDEX, package: str = "nixl"
) -> list[str]:
    doc = _get_json(f"{index_url}/{package}/json", opener)
    releases = doc.get("releases") or {}
    return [
        v
        for v, files in releases.items()
        if any(not f.get("yanked") for f in files or [])
    ]


def resolve_reference(
    candidate_version: str, opener: Opener, index_url: str = PYPI_INDEX
) -> str:
    if parse_version(candidate_version) is None:
        raise FetchError(
            f"candidate version {candidate_version!r} is not a plain release; pass --reference explicitly"
        )
    found = highest_below(pypi_versions(opener, index_url), candidate_version)
    if found is None:
        raise FetchError(f"no released version below {candidate_version} on PyPI")
    return found


def pypi_files(
    version: str, opener: Opener, index_url: str = PYPI_INDEX
) -> list[FileRef]:
    refs = []
    for package in PYPI_PACKAGES:
        doc = _get_json(f"{index_url}/{package}/{version}/json", opener)
        for item in doc.get("urls") or []:
            name = str(item.get("filename", ""))
            if not name.endswith(".whl") or item.get("yanked"):
                continue
            if "/" in name or name.startswith("."):
                raise FetchError(f"refusing PyPI filename {name!r}")
            refs.append(
                FileRef(
                    item["url"],
                    item["filename"],
                    (item.get("digests") or {}).get("sha256"),
                    item.get("size"),
                )
            )
    if not refs:
        raise FetchError(f"no wheels for {version} on PyPI")
    return refs


def _storage_api(folder_url: str) -> str:
    if "/artifactory/" not in folder_url:
        raise FetchError(f"not an Artifactory folder URL: {folder_url}")
    return folder_url.replace("/artifactory/", "/artifactory/api/storage/", 1)


def artifactory_files(folder_url: str, auth: Auth, opener: Opener) -> list[FileRef]:
    doc = _get_json(
        _storage_api(folder_url.rstrip("/")) + "?list&deep=0&listFolders=0",
        opener,
        auth,
    )
    refs = []
    for item in doc.get("files") or []:
        name = str(item.get("uri", "")).lstrip("/")
        if "/" in name or not (name.endswith(".whl") or name == "build-manifest.json"):
            continue
        refs.append(
            FileRef(
                f"{folder_url.rstrip('/')}/{name}",
                name,
                item.get("sha2"),
                item.get("size"),
            )
        )
    if not any(r.filename.endswith(".whl") for r in refs):
        raise FetchError(f"no wheels listed under {folder_url}")
    return refs


class _TransientError(FetchError):
    """A failure worth retrying: timeout, connection reset, HTTP 5xx/429, short or corrupt body."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream(ref: FileRef, part: Path, opener: Opener, auth: Auth) -> None:
    """One download attempt into ``part``; raises _TransientError for anything worth retrying."""
    digest = hashlib.sha256()
    written = 0
    try:
        out = part.open("wb")
    except OSError as exc:
        raise FetchError(f"cannot write {part}: {exc}") from exc
    try:
        with out, opener(_request(ref.url, auth)) as resp:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                try:
                    out.write(chunk)
                except OSError as exc:
                    raise FetchError(f"cannot write {part}: {exc}") from exc
                written += len(chunk)
    except urllib.error.HTTPError as exc:
        if exc.code >= 500 or exc.code == 429:
            raise _TransientError(f"GET {ref.url} returned {exc.code}") from exc
        raise FetchError(f"GET {ref.url} returned {exc.code}") from exc
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        raise _TransientError(f"GET {ref.url} failed: {exc}") from exc
    if ref.size is not None and written != ref.size:
        raise _TransientError(f"{ref.filename}: got {written} of {ref.size} bytes")
    actual = digest.hexdigest()
    if ref.sha256 and actual != ref.sha256:
        raise _TransientError(
            f"{ref.filename}: sha256 mismatch (expected {ref.sha256}, got {actual})"
        )


def download(ref: FileRef, dest_dir: Path, opener: Opener, auth: Auth = None) -> Path:
    """Download ``ref`` into ``dest_dir``, verifying size and sha256 when known; reuses a verified existing file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / ref.filename
    if target.is_file() and ref.sha256 and _sha256_file(target) == ref.sha256:
        return target
    part = dest_dir / (ref.filename + ".part")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            _stream(ref, part, opener, auth)
            part.replace(target)
            return target
        except _TransientError as exc:
            part.unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise FetchError(f"{exc} (after {attempt} attempts)") from exc
            time.sleep(RETRY_DELAY_SECONDS * attempt)
        except FetchError:
            part.unlink(missing_ok=True)
            raise
    raise AssertionError("unreachable")


def fetch(
    source: str, dest: Path, opener: Optional[Opener] = None, auth: Auth = None
) -> Path:
    """Materialise a source as a directory of wheels; returns the directory."""
    kind, value = parse_source(source)
    if kind == "dir":
        path = Path(value)
        if not path.is_dir():
            raise FetchError(f"not a directory: {path}")
        return path
    opener = opener or default_opener()
    dest = Path(dest)
    if kind == "pypi":
        refs = pypi_files(value, opener)
        auth = None
    else:
        _check_host(value)
        if not auth:
            raise FetchError(
                "artifactory sources need credentials (ARTIFACTORY_USER / ARTIFACTORY_TOKEN)"
            )
        refs = artifactory_files(value, auth, opener)
    for ref in refs:
        download(ref, dest, opener, auth)
    keep = {ref.filename for ref in refs}
    for stale in sorted(dest.iterdir()):
        if stale.name in keep or not (
            stale.suffix in (".whl", ".part") or stale.name == "build-manifest.json"
        ):
            continue
        stale.unlink()
        print(f"wheel_inspect: removed stale {stale}", file=sys.stderr)
    return dest


def publish_report(
    folder_url: str, report_dir: Path, auth: Auth, opener: Opener
) -> None:
    _check_host(folder_url)
    if not urllib.parse.urlparse(folder_url).path.startswith(ARTIFACTORY_REPO_PATH):
        raise FetchError(
            f"refusing to publish outside the wheel repo prefix {ARTIFACTORY_REPO_PATH}: {folder_url}"
        )
    if not auth:
        raise FetchError(
            "publishing needs credentials (ARTIFACTORY_USER / ARTIFACTORY_TOKEN)"
        )
    for name in REPORT_FILES:
        data = (Path(report_dir) / name).read_bytes()
        url = f"{folder_url.rstrip('/')}/{name}"
        try:
            with opener(_request(url, auth, method="PUT", data=data)) as resp:
                status = getattr(resp, "status", 200)
        except urllib.error.HTTPError as exc:
            raise FetchError(f"PUT {url} returned {exc.code}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise FetchError(f"PUT {url} failed: {exc}") from exc
        if status not in (200, 201):
            raise FetchError(f"PUT {url} returned {status}")


def default_opener() -> Opener:
    opener = urllib.request.build_opener(_RedirectHandler())
    return lambda request: opener.open(request, timeout=60)
