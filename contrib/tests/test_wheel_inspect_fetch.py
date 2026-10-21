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

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from wheel_inspect import cli, fetch, versions
from wheel_inspect_fixtures import make_set


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            out, self._body = self._body, b""
        else:
            out, self._body = self._body[:n], self._body[n:]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self):
        self.routes: dict[tuple[str, str], tuple[int, bytes]] = {}
        self.requests: list[urllib.request.Request] = []

    def add(self, url, payload, status=200, method="GET"):
        body = (
            json.dumps(payload).encode()
            if isinstance(payload, (dict, list))
            else payload
        )
        self.routes[(method, url)] = (status, body)

    def __call__(self, request: urllib.request.Request):
        self.requests.append(request)
        key = (request.get_method(), request.full_url)
        if key not in self.routes:
            raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)  # type: ignore[arg-type]
        status, body = self.routes[key]
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "error", {}, None)  # type: ignore[arg-type]
        return FakeResponse(status, body)


@pytest.fixture
def fake_opener():
    return FakeOpener()


def test_parse_version_and_highest_below():
    assert versions.parse_version("1.6.0") == (1, 6, 0)
    assert (
        versions.parse_version("1.6.0rc1") is None
        and versions.parse_version("1.6.0.post1") is None
    )
    assert (
        versions.parse_version("1.6.0.dev3") is None
        and versions.parse_version("1.6.0+local") is None
    )
    assert (
        versions.highest_below(
            ["1.4.0", "1.4.1", "1.5.0", "1.6.0rc1", "1.6.0"], "1.6.0"
        )
        == "1.5.0"
    )
    assert versions.highest_below(["1.10.0", "1.9.0"], "1.11.0") == "1.10.0"
    assert versions.highest_below(["1.6.0"], "1.6.0") is None


def test_resolve_reference_skips_yanked_and_prerelease(fake_opener):
    fake_opener.add(
        "https://pypi.org/pypi/nixl/json",
        {
            "releases": {
                "1.4.1": [{"yanked": False}],
                "1.5.0": [{"yanked": True}],
                "1.6.0rc1": [{"yanked": False}],
                "1.6.0": [{"yanked": False}],
                "1.5.1": [],
            }
        },
    )
    assert fetch.resolve_reference("1.6.0", fake_opener) == "1.4.1"
    with pytest.raises(fetch.FetchError):
        fetch.resolve_reference("1.4.1", fake_opener)


def test_parse_source_forms_and_host_allowlist(tmp_path):
    assert fetch.parse_source("pypi:1.5.0") == ("pypi", "1.5.0")
    assert fetch.parse_source("dir:/x") == ("dir", "/x")
    assert fetch.parse_source(
        "artifactory:https://artifactory.nvidia.com/artifactory/r/release/1.6.0/1"
    ) == (
        "artifactory",
        "https://artifactory.nvidia.com/artifactory/r/release/1.6.0/1",
    )
    with pytest.raises(fetch.FetchError):
        fetch.parse_source("ftp:nope")
    with pytest.raises(fetch.FetchError, match="host"):
        fetch.fetch(
            "artifactory:https://evil.example/repo/release/1.6.0/1",
            tmp_path,
            opener=lambda r: None,
            auth=("u", "t"),
        )
    with pytest.raises(fetch.FetchError, match="credentials"):
        fetch.fetch(
            "artifactory:https://artifactory.nvidia.com/artifactory/r/release/1.6.0/1",
            tmp_path,
            opener=lambda r: None,
            auth=None,
        )


def test_fetch_pypi_downloads_and_verifies_sha(fake_opener, tmp_path):
    body = b"wheel-bytes"
    sha = hashlib.sha256(body).hexdigest()
    for pkg in ("nixl", "nixl-cu12", "nixl-cu13"):
        name = pkg.replace("-", "_")
        fake_opener.add(
            f"https://pypi.org/pypi/{pkg}/1.5.0/json",
            {
                "urls": [
                    {
                        "url": f"https://files.pythonhosted.org/{pkg}.whl",
                        "filename": f"{name}-1.5.0-x.whl",
                        "digests": {"sha256": sha},
                        "size": len(body),
                        "yanked": False,
                    }
                ]
            },
        )
        fake_opener.add(f"https://files.pythonhosted.org/{pkg}.whl", body)
    out = fetch.fetch("pypi:1.5.0", tmp_path / "d", opener=fake_opener)
    assert sorted(p.name for p in out.iterdir()) == [
        "nixl-1.5.0-x.whl",
        "nixl_cu12-1.5.0-x.whl",
        "nixl_cu13-1.5.0-x.whl",
    ]
    assert (out / "nixl-1.5.0-x.whl").read_bytes() == body


def test_fetch_sha_mismatch_raises_and_leaves_no_file(fake_opener, tmp_path):
    fake_opener.add(
        "https://pypi.org/pypi/nixl/1.5.0/json",
        {
            "urls": [
                {
                    "url": "https://files.pythonhosted.org/a.whl",
                    "filename": "nixl-1.5.0-x.whl",
                    "digests": {"sha256": "0" * 64},
                    "size": 1,
                    "yanked": False,
                }
            ]
        },
    )
    for pkg in ("nixl-cu12", "nixl-cu13"):
        fake_opener.add(f"https://pypi.org/pypi/{pkg}/1.5.0/json", {"urls": []})
    fake_opener.add("https://files.pythonhosted.org/a.whl", b"x")
    with pytest.raises(fetch.FetchError, match="sha256"):
        fetch.fetch("pypi:1.5.0", tmp_path / "d", opener=fake_opener)
    assert not list((tmp_path / "d").glob("*.whl"))


def test_fetch_artifactory_listing_and_manifest(fake_opener, tmp_path):
    base = "https://artifactory.nvidia.com/artifactory/sw-nbu-swx-nixl-pypi-local/release/1.6.0/12"
    api = "https://artifactory.nvidia.com/artifactory/api/storage/sw-nbu-swx-nixl-pypi-local/release/1.6.0/12"
    fake_opener.add(
        api + "?list&deep=0&listFolders=0",
        {
            "files": [
                {
                    "uri": "/nixl-1.6.0-py3-none-any.whl",
                    "size": 1,
                    "sha2": hashlib.sha256(b"m").hexdigest(),
                },
                {
                    "uri": "/build-manifest.json",
                    "size": 2,
                    "sha2": hashlib.sha256(b"{}").hexdigest(),
                },
                {"uri": "/other.txt", "size": 1},
            ]
        },
    )
    fake_opener.add(base + "/nixl-1.6.0-py3-none-any.whl", b"m")
    fake_opener.add(base + "/build-manifest.json", b"{}")
    out = fetch.fetch(
        "artifactory:" + base, tmp_path / "d", opener=fake_opener, auth=("u", "t")
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "build-manifest.json",
        "nixl-1.6.0-py3-none-any.whl",
    ]
    assert fake_opener.requests[-1].get_header("Authorization", "").startswith("Basic ")


def test_publish_report_puts_three_files(fake_opener, tmp_path):
    for ext in ("json", "md", "html"):
        (tmp_path / f"wheel-inspection.{ext}").write_text("x")
    folder = "https://artifactory.nvidia.com/artifactory/sw-nbu-swx-nixl-pypi-local/release/1.6.0/12"
    for ext in ("json", "md", "html"):
        fake_opener.add(
            f"{folder}/wheel-inspection.{ext}", b"", status=201, method="PUT"
        )
    fetch.publish_report(folder, tmp_path, ("u", "t"), fake_opener)
    assert [r.get_method() for r in fake_opener.requests] == ["PUT"] * 3
    with pytest.raises(fetch.FetchError, match="prefix"):
        fetch.publish_report(
            "https://artifactory.nvidia.com/artifactory/other-repo/x",
            tmp_path,
            ("u", "t"),
            fake_opener,
        )


def test_release_run_end_to_end_with_dir_sources(tmp_path):
    make_set(tmp_path / "c", "1.6.1")
    make_set(tmp_path / "r", "1.6.0")
    rc = cli.main(
        [
            "release-run",
            "--candidate",
            f"dir:{tmp_path / 'c'}",
            "--reference",
            f"dir:{tmp_path / 'r'}",
            "--out-dir",
            str(tmp_path / "out"),
            "--no-publish",
        ]
    )
    assert rc == 0 and (tmp_path / "out" / "wheel-inspection.json").is_file()


def test_release_run_auto_reference_resolves_from_candidate_version(
    tmp_path, fake_opener, monkeypatch
):
    make_set(tmp_path / "c", "1.6.1")
    make_set(tmp_path / "r", "1.6.0")
    fake_opener.add(
        "https://pypi.org/pypi/nixl/json",
        {"releases": {"1.6.0": [{"yanked": False}], "1.6.1": [{"yanked": False}]}},
    )
    monkeypatch.setattr(cli, "default_opener", lambda: fake_opener)
    calls = []

    def fake_fetch(source, dest, opener=None, auth=None):
        calls.append(source)
        return (
            (tmp_path / "r")
            if source == "pypi:1.6.0"
            else Path(source.split(":", 1)[1])
        )

    monkeypatch.setattr(cli, "fetch_source", fake_fetch)
    rc = cli.main(
        [
            "release-run",
            "--candidate",
            f"dir:{tmp_path / 'c'}",
            "--reference",
            "auto",
            "--out-dir",
            str(tmp_path / "out"),
            "--no-publish",
        ]
    )
    assert rc == 0 and "pypi:1.6.0" in calls
    assert (
        json.loads((tmp_path / "out" / "wheel-inspection.json").read_text())[
            "reference"
        ]["label"]
        == "1.6.0"
    )


def test_release_run_fetch_error_exits_2(tmp_path, fake_opener, monkeypatch, capsys):
    make_set(tmp_path / "c", "1.6.1")
    monkeypatch.setattr(cli, "default_opener", lambda: fake_opener)
    rc = cli.main(
        [
            "release-run",
            "--candidate",
            f"dir:{tmp_path / 'c'}",
            "--reference",
            "pypi:9.9.9",
            "--out-dir",
            str(tmp_path / "out"),
            "--no-publish",
        ]
    )
    assert rc == 2 and "9.9.9" in capsys.readouterr().err


class FlakyOpener(FakeOpener):
    """Fails the first N calls for a URL with a timeout, then serves it."""

    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures
        self.seen: dict[str, int] = {}

    def __call__(self, request):
        url = request.full_url
        self.seen[url] = self.seen.get(url, 0) + 1
        if self.seen[url] <= self.failures and url.startswith(
            "https://files.pythonhosted.org/"
        ):
            raise urllib.error.URLError("The read operation timed out")
        return super().__call__(request)


def test_download_retries_transient_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    body = b"wheel"
    opener = FlakyOpener(failures=2)
    opener.add("https://files.pythonhosted.org/a.whl", body)
    ref = fetch.FileRef(
        "https://files.pythonhosted.org/a.whl",
        "nixl-1.5.0-x.whl",
        hashlib.sha256(body).hexdigest(),
        len(body),
    )
    assert fetch.download(ref, tmp_path, opener).read_bytes() == body
    assert opener.seen["https://files.pythonhosted.org/a.whl"] == 3


def test_download_gives_up_after_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    opener = FlakyOpener(failures=99)
    opener.add("https://files.pythonhosted.org/a.whl", b"x")
    ref = fetch.FileRef(
        "https://files.pythonhosted.org/a.whl", "nixl-1.5.0-x.whl", None, None
    )
    with pytest.raises(fetch.FetchError, match="timed out"):
        fetch.download(ref, tmp_path, opener)
    assert (
        opener.seen["https://files.pythonhosted.org/a.whl"] == fetch.DOWNLOAD_ATTEMPTS
    )
    assert not (tmp_path / "nixl-1.5.0-x.whl").exists()


def test_download_reuses_verified_existing_file(tmp_path, fake_opener):
    body = b"already here"
    (tmp_path / "nixl-1.5.0-x.whl").write_bytes(body)
    ref = fetch.FileRef(
        "https://files.pythonhosted.org/a.whl",
        "nixl-1.5.0-x.whl",
        hashlib.sha256(body).hexdigest(),
        len(body),
    )
    assert fetch.download(ref, tmp_path, fake_opener) == tmp_path / "nixl-1.5.0-x.whl"
    assert fake_opener.requests == []
    (tmp_path / "nixl-1.5.0-x.whl").write_bytes(b"corrupt")
    fake_opener.add("https://files.pythonhosted.org/a.whl", body)
    assert fetch.download(ref, tmp_path, fake_opener).read_bytes() == body
