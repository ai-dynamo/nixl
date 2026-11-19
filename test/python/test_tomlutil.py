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

import subprocess
import sys
from pathlib import Path

import pytest
import tomlkit

TOMLUTIL = Path(__file__).parents[2] / "contrib" / "tomlutil.py"


def run_tomlutil(*args):
    return subprocess.run(
        [sys.executable, str(TOMLUTIL), *args], capture_output=True, text=True
    )


def test_renames_existing_project_name(tmp_path):
    toml = tmp_path / "pyproject.toml"
    toml.write_text(
        '[project]\nname = "nixl-cu12"\nversion = "1.6.0"\n\n[tool.foo]\nbar = 1\n'
    )
    result = run_tomlutil("--wheel-name", "nixl-cu13", str(toml))
    assert result.returncode == 0, result.stderr
    parsed = tomlkit.parse(toml.read_text())
    assert parsed["project"]["name"] == "nixl-cu13"
    assert parsed["project"]["version"] == "1.6.0"
    assert parsed["tool"]["foo"]["bar"] == 1


def test_adds_missing_project_name(tmp_path):
    toml = tmp_path / "pyproject.toml"
    toml.write_text('[project]\nversion = "1.6.0"\n')
    result = run_tomlutil("--wheel-name", "nixl-cu13", str(toml))
    assert result.returncode == 0, result.stderr
    parsed = tomlkit.parse(toml.read_text())
    assert parsed["project"]["name"] == "nixl-cu13"
    assert parsed["project"]["version"] == "1.6.0"


def test_preserves_comments(tmp_path):
    toml = tmp_path / "pyproject.toml"
    toml.write_text('[project]  # wheel name is rewritten by CI\nname = "x"\n')
    result = run_tomlutil("--wheel-name", "nixl-cu13", str(toml))
    assert result.returncode == 0, result.stderr
    assert "rewritten by CI" in toml.read_text()


def test_fails_cleanly_without_project_table(tmp_path):
    toml = tmp_path / "pyproject.toml"
    toml.write_text("[tool.foo]\nbar = 1\n")
    result = run_tomlutil("--wheel-name", "nixl-cu13", str(toml))
    assert result.returncode == 2
    assert "no [project] table" in result.stderr


def test_fails_cleanly_when_project_is_not_a_table(tmp_path):
    toml = tmp_path / "pyproject.toml"
    toml.write_text("project = 1\n")
    result = run_tomlutil("--wheel-name", "nixl-cu13", str(toml))
    assert result.returncode == 2
    assert "no [project] table" in result.stderr


def test_fails_cleanly_for_missing_file(tmp_path):
    result = run_tomlutil("--wheel-name", "nixl-cu13", str(tmp_path / "nope.toml"))
    assert result.returncode == 2
    assert "cannot read" in result.stderr


def test_fails_cleanly_without_wheel_name(tmp_path):
    toml = tmp_path / "pyproject.toml"
    toml.write_text('[project]\nname = "x"\n')
    result = run_tomlutil(str(toml))
    assert result.returncode == 2
    assert "nothing to do" in result.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
