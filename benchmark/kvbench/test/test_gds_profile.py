# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

KVBENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KVBENCH))

from main import cli  # noqa: E402


@pytest.fixture
def model_args():
    return [
        "--model",
        str(KVBENCH / "examples/model_llama_3_1_8b.yaml"),
        "--model_config",
        str(KVBENCH / "examples/block-tp1-pp1.yaml"),
        "--isl",
        "16",
        "--num_requests",
        "1",
    ]


@pytest.fixture
def benchmark_child(tmp_path, monkeypatch):
    """Exercise real subprocess dispatch without requiring a GPU in CPU CI."""
    capture = tmp_path / "args"
    child = tmp_path / "nixlbench"
    child.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$NIXL_TEST_ARGS"\n'
        'case "$NIXL_TEST_RESULT" in\n'
        '  signal) kill -TERM "$$" ;;\n'
        '  *) exit "$NIXL_TEST_RESULT" ;;\n'
        "esac\n"
    )
    child.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("NIXL_TEST_ARGS", str(capture))
    monkeypatch.setenv("NIXL_TEST_RESULT", "0")
    return capture


@pytest.mark.parametrize("backend", ["GDS", "GDS_MT"])
@pytest.mark.parametrize("source,operation", [("gpu", "WRITE"), ("file", "READ")])
@pytest.mark.parametrize("explicit_conflict", [False, True])
def test_gds_plan_and_profile(
    model_args, benchmark_child, backend, source, operation, explicit_conflict
):
    args = model_args + ["--backend", backend, "--source", source]
    if explicit_conflict:
        args += [
            "--initiator_seg_type",
            "DRAM",
            "--target_seg_type",
            "FILE",
            "--op_type",
            "READ" if operation == "WRITE" else "WRITE",
        ]

    runner = CliRunner()
    plan = runner.invoke(cli, ["plan", *args, "--format", "json"])
    assert plan.exit_code == 0, plan.output
    params = json.loads(plan.output)[0]
    assert params["op_type"] == operation
    assert params["initiator_seg_type"] == "VRAM"
    assert params["target_seg_type"] == "DRAM"
    assert params["max_block_size"] == 1048576
    assert params["max_batch_size"] == 1

    profile = runner.invoke(cli, ["profile", *args])
    assert profile.exit_code == 0, profile.output
    argv = benchmark_child.read_text().splitlines()
    dispatched = dict(zip(argv[::2], argv[1::2]))
    assert dispatched["--backend"] == backend
    assert dispatched["--initiator_seg_type"] == "VRAM"
    # Default WRITE/DRAM values may be omitted. FILE is native-derived, not
    # accepted by nixlbench's target memory-type parser.
    assert dispatched.get("--target_seg_type", "DRAM") == params["target_seg_type"]
    assert dispatched.get("--op_type", "WRITE") == params["op_type"]
    assert int(dispatched["--max_block_size"]) == params["max_block_size"]

    text = runner.invoke(cli, ["plan", *args])
    assert text.exit_code == 0, text.output
    assert "--initiator_seg_type VRAM" in text.output
    assert "--target_seg_type FILE" not in text.output


@pytest.mark.parametrize("command", ["plan", "profile"])
@pytest.mark.parametrize("backend", ["GDS", "GDS_MT"])
def test_gds_invalid_source(model_args, benchmark_child, command, backend):
    result = CliRunner().invoke(
        cli, [command, *model_args, "--backend", backend, "--source", "memory"]
    )
    assert result.exit_code != 0
    diagnostic = result.output + str(result.exception)
    assert "Invalid source for GDS/GDS_MT: memory" in diagnostic
    assert not benchmark_child.exists()


@pytest.mark.parametrize("child_status,expected", [("0", 0), ("7", 7), ("signal", 143)])
def test_profile_child_status(
    model_args, benchmark_child, monkeypatch, child_status, expected
):
    monkeypatch.setenv("NIXL_TEST_RESULT", child_status)
    result = CliRunner().invoke(
        cli, ["profile", *model_args, "--backend", "GDS", "--source", "gpu"]
    )
    assert benchmark_child.exists()
    assert result.exit_code == expected, result.output


def test_profile_missing_executable(model_args, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    result = CliRunner().invoke(
        cli, ["profile", *model_args, "--backend", "GDS", "--source", "gpu"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


def test_ucx_plan_preserves_memory_mapping(model_args):
    result = CliRunner().invoke(
        cli,
        [
            "plan",
            *model_args,
            "--backend",
            "UCX",
            "--source",
            "gpu",
            "--destination",
            "memory",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    params = json.loads(result.output)[0]
    assert params["initiator_seg_type"] == "VRAM"
    assert params["target_seg_type"] == "DRAM"
