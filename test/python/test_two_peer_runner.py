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

import os
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

RUNNER = Path(__file__).resolve().parents[2] / ".gitlab" / "run_two_peers.sh"


@pytest.fixture
def peer(tmp_path):
    script = tmp_path / "peer.py"
    script.write_text(
        dedent(
            """\
            import argparse
            import logging
            import os
            import sys
            import time
            from pathlib import Path

            logging.basicConfig(level=logging.INFO, format="%(message)s")
            parser = argparse.ArgumentParser()
            parser.add_argument("--mode", required=True)
            parser.add_argument("--ip", required=True)
            parser.add_argument("--port", type=int, required=True)
            args = parser.parse_args()
            Path(args.mode + ".pid").write_text(str(os.getpid()))
            action = os.environ[args.mode.upper() + "_ACTION"]
            assert args.ip == "127.0.0.1"
            if args.mode == "target":
                assert args.port == 0 and os.environ["NIXL_LOG_LEVEL"] == "INFO"
                assert "INITIATOR_ONLY" not in os.environ
                logging.info("target diagnostic")
                if action == "early":
                    sys.exit(7)
                logging.info("MD listener is listening on port 4242")
                while not Path("initiator.pid").exists():
                    time.sleep(0.02)
            else:
                assert args.port == 4242 and os.environ["INITIATOR_ONLY"] == "1"
            if action == "hold":
                time.sleep(30)
            else:
                sys.exit(int(action))
            """
        )
    )
    return script


@pytest.mark.parametrize(
    ("target", "initiator", "expected"),
    [
        ("0", "0", 0),
        ("7", "0", 7),
        ("0", "3", 3),
        ("7", "hold", 7),
        ("hold", "3", 3),
        ("early", "hold", 7),
    ],
)
def test_two_peer_exit_status(peer, tmp_path, target, initiator, expected):
    logs = tmp_path / "logs"
    logs.mkdir()
    env = dict(
        os.environ, TARGET_ACTION=target, INITIATOR_ACTION=initiator, TMPDIR=str(logs)
    )
    env.pop("INITIATOR_ONLY", None)
    result = subprocess.run(
        ["bash", str(RUNNER), str(peer), "INITIATOR_ONLY=1"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == expected, result.stderr
    if expected:
        assert "target diagnostic" in result.stderr
        role = "target" if expected == 7 else "initiator"
        assert f"{role} exited with status {expected}" in result.stderr
    assert not list(logs.iterdir())
    assert (tmp_path / "initiator.pid").exists() == (target != "early")
    for pid_file in tmp_path.glob("*.pid"):
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
