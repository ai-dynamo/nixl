#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only subprocess tests: fake NVIDIA tools never create CUDA/MPS state."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_gpunetio_single_gpu.py"
GPU = "GPU-00000000-0000-0000-0000-000000000001"
FAKE = r"""#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
name = Path(sys.argv[0]).name
root = Path(os.environ['FAKE_ROOT'])
if name == 'ps':
    if os.environ.get('FAKE_MPS_EXISTS'):
        print('S nvidia-cuda-mps-control')
    if os.environ.get('FAKE_MPS_ZOMBIE'):
        print('Z nvidia-cuda-mps')
elif name == 'ip':
    address = '192.0.2.1' if sys.argv[-1] == 'net0' or os.environ.get('FAKE_SAME_IP') else '192.0.2.2'
    print(json.dumps([{'flags': ['UP'], 'addr_info': [{'family': 'inet', 'local': address}]}]))
elif name == 'nvidia-smi':
    if '--query-gpu=uuid' in sys.argv:
        print('GPU-00000000-0000-0000-0000-000000000001')
    elif os.environ.get('FAKE_BUSY'):
        print('1234')
elif name == 'nvidia-cuda-mps-control':
    command = '-d' if '-d' in sys.argv else sys.stdin.read().strip()
    with (root / 'control').open('a') as stream:
        stream.write(command + '\n')
    if command == '-d' and os.environ.get('FAKE_START_FAIL'):
        sys.exit(1)
    if command == 'get_server_list':
        print('9999')
    elif command.startswith('get_client_list'):
        if not os.environ.get('FAKE_NO_CLIENTS'):
            clients = root / 'clients'
            print(clients.read_text() if clients.exists() else '')
else:
    assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
    assert os.environ['CUDA_MPS_PIPE_DIRECTORY'].startswith(str(root))
    with (root / 'clients').open('a') as stream:
        stream.write(str(os.getpid()) + '\n')
    time.sleep(10 if os.environ.get('FAKE_HANG') else 0.3)
    if os.environ.get('FAKE_WORKER_FAIL'):
        sys.exit(2)
    if os.environ.get('FAKE_NO_ROW'):
        sys.exit(0)
    print('4096 1 0.15 26.0')
"""


class LauncherTest(unittest.TestCase):
    def run_case(self, extra=(), flags=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in (
                "nvidia-smi",
                "nvidia-cuda-mps-control",
                "nixlbench",
                "ip",
                "ps",
            ):
                path = root / name
                path.write_text(FAKE)
                path.chmod(0o700)
            env = dict(
                os.environ,
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                FAKE_ROOT=str(root),
            )
            env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
            env.update(flags or {})
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--gpu",
                    GPU,
                    "--oob-interfaces",
                    "net0",
                    "net1",
                    "--timeout",
                    "5",
                    "--output",
                    str(root / "out"),
                    *extra,
                    "--",
                    str(root / "nixlbench"),
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=25,
            )
            report_path = root / "out/result.json"
            report = (
                json.loads(report_path.read_text()) if report_path.exists() else None
            )
            control = (
                (root / "control").read_text() if (root / "control").exists() else ""
            )
            return result, report, control

    def test_auto_checks_both_clients_and_quits(self):
        result, report, control = self.run_case()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(report["mps_clients_verified"])
        self.assertEqual(report["exit_codes"], [0, 0])
        self.assertTrue(control.endswith("quit\n"))

    def test_off_never_starts_mps(self):
        result, report, control = self.run_case(("--mps", "off"))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertFalse(report["mps_clients_verified"])
        self.assertEqual(control, "")

    def test_busy_gpu_is_not_modified(self):
        result, report, control = self.run_case(flags={"FAKE_BUSY": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(report)
        self.assertEqual(control, "")

    def test_unverified_membership_fails(self):
        result, report, control = self.run_case(flags={"FAKE_NO_CLIENTS": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(control.endswith("quit\n"))

    def test_worker_failure_is_propagated(self):
        result, report, control = self.run_case(flags={"FAKE_WORKER_FAIL": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(report["workers_drained"])
        self.assertTrue(control.endswith("quit\n"))

    def test_empty_success_is_rejected(self):
        result, report, control = self.run_case(flags={"FAKE_NO_ROW": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No benchmark result row", report["error"])

    def test_timeout_cleans_up_own_workers(self):
        result, report, control = self.run_case(
            ("--timeout", "0.15"), {"FAKE_HANG": "1"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(report["workers_drained"])
        self.assertTrue(control.endswith("quit\n"))

    def test_start_failure_does_not_launch_workers(self):
        result, report, control = self.run_case(flags={"FAKE_START_FAIL": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["worker_pids"], [])
        self.assertTrue(control.endswith("quit\n"))

    def test_existing_mps_environment_is_not_taken_over(self):
        result, report, control = self.run_case(
            flags={"CUDA_MPS_PIPE_DIRECTORY": "/managed"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(report)
        self.assertEqual(control, "")

    def test_existing_daemon_without_env_is_not_taken_over(self):
        result, report, control = self.run_case(flags={"FAKE_MPS_EXISTS": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(report)
        self.assertEqual(control, "")

    def test_interface_aliases_do_not_bypass_oob_isolation(self):
        result, report, control = self.run_case(flags={"FAKE_SAME_IP": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(report)
        self.assertEqual(control, "")

    def test_exited_unreaped_daemon_does_not_block(self):
        result, report, control = self.run_case(flags={"FAKE_MPS_ZOMBIE": "1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue(report["mps_clients_verified"])


if __name__ == "__main__":
    unittest.main()
