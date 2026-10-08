#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the temporary CI capture wrapper."""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

WRAPPER = str(Path(__file__).with_name("capture_gpunetio_hang.py"))


class CaptureTest(unittest.TestCase):
    def run_worker(self, code, delay=120, prefix=()):
        return subprocess.run(
            [
                *prefix,
                sys.executable,
                WRAPPER,
                "--capture-after",
                str(delay),
                "--",
                sys.executable,
                "-c",
                code,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_success(self):
        result = self.run_worker(
            "import os; assert os.environ['NIXL_LOG_LEVEL'] == 'DEBUG'"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("GPUNETIO_DIAG", result.stdout)

    def test_failure(self):
        self.assertEqual(self.run_worker("raise SystemExit(7)").returncode, 7)

    def test_capture_then_success(self):
        result = self.run_worker("import time; time.sleep(2)", delay=0.1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("capture before outer timeout", result.stdout)

    @unittest.skipUnless(sys.platform == "linux", "requires GNU timeout and /proc")
    def test_outer_timeout_and_cleanup(self):
        result = self.run_worker(
            "import os,time; print('CHILD_PID='+str(os.getpid()), flush=True); time.sleep(30)",
            delay=0.1,
            prefix=("timeout", "--signal=INT", "--kill-after=3s", "2s"),
        )
        self.assertEqual(result.returncode, 124, result.stderr)
        pid = int(re.search(r"CHILD_PID=(\d+)", result.stdout)[1])
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
