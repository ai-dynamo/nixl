#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture a slow worker's CPU stacks; keep the caller's timeout authoritative."""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-after", type=float, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or args.capture_after <= 0:
        parser.error("a command and positive capture delay are required")

    env = dict(os.environ)
    env["NIXL_LOG_LEVEL"] = "DEBUG"
    child = subprocess.Popen(command, env=env)

    def forward_signal(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    try:
        try:
            returncode = child.wait(timeout=args.capture_after)
        except subprocess.TimeoutExpired:
            print(
                f"GPUNETIO_DIAG pid={child.pid}: capture before outer timeout",
                flush=True,
            )
            tasks = Path(f"/proc/{child.pid}/task")
            for task in sorted(tasks.glob("*")):
                try:
                    print(
                        f"tid={task.name} wchan={task.joinpath('wchan').read_text().strip()}",
                        flush=True,
                    )
                except OSError:
                    pass  # A thread may exit during capture.
            try:
                subprocess.run(
                    [
                        "gdb",
                        "-nx",
                        "-batch",
                        "-q",
                        "-p",
                        str(child.pid),
                        "-ex",
                        "set pagination off",
                        "-ex",
                        "thread apply all bt",
                        "-ex",
                        "detach",
                    ],
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"GPUNETIO_DIAG stack capture unavailable: {error}", flush=True)
            returncode = child.wait()
        return returncode if returncode >= 0 else 128 - returncode
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    sys.exit(main())
