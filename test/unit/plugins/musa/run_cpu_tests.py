# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile and run SDK-free production-contract tests. Not a hardware test."""

import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(__file__).resolve().parents[4]
    tests = Path(__file__).resolve().parent
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    common = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I" + str(root / "src/api/cpp"),
        "-I" + str(root / "src/plugins"),
    ]
    cases = {
        "memory_policy_test": [],
        "registration_test": [],
        "musa_policy_test": [str(root / "src/plugins/musa/musa_policy.cpp")],
        "musa_runtime_test": [
            "-I" + str(tests / "fake_sdk"),
            str(root / "src/plugins/musa/musa_runtime.cpp"),
            str(root / "src/plugins/musa/musa_policy.cpp"),
            str(tests / "fake_sdk/runtime.cpp"),
        ],
    }
    with tempfile.TemporaryDirectory(prefix="nixl-musa-cpu-") as directory:
        for name, extra in cases.items():
            executable = str(Path(directory) / name)
            subprocess.run(
                compiler
                + common
                + [str(tests / (name + ".cpp"))]
                + extra
                + ["-o", executable],
                check=True,
            )
            subprocess.run([executable], check=True, timeout=30)
    for source in [root / "test/integration/musa/musa_ucx_e2e.cpp"]:
        subprocess.run(
            compiler
            + [
                "-std=c++20",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-isystem",
                str(root / "src/api/cpp"),
                "-I" + str(tests / "fake_sdk"),
                "-fsyntax-only",
                str(source),
            ],
            check=True,
            timeout=30,
        )
    LOGGER.info("PASS: %d SDK-free C++ test executables", len(cases))
    LOGGER.info(
        "PASS: hardware harness NIXL API syntax (fake SDK declarations, not hardware)"
    )


if __name__ == "__main__":
    main()
