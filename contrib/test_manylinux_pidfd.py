# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the UCX pidfd fallback header embedded in Dockerfile.manylinux."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

DOCKERFILE = Path(__file__).with_name("Dockerfile.manylinux")


class TestManylinuxPidfd(unittest.TestCase):
    def test_generated_header_compiles(self):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler is required")

        source = DOCKERFILE.read_text()
        marker = "printf '%s\\n'"
        self.assertIn(marker, source, "compatibility header must use newline printf")
        self.assertNotIn(
            "printf '%s\\\\n'", source, "double-escaped format emits literal \\n"
        )
        self.assertNotIn(
            "Unknown pidfd syscall ABI",
            source,
            "unsupported architectures must keep building instead of failing",
        )
        self.assertIn(
            "building UCX without the compatibility header",
            source,
            "unsupported architectures must skip the header, not abort",
        )

        section = source.split(marker, 1)[1].split("> /tmp/nixl-pidfd-syscalls.h", 1)[0]
        lines = re.findall(r"'([^']*)'", section)
        self.assertEqual(
            lines,
            [
                "#include <sys/syscall.h>",
                "#ifndef SYS_pidfd_open",
                "#define SYS_pidfd_open 434",
                "#endif",
                "#ifndef SYS_pidfd_getfd",
                "#define SYS_pidfd_getfd 438",
                "#endif",
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = root / "compat.h"
            header.write_text("\n".join(lines) + "\n")
            test_c = root / "check.c"
            test_c.write_text(
                "#include <sys/syscall.h>\n"
                '_Static_assert(SYS_pidfd_open == 434, "wrong pidfd_open");\n'
                '_Static_assert(SYS_pidfd_getfd == 438, "wrong pidfd_getfd");\n'
                "int main(void) { return 0; }\n"
            )
            old_headers = root / "old_headers" / "sys"
            old_headers.mkdir(parents=True)
            (old_headers / "syscall.h").write_text("#pragma once\n")
            for include_dirs in [[], ["-I", str(root / "old_headers")]]:
                executable = root / ("old" if include_dirs else "native")
                subprocess.run(
                    [
                        compiler,
                        "-std=c11",
                        "-Werror",
                        "-Wall",
                        *include_dirs,
                        "-include",
                        str(header),
                        str(test_c),
                        "-o",
                        str(executable),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
