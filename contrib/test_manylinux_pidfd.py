# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the UCX pidfd fallback header embedded in Dockerfile.manylinux."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

DOCKERFILE = Path(__file__).with_name("Dockerfile.manylinux")


class TestManylinuxPidfd(unittest.TestCase):
    """Exercise the Dockerfile's pidfd setup without building the image."""

    def _run_compat_setup(self, architecture, root):
        """Run the real architecture conditional with an isolated output path."""
        source = DOCKERFILE.read_text().replace("\\\n", " ")
        start = source.index('if [ "$(uname -m)"')
        end = source.index("./contrib/configure-release-mt", start)
        conditional = source[start:end]
        header = root / "compat.h"
        conditional = conditional.replace("/tmp/nixl-pidfd-syscalls.h", str(header))
        environment = dict(os.environ)
        environment.update(
            TEST_ARCH=architecture, CPPFLAGS="-DNIXL_EXISTING_CPPFLAGS=1"
        )
        result = subprocess.run(
            [
                "sh",
                "-c",
                'uname() { printf "%s\\n" "$TEST_ARCH"; }; '
                + conditional
                + ' printf "%s" "$CPPFLAGS"',
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        return header, shlex.split(result.stdout)

    def test_generated_header_compiles(self):
        """Supported ABIs export usable flags for old and native headers."""
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler is required")
        for architecture in ["x86_64", "aarch64"]:
            with (
                self.subTest(architecture=architecture),
                tempfile.TemporaryDirectory() as tmp,
            ):
                self._check_supported_architecture(compiler, architecture, Path(tmp))

    def _check_supported_architecture(self, compiler, architecture, root):
        """Compile using CPPFLAGS emitted by the Dockerfile's actual shell."""
        header, flags = self._run_compat_setup(architecture, root)
        self.assertTrue(header.is_file(), "supported ABI must create the header")
        self.assertEqual(flags, ["-include", str(header), "-DNIXL_EXISTING_CPPFLAGS=1"])
        test_c = root / "check.c"
        test_c.write_text(
            "#include <sys/syscall.h>\n"
            '_Static_assert(SYS_pidfd_open == 434, "wrong pidfd_open");\n'
            '_Static_assert(SYS_pidfd_getfd == 438, "wrong pidfd_getfd");\n'
            '_Static_assert(NIXL_EXISTING_CPPFLAGS == 1, "lost CPPFLAGS");\n'
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
                    *flags,
                    str(test_c),
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(executable)], check=True)

    def test_unsupported_architecture_preserves_flags(self):
        """Unknown ABIs keep building without injecting syscall numbers."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header, flags = self._run_compat_setup("ppc64le", root)
            self.assertFalse(header.exists())
            self.assertEqual(flags, ["-DNIXL_EXISTING_CPPFLAGS=1"])


if __name__ == "__main__":
    unittest.main()
