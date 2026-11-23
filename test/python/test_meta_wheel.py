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

"""Exercise the real meta-package build without native NIXL dependencies."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from email.parser import Parser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION = "1.2.3"


class MetaWheelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for program in ("meson", "ninja", "uv"):
            if shutil.which(program) is None:
                raise unittest.SkipTest(f"{program} is required for packaging tests")

    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.root = Path(temporary_dir.name)
        self.env = dict(
            os.environ,
            UV_PYTHON=sys.executable,
            # Build against the already-installed setuptools instead of letting
            # uv provision an isolated build env (which needs network access and
            # can fail offline or blow the timeout on repeated builds).
            UV_NO_BUILD_ISOLATION="1",
        )

    def run_command(self, *args, env=None):
        result = subprocess.run(
            [str(arg) for arg in args],
            cwd=self.root,
            env=self.env if env is None else env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def project(self, name):
        source = self.root / name / "source tree"
        build = self.root / name / "build tree"
        shutil.copytree(
            REPO_ROOT / "src/bindings/python/nixl-meta",
            source / "meta",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        for filename in ("LICENSE", "meson_options.txt"):
            shutil.copyfile(REPO_ROOT / filename, source / filename)
        (source / "meson.build").write_text(
            f"project('meta-wheel-test', version: '{VERSION}', "
            "meson_version: '>=0.64.0')\n"
            "py = import('python').find_installation('python3')\n"
            "variant = get_option('wheel_variant')\n"
            "cuda_wheel_dir = variant == '' ? 'nixl_cu13' : 'nixl_' + variant\n"
            "subdir('meta')\n"
        )
        return source, build

    def check_wheel(self, build, ep, variant="cu13", release=False):
        wheel = build / "meta" / f"nixl-{VERSION}-py3-none-any.whl"
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            self.assertTrue(
                {
                    "nixl_meta_utils.py",
                    "nixl/__init__.py",
                    "nixl/_api.py",
                    "nixl/logging.py",
                    f"nixl-{VERSION}.dist-info/licenses/LICENSE",
                    f"nixl-{VERSION}.dist-info/RECORD",
                }.issubset(names)
            )
            self.assertEqual(
                {name for name in names if name.startswith("nixl_ep/")},
                {"nixl_ep/__init__.py"} if ep else set(),
            )
            self.assertFalse(any(name.endswith(".so") for name in names))
            metadata = Parser().parsestr(
                archive.read(f"nixl-{VERSION}.dist-info/METADATA").decode()
            )
            dependencies = {
                item
                for item in metadata.get_all("Requires-Dist", [])
                if ";" not in item
            }
            variants = ("cu12", "cu13") if release else (variant,)
            self.assertEqual(
                dependencies, {f"nixl-{item}=={VERSION}" for item in variants}
            )
        # Exclude site-packages and the checkout so they cannot hide omissions.
        self.run_command(
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import importlib.util, sys; sys.path.insert(0, sys.argv[1]); "
            "import nixl_meta_utils; "
            "assert bool(importlib.util.find_spec('nixl_ep')) == "
            "(sys.argv[2] == 'True')",
            wheel,
            ep,
        )
        return wheel

    def test_clean_wheel_variants(self):
        cases = [
            (variant, ep, release)
            for variant in ("cu12", "cu13")
            for ep in (False, True)
            for release in (False, True)
        ] + [("rocm", False, False)]
        for variant, ep, release in cases:
            with self.subTest(variant=variant, ep=ep, release=release):
                source, build = self.project(f"{variant}-{ep}-{release}")
                self.run_command(
                    "meson",
                    "setup",
                    build,
                    source,
                    f"-Dwheel_variant={variant}",
                    f"-Dbuild_nixl_ep={str(ep).lower()}",
                    f"-Drelease_wheel={str(release).lower()}",
                )
                self.run_command("ninja", "-C", build)
                self.check_wheel(build, ep, variant, release)

    def test_reconfigure_ep(self):
        source, build = self.project("reconfigure")
        self.run_command("meson", "setup", build, source, "-Dbuild_nixl_ep=true")
        for ep in (True, False, True):
            self.run_command(
                "meson", "configure", build, f"-Dbuild_nixl_ep={str(ep).lower()}"
            )
            self.run_command("ninja", "-C", build)
            self.check_wheel(build, ep)

    def test_default_and_incremental_source_edit(self):
        source, build = self.project("incremental")
        self.run_command("meson", "setup", build, source)
        self.run_command("ninja", "-C", build)
        self.check_wheel(build, False)
        helper = source / "meta/nixl_meta_utils.py"
        with helper.open("a") as stream:
            stream.write("\nPACKAGING_TEST_MARKER = True\n")
        self.run_command("ninja", "-C", build)
        wheel = self.check_wheel(build, False)
        with zipfile.ZipFile(wheel) as archive:
            self.assertIn(b"PACKAGING_TEST_MARKER", archive.read(helper.name))

    def test_without_uv(self):
        source, build = self.project("without-uv")
        tools = self.root / "tools"
        tools.mkdir()
        for program in ("meson", "ninja"):
            (tools / program).symlink_to(shutil.which(program))
        (tools / "python3").symlink_to(sys.executable)
        env = dict(self.env, PATH=str(tools))
        output = self.run_command("meson", "setup", build, source, env=env)
        self.assertIn("uv not found, skipping meta package build", output)
        self.run_command("ninja", "-C", build, env=env)
        self.assertFalse(list((build / "meta").glob("*.whl")))


if __name__ == "__main__":
    unittest.main()
