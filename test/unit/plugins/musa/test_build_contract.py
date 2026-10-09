# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute production Meson fragments in small isolated configure projects."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MESON = shutil.which("meson")


class UcxBuildSplitTests(unittest.TestCase):
    def configure(self, public_plugin, static=False):
        self.assertIsNotNone(
            MESON, "meson is required; this test must not silently skip"
        )
        with tempfile.TemporaryDirectory(prefix="nixl-ucx-build-") as temporary:
            project = Path(temporary)
            (project / "ucx").symlink_to(
                ROOT / "src/plugins/ucx", target_is_directory=True
            )
            (project / "meson_options.txt").write_text(
                "option('ucx_path', type: 'string', value: '')\n"
            )
            (project / "meson.build").write_text(
                "project('ucx-build-contract', 'cpp', default_options: ['cpp_std=c++20'])\n"
                "nixl_common_dep = declare_dependency()\n"
                "nixl_infra = declare_dependency()\n"
                "serdes_interface = declare_dependency()\n"
                "thread_dep = dependency('threads')\n"
                "ucx_dep = declare_dependency()\n"
                "nixl_inc_dirs = include_directories('.')\n"
                "utils_inc_dirs = include_directories('.')\n"
                "plugin_install_dir = get_option('libdir')\n"
                "plugin_build_dir = meson.current_build_dir()\n"
                f"static_plugins = {['UCX'] if static else []!r}\n"
                f"enabled_plugins = {{'UCX': {'true' if public_plugin else 'false'}}}\n"
                "subdir('ucx')\n"
            )
            env = os.environ.copy()
            # Configure tests do not link against this declaration. No real UCX is substituted.
            pkgconfig = project / "pkgconfig"
            pkgconfig.mkdir()
            (pkgconfig / "dl.pc").write_text(
                "Name: dl\nDescription: configure-only dl declaration\nVersion: 1\nLibs:\n"
            )
            env["PKG_CONFIG_PATH"] = (
                str(pkgconfig) + os.pathsep + env.get("PKG_CONFIG_PATH", "")
            )
            build = project / "build"
            result = subprocess.run(
                [MESON, "setup", str(build), str(project), "--buildtype=debug"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            return json.loads(
                subprocess.check_output(
                    [MESON, "introspect", str(build), "--targets"],
                    text=True,
                    timeout=30,
                )
            )

    def test_public_ucx_links_private_implementation(self):
        targets = self.configure(True)
        by_name = {target["name"]: target for target in targets}
        self.assertIn("ucx_impl", by_name)
        self.assertIn("UCX", by_name)
        sources = [
            source
            for group in by_name["ucx_impl"]["target_sources"]
            for source in group.get("sources", [])
        ]
        self.assertFalse(any(source.endswith("ucx_plugin.cpp") for source in sources))
        self.assertTrue(any(source.endswith("ucx_backend.cpp") for source in sources))

    def test_musa_only_does_not_export_public_ucx(self):
        names = {target["name"] for target in self.configure(False)}
        self.assertIn("ucx_impl", names)
        self.assertNotIn("UCX", names)

    def test_static_ucx_keeps_its_entry_point(self):
        by_name = {
            target["name"]: target for target in self.configure(True, static=True)
        }
        self.assertEqual(by_name["UCX"]["type"], "static library")
        self.assertIn("ucx_impl", by_name)


class MusaDependencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="nixl-musa-probe-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.project = self.work / "project"
        self.project.mkdir()
        self.fragment = ROOT / "src/plugins/musa/dependency"

    def sdk(self, complete=True, static=False):
        prefix = self.work / "sdk"
        (prefix / "include").mkdir(parents=True)
        (prefix / "lib").mkdir()
        shutil.copy(
            ROOT / "test/unit/plugins/musa/fake_sdk/musa_runtime_api.h",
            prefix / "include",
        )
        if complete:
            source = ROOT / "test/unit/plugins/musa/fake_sdk/runtime.cpp"
        else:
            source = self.work / "missing_symbols.cpp"
            source.write_text('extern "C" int unrelated() { return 0; }\n')
        library = (
            prefix
            / "lib"
            / (
                "libmusart.a"
                if static
                else ("libmusart.dylib" if sys.platform == "darwin" else "libmusart.so")
            )
        )
        compile_flags = ["-std=c++20", "-fPIC", "-I" + str(prefix / "include")]
        if static:
            compile_flags.extend(
                ["-c", str(source), "-o", str(self.work / "runtime.o")]
            )
            subprocess.run(
                shlex.split(os.environ.get("CXX", "c++")) + compile_flags,
                check=True,
                timeout=60,
            )
            subprocess.run(
                ["ar", "rcs", str(library), str(self.work / "runtime.o")],
                check=True,
                timeout=60,
            )
            return prefix
        subprocess.run(
            shlex.split(os.environ.get("CXX", "c++"))
            + [
                "-std=c++20",
                "-fPIC",
                "-dynamiclib" if sys.platform == "darwin" else "-shared",
                "-I" + str(prefix / "include"),
                str(source),
                "-o",
                str(library),
            ],
            check=True,
            timeout=60,
        )
        return prefix

    def configure(self, prefix="", static=""):
        self.assertTrue(
            self.fragment.is_dir(), "production MUSA dependency fragment missing"
        )
        (self.project / "probe").symlink_to(self.fragment, target_is_directory=True)
        (self.project / "meson_options.txt").write_text(
            "option('musa_path', type: 'string', value: '')\n"
            "option('static_plugins', type: 'string', value: '')\n"
        )
        (self.project / "meson.build").write_text(
            "project('musa-sdk-contract', 'cpp', default_options: ['cpp_std=c++20'])\n"
            "cpp = meson.get_compiler('cpp')\n"
            "fs = import('fs')\n"
            "subdir('probe')\n"
            "message('musa-found=' + musa_dep.found().to_string())\n"
        )
        return subprocess.run(
            [
                MESON,
                "setup",
                str(self.work / "build"),
                str(self.project),
                "-Dmusa_path=" + str(prefix),
                "-Dstatic_plugins=" + static,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )

    def test_default_is_disabled(self):
        result = self.configure()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("musa-found=false", result.stdout)

    def test_explicit_missing_prefix_fails(self):
        result = self.configure(self.work / "absent")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MUSA", result.stdout)

    def test_runtime_symbols_are_linked(self):
        result = self.configure(self.sdk())
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("musa-found=true", result.stdout)

    def test_driver_or_empty_runtime_is_not_enough(self):
        result = self.configure(self.sdk(complete=False))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime", result.stdout)

    def test_static_musa_is_explicitly_rejected(self):
        result = self.configure(static="MUSA_UCX")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("static", result.stdout)

    def test_static_runtime_is_explicitly_rejected(self):
        result = self.configure(self.sdk(static=True))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("static", result.stdout)


if __name__ == "__main__":
    unittest.main()
