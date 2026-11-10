# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the actual JIT cache classes without CUDA; requires a C++ compiler and fmt.

Set FMT_INCLUDE_DIR if fmt is not installed or fetched into subprojects/fmt.
Only CUDA loading and compiler tools are replaced; command execution and cache
I/O use the production code. Each load runs in a fresh process.
"""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

EP_ROOT = Path(__file__).resolve().parents[2]


class JitSymbolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        fmt_candidates = [
            Path(os.environ.get("FMT_INCLUDE_DIR", "/usr/include")),
            EP_ROOT.parents[1] / "subprojects/fmt/include",
            Path("/usr/local/include"),
        ]
        fmt_include = next(
            (p for p in fmt_candidates if (p / "fmt/base.h").is_file()), None
        )
        if not compiler or shutil.which(compiler[0]) is None or fmt_include is None:
            raise unittest.SkipTest(
                "requires a C++ compiler and fmt headers (FMT_INCLUDE_DIR)"
            )
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        for relative in (
            "jit/compiler.hpp",
            "jit/kernel_runtime.hpp",
            "jit/cache.hpp",
            "utils/format.hpp",
            "utils/hash.hpp",
            "utils/lazy_init.hpp",
            "utils/system.hpp",
        ):
            target = cls.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EP_ROOT / "csrc" / relative, target)
        # External CUDA boundaries only; the classes under test are copied verbatim.
        stubs = {
            "ATen/cuda/CUDAContext.h": "",
            "jit/device_runtime.hpp": """
namespace nixl_ep_v2::jit {
struct DeviceRuntime { std::string get_arch() { return "90a"; } };
static DeviceRuntime fake_device;
static DeviceRuntime* device_runtime = &fake_device;
}
""",
            "jit/handle.hpp": """
namespace nixl_ep_v2::jit {
using LibraryHandle = int;
using KernelHandle = std::string;
static KernelHandle load_kernel(const std::filesystem::path& cubin,
                                const std::string& symbol, LibraryHandle* library) {
    EP_HOST_ASSERT(std::filesystem::exists(cubin));
    *library = 1;
    return symbol;
}
static void unload_library(LibraryHandle) {}
}
""",
        }
        for relative, contents in stubs.items():
            target = cls.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
        (cls.root / "main.cpp").write_text(
            r"""
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <vector>
#include "jit/compiler.hpp"
using namespace nixl_ep_v2::jit;
int main(int argc, char** argv) {
    EP_HOST_ASSERT(argc == 2 or argc == 3);
    try {
        if (argc == 3) {
            printf("%d\n", nixl_ep_v2::get_env<int>(argv[2], 7));
            return 0;
        }
        const std::string root = argv[1];
        setenv("NIXL_EP_JIT_CACHE_DIR", argv[1], 1);
        Compiler::prepare_init(root, root, root, root);
        KernelRuntime::prepare_init(root);
        auto runtime = NVCCCompiler().build("probe", "source fixture");
        EP_HOST_ASSERT(runtime->kernel == "_Z12symbol_probev");
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
"""
        )
        cls.binaries = {}
        for bits in (32, 64):
            cls.binaries[bits] = cls.root / f"jit-symbol-test-{bits}"
            subprocess.run(
                compiler
                + [
                    "-std=c++17",
                    f"-DEP_NUM_TOPK_IDX_BITS={bits}",
                    "-I" + str(cls.root),
                    "-I" + str(fmt_include),
                    "-I" + str(EP_ROOT / "nixl_ep_v2/include"),
                    str(cls.root / "main.cpp"),
                    "-o",
                    str(cls.binaries[bits]),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        cls.binary = cls.binaries[64]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name) / "cache space'$(false)"
        (self.cache / "bin").mkdir(parents=True)
        (self.cache / "include").mkdir()
        self.dump = self.cache / "bin/cuobjdump"
        self.dump.write_text(
            '#!/bin/sh\nset -eu\ntest -f "$2"\n'
            'echo called >> "$(dirname "$0")/../dump.calls"\n'
            'echo "STT_FUNC STB_GLOBAL STO_ENTRY _Z12symbol_probev"\n'
        )
        self.dump.chmod(0o755)
        nvcc = self.cache / "bin/nvcc"
        nvcc.write_text(
            "#!/bin/sh\nset -eu\n"
            'if [ "$1" = --version ]; then echo "release 13.0"; exit 0; fi\n'
            'test -f "$1"\n'
            'echo called >> "$(dirname "$0")/../compile.calls"\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  case "$1" in\n'
            '    -o) shift; output="$1";;\n'
            '    -I) shift; test -d "$1";;\n'
            '    -I*) test -d "${1#-I}";;\n'
            '    -DEP_NUM_TOPK_IDX_BITS=*) echo "$1" > "$(dirname "$0")/../index-width.flag";;\n'
            "  esac\n"
            "  shift\n"
            "done\n"
            'echo "host fixture" > "$output"\n'
        )
        nvcc.chmod(0o755)

    def run_cache(self, relative=False, query_setting=None, **extra_env):
        env = {k: v for k, v in os.environ.items() if not k.startswith("NIXL_EP_JIT_")}
        args = [str(self.binary), self.cache.name if relative else str(self.cache)]
        if query_setting is not None:
            args.append(query_setting)
        return subprocess.run(
            args,
            cwd=self.cache.parent,
            env={**env, **extra_env},
            capture_output=True,
            text=True,
        )

    def test_persisted_symbol_load_in_fresh_processes(self):
        result = self.run_cache()
        self.assertEqual(result.returncode, 0, result.stderr)
        entries = list((self.cache / "cache").iterdir())
        self.assertEqual(len(entries), 1)
        symbol = entries[0] / "kernel.symbol"
        self.assertEqual(symbol.read_text(), "_Z12symbol_probev")
        self.assertEqual((self.cache / "dump.calls").read_text(), "called\n")
        self.assertEqual((self.cache / "compile.calls").read_text(), "called\n")

        # A fresh process must load the sidecar without a cuobjdump executable.
        disabled = self.dump.with_suffix(".disabled")
        self.dump.rename(disabled)
        result = self.run_cache()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.cache / "dump.calls").read_text(), "called\n")

        # Current-format cache entries require the symbol file, even if cuobjdump is available.
        disabled.rename(self.dump)
        symbol.unlink()
        result = self.run_cache()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kernel.symbol: clear the JIT cache", result.stderr)
        self.assertEqual((self.cache / "dump.calls").read_text(), "called\n")
        self.assertFalse(symbol.exists())
        self.assertEqual((self.cache / "compile.calls").read_text(), "called\n")

        # A present malformed sidecar is refused with the cache remedy, not recovered.
        symbol.write_text("")
        result = self.run_cache()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kernel.symbol: clear the JIT cache", result.stderr)
        self.assertEqual((self.cache / "dump.calls").read_text(), "called\n")

    def test_routing_index_width_comes_from_build(self):
        for bits, runtime_bits in ((32, 64), (64, 32)):
            self.binary = self.binaries[bits]
            for value in (str(runtime_bits), "invalid"):
                with self.subTest(bits=bits, runtime_setting=value):
                    result = self.run_cache(NIXL_EP_NUM_TOPK_IDX_BITS=value)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        (self.cache / "index-width.flag").read_text(),
                        f"-DEP_NUM_TOPK_IDX_BITS={bits}\n",
                    )

    def test_relative_and_absolute_paths(self):
        for relative in (True, False):
            with self.subTest(relative=relative):
                shutil.rmtree(self.cache / "cache", ignore_errors=True)
                for name in ("compile.calls", "dump.calls"):
                    (self.cache / name).unlink(missing_ok=True)
                for _ in range(2):
                    result = self.run_cache(relative, NIXL_EP_JIT_DUMP_ASM="1")
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual(
                        (self.cache / "compile.calls").read_text(), "called\ncalled\n"
                    )
                    self.assertEqual(
                        (self.cache / "dump.calls").read_text(), "called\ncalled\n"
                    )
                (entry,) = (self.cache / "cache").iterdir()
                self.assertTrue((entry / "kernel.ptx").is_file())
                self.assertTrue((entry / "kernel.sass").is_file())

    def test_relative_and_path_compiler_overrides(self):
        for override in (str(Path(self.cache.name) / "bin/nvcc"), "nvcc"):
            with self.subTest(override=override):
                shutil.rmtree(self.cache / "cache", ignore_errors=True)
                result = self.run_cache(
                    relative=True,
                    NIXL_EP_JIT_NVCC_COMPILER=override,
                    PATH=str(self.cache / "bin") + os.pathsep + os.environ["PATH"],
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_numeric_environment_settings(self):
        name = "NIXL_EP_JIT_CPP_STANDARD"
        for value in (None, "0", "1", "+1", "-1", "2147483647", "-2147483648"):
            with self.subTest(value=value):
                env = {} if value is None else {name: value}
                result = self.run_cache(query_setting=name, **env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(int(result.stdout), 7 if value is None else int(value))
        for value in ("", "off", "1x", "2147483648", "-2147483649", "1" * 100):
            with self.subTest(value=value):
                result = self.run_cache(query_setting=name, **{name: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"Invalid {name}='{value}'", result.stderr)
                self.assertIn("set an in-range base-10 integer", result.stderr)

    def test_symbol_extraction_failure_does_not_publish_cache_entry(self):
        self.dump.write_text("#!/bin/sh\nexit 1\n")
        result = self.run_cache()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.cache / "cache").exists())


if __name__ == "__main__":
    unittest.main()
