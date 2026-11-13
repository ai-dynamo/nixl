# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build source-matched NIXL device bitcode for CuTe (LLVM 20, CUDA 13)."""

import argparse
import logging
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


def clean_ir(text):
    """Remove Clang metadata that the CuTe 4.5.1 libNVVM reader rejects."""
    ids = re.findall(r"^(!\d+)\s*=.*nvvm-reflect-ftz.*$", text, re.MULTILINE)
    lines = []
    for line in text.splitlines():
        if any(line.startswith(f"{key} =") for key in ids):
            continue
        if line.startswith("!llvm.module.flags ="):
            for key in ids:
                line = re.sub(rf"\s*{re.escape(key)}(?!\d),?", "", line)
            line = re.sub(r"!\{\s*,?\s*", "!{", line)
            line = re.sub(r",\s*}", "}", line)
        if line.startswith("define "):
            line = re.sub(r"\s+align\s+\d+(?=\s|\{)", "", line)
        lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ucx", required=True, type=Path)
    parser.add_argument("--arch", required=True, help="e.g. sm_90 or sm_100")
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--llvm", type=Path, default=Path("/usr/lib/llvm-20/bin"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--depfile", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    output = args.output or here / "device.bc"
    dependencies = ["-MD", "-MF", args.depfile, "-MQ", output] if args.depfile else []
    if not (args.ucx / "include/ucp/api/device/ucp_device_impl.h").is_file():
        parser.error("UCX device development headers are required")
    includes = [
        here / "compat",
        root / "src/api/device",
        root / "src/api/cpp",
        root / "src/utils",
        args.ucx / "include",
        args.cuda / "include/cccl",
    ]
    doca = subprocess.run(
        ["pkg-config", "--cflags", "doca-gpunetio"], text=True, capture_output=True
    )
    extra = shlex.split(doca.stdout) if doca.returncode == 0 else []

    def run(tool, *flags):
        subprocess.run([str(args.llvm / tool), *map(str, flags)], check=True)

    with tempfile.TemporaryDirectory() as tmp:
        raw, optimized, cleaned = (
            Path(tmp) / name for name in ("raw.bc", "opt.ll", "clean.ll")
        )
        run(
            "clang++",
            "-std=gnu++20",  # UCX GDA headers use GNU typeof.
            "-fdelayed-template-parsing",  # GDA completes request types after templates.
            "-x",
            "cuda",
            "--cuda-device-only",
            f"--cuda-path={args.cuda}",
            f"--cuda-gpu-arch={args.arch}",
            "-Wno-unknown-cuda-version",
            "-D_NV_RSQRT_SPECIFIER=",
            "-DNDEBUG",
            "-O3",
            "-emit-llvm",
            "-c",
            *(f"-I{path}" for path in includes),
            *extra,
            *dependencies,
            here / "device.cu",
            "-o",
            raw,
        )
        run(
            "opt",
            "--passes=internalize,inline,globaldce",
            "-internalize-public-api-list=cute_nixl_put,cute_nixl_atomic_add,"
            "cute_nixl_get_xfer_status",
            "-S",
            raw,
            "-o",
            optimized,
        )
        cleaned.write_text(clean_ir(optimized.read_text()))
        run("llvm-as", cleaned, "-o", output)
    logging.info("Built %s; rebuild after changing NIXL/UCX/GPU.", output)


if __name__ == "__main__":
    main()
