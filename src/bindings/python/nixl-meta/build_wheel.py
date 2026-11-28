# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage only the configured sources so disabling CuTe also removes it on rebuild."""

import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    uv, output, *inputs = sys.argv[1:]
    wheel_dir = Path(output).resolve().parent
    with TemporaryDirectory() as project:
        for source in map(Path, inputs):
            target = Path(project) / source.resolve().relative_to(wheel_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(wheel_dir), project], check=True
        )


if __name__ == "__main__":
    main()
