#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse

import tomlkit

parser = argparse.ArgumentParser()
parser.add_argument("--wheel-name", type=str, help="Set the project name")
parser.add_argument("--wheel-dir", type=str, help="Set the wheel dir")
parser.add_argument("file", type=str, help="The toml file to modify")
args = parser.parse_args()

with open(args.file) as f:
    doc = tomlkit.parse(f.read())

if args.wheel_name:
    # Set the wheel name
    # Example:
    # ```toml
    # [project]
    # name = "<wheel_name>"
    # ```
    doc["project"]["name"] = args.wheel_name

if args.wheel_dir:
    # Set the wheel dir
    # Example:
    # ```toml
    # [tool.meson-python.args]
    # setup = ["-Dinstall_headers=false", "-Dwheel_dir=<wheel_dir>"]
    # ```
    if "meson-python" not in doc["tool"]:
        doc["tool"]["meson-python"] = tomlkit.table()
    if "args" not in doc["tool"]["meson-python"]:
        doc["tool"]["meson-python"]["args"] = tomlkit.table()
    setup = doc["tool"]["meson-python"]["args"].get("setup", [])
    setup = [s for s in setup if not s.startswith("-Dwheel_dir=")]
    setup.append(f"-Dwheel_dir={args.wheel_dir}")
    doc["tool"]["meson-python"]["args"]["setup"] = setup

with open(args.file, "w") as f:
    f.write(tomlkit.dumps(doc))
