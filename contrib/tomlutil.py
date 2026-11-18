#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re

import tomlkit

parser = argparse.ArgumentParser()
parser.add_argument("--wheel-name", type=str, help="Set the project name")
parser.add_argument(
    "--remove-dependency",
    action="append",
    default=[],
    metavar="NAME",
    help="Remove a requirement from the project dependencies (repeatable)",
)
parser.add_argument("file", type=str, help="The toml file to modify")
args = parser.parse_args()


def requirement_name(requirement):
    """Return the normalized (PEP 503) project name of a requirement string."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else None


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

if args.remove_dependency:
    # Drop the named requirements whatever their extras or version specifiers
    # Example:
    # ```toml
    # [project]
    # dependencies = ["torch>=2", "numpy"] -> ["numpy"]
    # ```
    names = {requirement_name(name) for name in args.remove_dependency}
    dependencies = doc["project"].get("dependencies", [])
    for index in reversed(range(len(dependencies))):
        if requirement_name(str(dependencies[index])) in names:
            del dependencies[index]

with open(args.file, "w") as f:
    f.write(tomlkit.dumps(doc))
