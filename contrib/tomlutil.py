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
parser.add_argument("--set-name", type=str, help="Set the project name")
parser.add_argument("file", type=str, help="The toml file to modify")
args = parser.parse_args()

with open(args.file) as f:
    doc = tomlkit.parse(f.read())

if args.set_name:
    doc["project"]["name"] = args.set_name

with open(args.file, "w") as f:
    f.write(tomlkit.dumps(doc))
