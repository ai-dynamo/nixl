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

import commands 
import argparse 
def main():
    parser = argparse.ArgumentParser(description="KVBench")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    for command in commands.available_commands:
        subparser = subparsers.add_parser(command.name, help=command.help)
        command.add_arguments(subparser)

    args = parser.parse_args()

    if args.command:
        for command in commands.available_commands:
            if command.name == args.command:
                command.execute(args)
                break
    else:
        parser.print_help()

if __name__ == "__main__":
    main()