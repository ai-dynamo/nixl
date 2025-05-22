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

from commands.args import add_cli_args, add_common_args
from models.model_config import ModelConfig
from models.models import BaseModelArch
from models.utils import get_batch_size, override_yaml_args


class Command:
    """
    Command handler for the 'kvcache' subcommand.

    This command analyzes and displays key-value cache size information for the
    specified model architecture and configuration, including per-token size,
    total size, and related metrics.
    """

    def __init__(self):
        """
        Initialize the kvcache command.

        Sets the command name and help text for the command-line interface.
        """
        self.name = "kvcache"
        self.help = "Display kvcache information"

    def add_arguments(
        self, subparser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """
        Add command-specific arguments to the argument parser.

        Args:
            subparser (argparse.ArgumentParser): The parser for this command.

        Returns:
            argparse.ArgumentParser: The updated argument parser with added arguments.
        """
        add_common_args(subparser)
        add_cli_args(subparser)
        return subparser

    def execute(self, args: argparse.Namespace):
        """
        Execute the kvcache command with the provided arguments.

        Loads the model architecture and configuration from the specified files,
        calculates KV cache metrics (per-token size, total size, page size), and
        displays the information in a formatted output.

        Args:
            args (argparse.Namespace): Command-line arguments.

        Returns:
            int: -1 if required arguments are missing, otherwise None.
        """
        if not args.model or not args.model_config:
            print("Error: --model and --model_config are required")
            return -1

        # Load model architecture
        model = BaseModelArch.from_yaml(args.model)

        # Load model configuration
        model_config = ModelConfig.from_yaml(args.model_config)
        override_yaml_args(model_config, args)
        # Set model_config on the model instance using the new method
        model.set_model_config(model_config)

        from math import floor, log

        def format_bytes(size):
            power = 0 if size <= 0 else floor(log(size, 1024))
            return f"{round(size / 1024 ** power, 2)} {['B', 'KB', 'MB', 'GB', 'TB'][int(power)]}"

        labels = [
            "KV Cache Size Per Token",
            "IO Size",
            "Batch Size",
            "Model",
            "Input Sequence Length",
        ]
        max_width = max(len(label) for label in labels)
        io_size = model.get_io_size(model_config.system.page_size)
        batch_size = get_batch_size(model, model_config, io_size)
        print(f"{'Model':{max_width}}: {model.model}")
        print(f"{'Input Sequence Length':{max_width}}: {model_config.runtime.isl}")
        print(f"{'Batch Size':{max_width}}: {batch_size}")
        print(f"{'IO Size':{max_width}}: {format_bytes(io_size)}")
