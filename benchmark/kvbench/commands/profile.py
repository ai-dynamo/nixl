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

from commands.args import add_cli_args, add_common_args, add_nixl_bench_args
from commands.nixlbench import NIXLBench
from models.model_config import ModelConfig
from models.models import BaseModelArch
from models.utils import get_batch_size, override_yaml_args


class Command:
    """
    Command handler for the 'plan' subcommand.

    This command displays the recommended configuration for nixlbench based on
    the provided model architecture and model configuration files.
    """

    def __init__(self):
        """
        Initialize the plan command.

        Sets the command name and help text for the command-line interface.
        """
        self.name = "profile"
        self.help = "Run nixlbench"

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
        add_nixl_bench_args(subparser)
        add_cli_args(subparser)
        return subparser

    def execute(self, args: argparse.Namespace):
        """
        Execute the plan command with the provided arguments.

        Loads the model architecture and configuration from the specified files,
        creates a NIXLBench instance with the provided arguments, and generates
        a nixlbench command plan.

        Args:
            args (argparse.Namespace): Command-line arguments.

        Returns:
            int: -1 if required arguments are missing, otherwise None.
        """
        if not args.model or not args.model_config:
            print("Error: --model and --model_config are required")
            return -1

        if args.model:
            model = BaseModelArch.from_yaml(args.model)
        if args.model_config:
            model_config = ModelConfig.from_yaml(args.model_config)
            override_yaml_args(model_config, args)
            model.set_model_config(model_config)

        filtered_args = {
            k: v for k, v in args.__dict__.items() if k in NIXLBench.defaults()
        }
        nixl_bench = NIXLBench(model, model_config, **filtered_args)
        io_size = model.get_io_size(model_config.system.page_size)
        batch_size = get_batch_size(model, model_config, io_size)
        nixl_bench.set_io_size(io_size)
        nixl_bench.set_batch_size(batch_size)
        nixl_bench.configure_buffer_size()

        nixl_bench.configure_scheme(direction="isl")
        nixl_bench.configure_segment_type(args.backend, args.source, args.destination)
        separator = "=" * 80

        print(f"Model Config: {args.model_config}")
        print(f"ISL: {model_config.runtime.isl} tokens")
        print(f"Page Size: {model_config.system.page_size}")
        print(f"Requests: {model_config.runtime.num_requests}")
        print(f"TP: {model_config.model.tp_size}")
        print(f"PP: {model_config.model.pp_size}")
        print(separator)
        nixl_bench.profile()
