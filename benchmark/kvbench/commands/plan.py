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
import glob
import os
import json
import csv
import io
from models.model_config import ModelConfig
from models.models import BaseModelArch
from commands.args import (
    add_common_args,
    add_nixl_bench_args,
    add_plan_args,
    add_cli_args,
)
from commands.nixlbench import NIXLBench
from models.utils import get_batch_size, override_yaml_args


class Command:
    """
    Command handler for the 'plan' subcommand.

    This command displays the recommended configuration for nixlbench based on
    the provided model architecture and model configuration files, showing both
    ISL (Input Sequence Length) and OSL (Output Sequence Length) versions.
    """

    def __init__(self):
        """
        Initialize the plan command.

        Sets the command name and help text for the command-line interface.
        """
        self.name = "plan"
        self.help = "Display the recommended configuration for nixlbench"

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
        add_plan_args(subparser)
        add_nixl_bench_args(subparser)
        add_cli_args(subparser)
        return subparser

    def execute(self, args: argparse.Namespace):
        """
        Execute the plan command with the provided arguments.

        Loads the model architecture and configuration from the specified files,
        creates NIXLBench instances for both ISL and OSL configurations, and
        generates nixlbench command plans for both sequence types.

        Args:
            args (argparse.Namespace): Command-line arguments.

        Returns:
            int: -1 if required arguments are missing, otherwise None.
        """
        if not args.model:
            print("Error: --model is required")
            return -1

        if not args.model_config and not args.model_configs:
            print("Error: either --model_config or --model_configs is required")
            return -1

        # Load model architecture
        model = BaseModelArch.from_yaml(args.model)

        # Get list of model config files
        config_files = []

        if args.model_config:
            config_files.append(args.model_config)

        if args.model_configs:
            # Expand glob patterns into list of files
            expanded_files = glob.glob(args.model_configs)
            if not expanded_files:
                print(f"Warning: No files matched the pattern: {args.model_configs}")
            config_files.extend(expanded_files)

        if not config_files:
            print("Error: No valid model config files specified")
            return -1

        # Filter out duplicate paths
        config_files = list(dict.fromkeys(config_files))

        filtered_args = {
            k: v for k, v in args.__dict__.items() if k in NIXLBench.defaults()
        }

        # Process each model config
        all_plans = []
        for config_file in config_files:
            # Skip if file doesn't exist
            if not os.path.exists(config_file):
                print(f"Warning: Config file not found: {config_file}")
                continue

            try:
                # Load model configuration
                model_config = ModelConfig.from_yaml(config_file)
                # override yaml args with cli args if supplied
                # i.e. if --pp is supplied, use it to override the pp size in the yaml file
                override_yaml_args(model_config, args)
                model.set_model_config(model_config)

                separator = "=" * 80
                isl_nixl_bench = NIXLBench(model, model_config, **filtered_args)

                io_size = model.get_io_size(model_config.system.page_size)
                batch_size = get_batch_size(model, model_config, io_size)
                isl_nixl_bench.set_io_size(io_size)
                isl_nixl_bench.set_batch_size(batch_size)

                isl_nixl_bench.configure_scheme(direction="isl")
                isl_nixl_bench.configure_segment_type(
                    args.backend, args.source, args.destination
                )

                # Generate plan
                plan = isl_nixl_bench.plan(format=args.format)

                # For JSON format, add config filename to the output
                if args.format == "json":
                    plan_with_config = plan.copy() if isinstance(plan, dict) else {}
                    plan_with_config["config_file"] = config_file
                    all_plans.append(plan_with_config)
                elif args.format == "csv":
                    plan_data = plan
                    # Add metadata
                    plan_data["config_file"] = config_file
                    plan_data["model"] = model.to_dict().get("model")

                    # Add all model_config parameters with proper prefixes
                    model_config_dict = model_config.to_dict()

                    # Add strategy parameters
                    for key, value in model_config_dict.get("strategy", {}).items():
                        plan_data[f"model_strategy_{key}"] = value

                    # Add runtime parameters
                    for key, value in model_config_dict.get("runtime", {}).items():
                        plan_data[f"model_runtime_{key}"] = value

                    # Add system parameters
                    for key, value in model_config_dict.get("system", {}).items():
                        plan_data[f"model_system_{key}"] = value

                    all_plans.append(plan_data)
                else:
                    print(separator)
                    print(f"Model Config: {config_file}")
                    print(f"ISL: {model_config.runtime.isl} tokens")
                    print(f"Page Size: {model_config.system.page_size}")
                    print(f"Requests: {model_config.runtime.num_requests}")
                    print(f"TP: {model_config.model.tp_size}")
                    print(f"PP: {model_config.model.pp_size}")
                    print(separator)
                    print(plan)
                    print()
            except Exception as e:
                print(f"Error processing config file {config_file}: {str(e)}")

        # For JSON format, output all plans as an array
        if args.format == "json" and all_plans:
            print(json.dumps(all_plans, indent=2))
        # For CSV format, output all plans as CSV
        elif args.format == "csv" and all_plans:
            # Get all unique keys from all plans
            fieldnames = set()
            for plan in all_plans:
                fieldnames.update(plan.keys())
            fieldnames = sorted(list(fieldnames))

            # Write CSV to string buffer
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for plan in all_plans:
                writer.writerow(plan)

            # Print CSV output
            print(output.getvalue())

        return 0
