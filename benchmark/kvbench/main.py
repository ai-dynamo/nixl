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

import csv
import glob
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import click
import numpy as np
import yaml
from commands.args import cli_args, common_args, nixl_bench_args, plan_args
from commands.nixlbench import NIXLBench
from models.model_config import ModelConfig
from models.models import BaseModelArch
from models.utils import get_batch_size, override_yaml_args
from tabulate import tabulate


def parse_size(nbytes) -> int:
    """Convert formatted string with unit (e.g. '1M', '512K') or int to bytes."""
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(nbytes, float):
        return int(nbytes)

    options = {"g": 1024 * 1024 * 1024, "m": 1024 * 1024, "k": 1024, "b": 1}
    unit = 1
    key = nbytes[-1].lower()
    if key in options:
        unit = options[key]
        value = float(nbytes[:-1])
    else:
        value = float(nbytes)
    return int(unit * value)


def load_matrix(matrix_file) -> np.ndarray:
    """Load traffic pattern matrix from file"""
    matrix = []
    with open(matrix_file, "r") as f:
        for line in f:
            row = line.strip().split()
            matrix.append([parse_size(x) for x in row])
    mat = np.array(matrix)
    return mat


def align_to_4k(size: int) -> int:
    """Align size up to 4K boundary for O_DIRECT compatibility."""
    ALIGNMENT = 4096
    return ((size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def parse_storage_config(
    storage_config: Dict,
    tp_idx: int,
    storage_base_path: Path,
) -> Optional[Dict[int, Any]]:
    """Parse per-rank storage requirements from YAML config.

    Format (array-based, index is rank):
        storage:
          read: [1M, 1M, 0, 1M, ...]   # 0 or omit for no read
          write: [1M, 1M, 1M, 0, ...]  # 0 or omit for no write

    Args:
        storage_config: Storage configuration with 'read' and/or 'write' arrays
        tp_idx: Traffic pattern index (for file path generation)
        storage_base_path: Base path for storage files

    Returns:
        Dict mapping rank -> StorageOp, or None if empty

    Note:
        Sizes are aligned to 4K boundaries for O_DIRECT compatibility.
    """
    from test.traffic_pattern import StorageOp

    if not storage_config:
        return None

    if "read" not in storage_config and "write" not in storage_config:
        raise ValueError("Storage config must have 'read' and/or 'write' arrays")

    storage_ops = {}
    read_sizes = storage_config.get("read", [])
    write_sizes = storage_config.get("write", [])

    # Determine number of ranks from array lengths
    num_ranks = max(len(read_sizes), len(write_sizes))

    for rank in range(num_ranks):
        # Get size for this rank (0 if not specified)
        read_val = read_sizes[rank] if rank < len(read_sizes) else 0
        write_val = write_sizes[rank] if rank < len(write_sizes) else 0

        # Parse size strings (e.g., "1M", "512K")
        read_size = parse_size(read_val) if read_val else 0
        write_size = parse_size(write_val) if write_val else 0

        # Align sizes to 4K for O_DIRECT compatibility
        read_size = align_to_4k(read_size) if read_size > 0 else 0
        write_size = align_to_4k(write_size) if write_size > 0 else 0
        file_size = read_size + write_size

        if file_size == 0:
            continue

        file_path = storage_base_path / f"tp_{tp_idx}" / f"rank_{rank}.bin"
        storage_ops[rank] = StorageOp(
            file_path=str(file_path),
            file_size=file_size,
            read_offset=0,
            read_size=read_size,
            write_offset=read_size,
            write_size=write_size,
        )

    return storage_ops if storage_ops else None


@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
def cli(debug):
    """KVBench - NIXL Performance Testing CLI"""
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set level for all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


@cli.command("plan")
@cli_args
@common_args
@plan_args
@nixl_bench_args
def plan_command(model, model_config, model_configs, format, **kwargs):
    """Display the recommended configuration for nixlbench"""
    if not model:
        click.echo("Error: --model is required")
        return

    if not model_config and not model_configs:
        click.echo("Error: either --model_config or --model_configs is required")
        return

    # Load model architecture
    model_arch = BaseModelArch.from_yaml(model, None)

    # Get list of model config files
    config_files = []

    if model_config:
        config_files.append(model_config)

    if model_configs:
        # Expand glob patterns into list of files
        expanded_files = glob.glob(model_configs)
        if not expanded_files:
            click.echo(f"Warning: No files matched the pattern: {model_configs}")
        config_files.extend(expanded_files)

    if not config_files:
        click.echo("Error: No valid model config files specified")
        return

    # Filter out duplicate paths
    config_files = list(dict.fromkeys(config_files))

    # Filter arguments for NIXLBench
    filtered_args = {
        k: v for k, v in kwargs.items() if k in NIXLBench.defaults() and v is not None
    }

    # Process each model config
    all_plans = []
    for config_file in config_files:
        # Skip if file doesn't exist
        if not os.path.exists(config_file):
            click.echo(f"Warning: Config file not found: {config_file}")
            continue

        try:
            # Load model configuration
            model_configuration = ModelConfig.from_yaml(config_file)
            # Override yaml args with cli args if supplied
            override_yaml_args(model_configuration, type("Args", (), kwargs)())
            model_arch.set_model_config(model_configuration)

            separator = "=" * 80
            isl_nixl_bench = NIXLBench(model_arch, model_configuration, **filtered_args)

            io_size = model_arch.get_io_size(model_configuration.system.page_size)
            batch_size = get_batch_size(model_arch, model_configuration, io_size)
            isl_nixl_bench.set_io_size(io_size)
            isl_nixl_bench.set_batch_size(batch_size)

            isl_nixl_bench.configure_scheme(direction="isl")
            isl_nixl_bench.configure_segment_type(
                kwargs.get("backend"), kwargs.get("source"), kwargs.get("destination")
            )

            # Generate plan
            plan = isl_nixl_bench.plan(format=format)

            # For JSON format, add config filename to the output
            if format == "json":
                plan_with_config = plan.copy() if isinstance(plan, dict) else {}
                plan_with_config["config_file"] = config_file
                all_plans.append(plan_with_config)
            elif format == "csv":
                plan_data = plan
                # Add metadata
                plan_data["config_file"] = config_file
                plan_data["model"] = model_arch.to_dict().get("model")

                # Add all model_config parameters with proper prefixes
                model_config_dict = model_configuration.to_dict()

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
                click.echo(separator)
                click.echo(f"Model Config: {config_file}")
                click.echo(f"ISL: {model_configuration.runtime.isl} tokens")
                click.echo(f"Page Size: {model_configuration.system.page_size}")
                click.echo(f"Requests: {model_configuration.runtime.num_requests}")
                click.echo(f"TP: {model_configuration.model.tp_size}")
                click.echo(f"PP: {model_configuration.model.pp_size}")
                click.echo(separator)
                click.echo(plan)
                click.echo()
        except Exception as e:
            click.echo(f"Error processing config file {config_file}: {str(e)}")

    # For JSON format, output all plans as an array
    if format == "json" and all_plans:
        click.echo(json.dumps(all_plans, indent=2))
    # For CSV format, output all plans as CSV
    elif format == "csv" and all_plans:
        # Get all unique keys from all plans
        fieldnames = set()
        for plan in all_plans:
            fieldnames.update(plan.keys())
        fieldnames = set(sorted(list(fieldnames)))

        # Write CSV to string buffer
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for plan in all_plans:
            writer.writerow(plan)

        # Print CSV output
        click.echo(output.getvalue())


@cli.command("profile")
@cli_args
@common_args
@nixl_bench_args
def profile_command(model, model_config, **kwargs):
    """Run nixlbench"""
    if not model or not model_config:
        click.echo("Error: --model and --model_config are required")
        return

    model_arch = BaseModelArch.from_yaml(model, None)
    model_configuration = ModelConfig.from_yaml(model_config)
    override_yaml_args(model_configuration, type("Args", (), kwargs)())
    model_arch.set_model_config(model_configuration)

    filtered_args = {
        k: v for k, v in kwargs.items() if k in NIXLBench.defaults() and v is not None
    }
    nixl_bench = NIXLBench(model_arch, model_configuration, **filtered_args)
    io_size = model_arch.get_io_size(model_configuration.system.page_size)
    batch_size = get_batch_size(model_arch, model_configuration, io_size)
    nixl_bench.set_io_size(io_size)
    nixl_bench.set_batch_size(batch_size)
    nixl_bench.configure_buffer_size()

    nixl_bench.configure_scheme(direction="isl")
    nixl_bench.configure_segment_type(
        kwargs.get("backend"), kwargs.get("source"), kwargs.get("destination")
    )
    separator = "=" * 80

    click.echo(f"Model Config: {model_config}")
    click.echo(f"ISL: {model_configuration.runtime.isl} tokens")
    click.echo(f"Page Size: {model_configuration.system.page_size}")
    click.echo(f"Requests: {model_configuration.runtime.num_requests}")
    click.echo(f"TP: {model_configuration.model.tp_size}")
    click.echo(f"PP: {model_configuration.model.pp_size}")
    click.echo(separator)
    nixl_bench.profile()


@cli.command("kvcache")
@cli_args
@common_args
def kvcache_command(model, model_config, **kwargs):
    """Display kvcache information"""
    if not model or not model_config:
        click.echo("Error: --model and --model_config are required")
        return

    # Load model architecture
    model_arch = BaseModelArch.from_yaml(model, None)

    # Load model configuration
    model_configuration = ModelConfig.from_yaml(model_config)
    override_yaml_args(model_configuration, type("Args", (), kwargs)())
    # Set model_config on the model instance using the new method
    model_arch.set_model_config(model_configuration)

    from math import floor, log

    def format_bytes(size):
        power = 0 if size <= 0 else floor(log(size, 1024))
        return f"{round(size / 1024**power, 2)} {['B', 'KB', 'MB', 'GB', 'TB'][int(power)]}"

    labels = [
        "Model",
        "ISL",
        "Num Requests",
        "Batch Size",
        "IO Size",
        "TP",
        "PP",
        "Page Size",
        "Access",
    ]
    io_size = model_arch.get_io_size(model_configuration.system.page_size)
    batch_size = get_batch_size(model_arch, model_configuration, io_size)

    data = [
        [
            model_arch.model_name,
            model_configuration.runtime.isl,
            model_configuration.runtime.num_requests,
            batch_size,
            format_bytes(io_size),
            model_configuration.model.tp_size,
            model_configuration.model.pp_size,
            model_configuration.system.page_size,
            model_configuration.system.access_pattern,
        ]
    ]
    click.echo(tabulate(data, headers=labels, floatfmt=".6f"))


@cli.command("sequential-ct-perftest")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--verify-buffers/--no-verify-buffers",
    default=False,
    help="Verify buffer contents after transfer",
)
@click.option(
    "--print-recv-buffers/--no-print-recv-buffers",
    default=False,
    help="Print received buffer contents",
)
@click.option(
    "--json-output-path",
    type=click.Path(),
    help="Path to save JSON output",
    default=None,
)
@click.option(
    "--storage-path",
    type=click.Path(),
    help="Base path for storage files (default: <config_dir>/storage)",
    default=None,
)
@click.option(
    "--storage-backend",
    type=click.Choice(["POSIX", "GDS", "GDS_MT"]),
    default="POSIX",
    help="NIXL storage backend (POSIX, GDS, or GDS_MT for multi-threaded GDS)",
)
@click.option(
    "--storage-direct-io/--no-storage-direct-io",
    default=None,
    help="Use O_DIRECT for file I/O. Auto-enabled for GDS/GDS_MT if not specified.",
)
@click.option(
    "--warmup-iters",
    type=int,
    default=30,
    help="Number of warmup iterations per TP (default: 30)",
)
@click.option(
    "--isolation-iters",
    type=int,
    default=10,
    help="Number of isolation benchmark iterations per TP (default: 10, was 30)",
)
def sequential_ct_perftest(
    config_file,
    verify_buffers,
    print_recv_buffers,
    json_output_path,
    storage_path,
    storage_backend,
    storage_direct_io,
    warmup_iters,
    isolation_iters,
):
    """Run sequential custom traffic performance test using patterns defined in YAML config."""
    from test.sequential_custom_traffic_perftest import SequentialCTPerftest
    from test.traffic_pattern import TrafficPattern

    logger = logging.getLogger(__name__)

    config_path = Path(config_file)
    config_dir = config_path.parent

    logger.info("Loading config from: %s", config_file)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if "traffic_patterns" not in config:
        raise ValueError("Config file must contain 'traffic_patterns' key")

    # Determine storage base path (CLI override > default)
    if storage_path:
        storage_base_path = Path(storage_path)
    else:
        storage_base_path = config_dir / "storage"

    logger.info("Loading %d traffic patterns...", len(config["traffic_patterns"]))

    patterns = []
    has_storage = False

    for idx, tp_config in enumerate(config["traffic_patterns"]):
        # Load matrix if provided (optional for storage-only patterns)
        matrix = None
        if "matrix_file" in tp_config:
            matrix_file = tp_config["matrix_file"]
            if not os.path.isabs(matrix_file):
                matrix_file = config_dir / matrix_file
            matrix = load_matrix(matrix_file)
            logger.debug(
                "TP %d: matrix=%s, shape=%s, mem_type=%s",
                idx,
                tp_config["matrix_file"],
                matrix.shape,
                tp_config.get("mem_type", "cuda"),
            )
        elif "matrix" in tp_config:
            # Inline matrix in config
            matrix = np.array(tp_config["matrix"])
            logger.debug(
                "TP %d: inline matrix, shape=%s, mem_type=%s",
                idx,
                matrix.shape,
                tp_config.get("mem_type", "cuda"),
            )

        # Parse per-TP storage config if present
        storage_ops = None
        if "storage" in tp_config:
            storage_ops = parse_storage_config(
                tp_config["storage"], idx, storage_base_path
            )
            if storage_ops:
                has_storage = True
                logger.debug(
                    "TP %d: storage config, ranks=%s",
                    idx,
                    list(storage_ops.keys()),
                )

        # Validate: must have either RDMA matrix or storage ops
        if matrix is None and storage_ops is None:
            raise ValueError(
                f"Traffic pattern {idx} must have either 'matrix_file'/'matrix' or 'storage' config"
            )

        # For storage-only patterns without matrix, log it
        if matrix is None:
            logger.debug("TP %d: storage-only pattern (no RDMA)", idx)

        compute_time = tp_config.get("sleep_before_launch_sec")
        if compute_time:
            logger.debug("TP %d: compute_time=%.3f sec", idx, compute_time)

        pattern = TrafficPattern(
            mem_type=tp_config.get("mem_type", "cuda").lower(),  # Default: GPU memory
            matrix=matrix,
            shards=tp_config.get("shards", 1),
            xfer_op=tp_config.get("xfer_op", "WRITE").upper(),
            sleep_before_launch_sec=compute_time,
            sleep_after_launch_sec=tp_config.get("sleep_after_launch_sec", None),
            storage_ops=storage_ops,
        )

        # Storage operations are flexible - any rank can have read/write:
        #   - Storage READ happens before RDMA (for all ranks with read_h)
        #   - Storage WRITE happens after RDMA (for all ranks with write_h)
        #
        # Example configurations:
        #   - Senders: read → send → write
        #   - Receivers: read → receive → write
        #   - Mixed: any combination

        patterns.append(pattern)

    # Determine direct_io setting (auto-enable for GDS if not specified)
    use_direct_io = storage_direct_io
    if storage_direct_io is None and storage_backend in ("GDS", "GDS_MT"):
        use_direct_io = True  # Auto-enable for GDS backends
    elif storage_direct_io is None:
        use_direct_io = False  # Default off for POSIX

    if has_storage:
        logger.info(
            "Loaded %d traffic patterns, storage enabled (path=%s, backend=%s, direct_io=%s)",
            len(patterns),
            storage_base_path,
            storage_backend,
            use_direct_io,
        )
    else:
        logger.info("Loaded %d traffic patterns, no storage", len(patterns))

    # Pass storage config to perftest - it creates the backend with its nixl_agent
    perftest = SequentialCTPerftest(
        patterns,
        warmup_iters=warmup_iters,
        n_isolation_iters=isolation_iters,
        storage_path=storage_base_path if has_storage else None,
        storage_nixl_backend=storage_backend if has_storage else None,
        storage_direct_io=use_direct_io if has_storage else False,
    )
    perftest.run(
        verify_buffers=verify_buffers,
        print_recv_buffers=print_recv_buffers,
        json_output_path=json_output_path,
    )


@cli.command("ct-perftest")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--verify-buffers/--no-verify-buffers",
    default=False,
    help="Verify buffer contents after transfer",
)
@click.option(
    "--print-recv-buffers/--no-print-recv-buffers",
    default=False,
    help="Print received buffer contents",
)
def ct_perftest(config_file, verify_buffers, print_recv_buffers):
    """Run custom traffic performance test using patterns defined in YAML config"""
    from test.custom_traffic_perftest import CTPerftest
    from test.traffic_pattern import TrafficPattern

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    tp_config = config.get("traffic_pattern")
    if tp_config is None:
        raise ValueError("Config file must contain 'traffic_pattern' key")

    iters = config.get("iters", 1)
    warmup_iters = config.get("warmup_iters", 0)

    pattern = TrafficPattern(
        matrix=load_matrix(Path(tp_config["matrix_file"])),
        shards=tp_config.get("shards", 1),
        mem_type=tp_config.get("mem_type", "cuda").lower(),
        xfer_op=tp_config.get("xfer_op", "WRITE").upper(),
    )

    perftest = CTPerftest(pattern, iters=iters, warmup_iters=warmup_iters)
    perftest.run(verify_buffers=verify_buffers, print_recv_buffers=print_recv_buffers)


if __name__ == "__main__":
    cli()
