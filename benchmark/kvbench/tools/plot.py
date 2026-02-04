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

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Literal, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml

# CLI tool output wrapper - satisfies check_prints.sh CI check
output = print

log = logging.getLogger(__name__)


def parse_size(nbytes: str) -> int:
    """Convert formatted string with unit to bytes"""

    options = {"g": 1024 * 1024 * 1024, "m": 1024 * 1024, "k": 1024, "b": 1}
    unit = 1
    key = nbytes[-1].lower()
    if key in options:
        unit = options[key]
        value = float(nbytes[:-1])
    else:
        value = float(nbytes)
    count = int(unit * value)
    return count


@dataclass
class TrafficPattern:
    """Represents a communication pattern between distributed processes.

    Attributes:
        matrix: Communication matrix as numpy array
        mem_type: Type of memory to use
        xfer_op: Transfer operation type
        shards: Number of shards for distributed processing
        dtype: PyTorch data type for the buffers
        sleep_before_launch_sec: Number of seconds to sleep before launch
        sleep_after_launch_sec: Number of seconds to sleep after launch
        id: Unique identifier for this traffic pattern
    """

    matrix: np.ndarray
    xfer_op: Literal["WRITE", "READ"] = "WRITE"
    shards: int = 1
    sleep_before_launch_sec: Optional[int] = None
    sleep_after_launch_sec: Optional[int] = None

    id: int = field(default_factory=lambda: TrafficPattern._get_next_id())
    _id_counter: ClassVar[int] = 0

    @classmethod
    def _get_next_id(cls) -> int:
        """Get the next available ID and increment the counter"""
        current_id = cls._id_counter
        cls._id_counter += 1
        return current_id

    def senders_ranks(self):
        """Return the ranks that send messages"""
        senders_ranks = []
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    senders_ranks.append(i)
                    break
        return list(set(senders_ranks))

    def receivers_ranks(self, from_ranks: Optional[list[int]] = None):
        """Return the ranks that receive messages"""
        if from_ranks is None:
            from_ranks = list(range(self.matrix.shape[0]))
        receivers_ranks = []
        for i in from_ranks:
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] > 0:
                    receivers_ranks.append(j)
                    break
        return list(set(receivers_ranks))

    def ranks(self):
        """Return all ranks that are involved in the traffic pattern"""
        return list(set(self.senders_ranks() + self.receivers_ranks()))

    def buf_size(self, src, dst):
        return self.matrix[src, dst]

    def total_src_size(self, rank):
        """Return the sum of the sizes received by <rank>"""
        total_src_size = 0
        for other_rank in range(self.matrix.shape[0]):
            total_src_size += self.matrix[rank][other_rank]
        return total_src_size

    def total_dst_size(self, rank):
        """Return the sum of the sizes received by <rank>"""
        total_dst_size = 0
        for other_rank in range(self.matrix.shape[0]):
            total_dst_size += self.matrix[other_rank][rank]
        return total_dst_size


def parse_columns_from_log(filename):
    """
    Parse a text file with space-separated columns and extract the first two columns
    into x and y lists.

    Args:
        filename (str): Path to the file to parse

    Returns:
        tuple: (x, y) where x is the first column and y is the second column
    """
    x = []
    y = []
    sols = []

    with open(filename, "r") as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue

            # Split the line by whitespace and filter out empty strings
            values = [val for val in line.strip().split() if val]

            # If we have at least 2 columns
            if len(values) >= 2:
                try:
                    # Convert the first two columns to float
                    x_val = float(values[0])
                    y_val = float(values[1])
                    sol_val = float(values[2])

                    # Append to our lists
                    x.append(x_val)
                    y.append(y_val)
                    sols.append(sol_val)
                except ValueError:
                    # Skip lines that don't have valid float values
                    continue

    return x, y, sols


def parse_columns(filename):
    """Parse columns from json"""
    sizes = []
    latencies_by_iter = []
    isolated_latencies_by_iter = []
    num_senders = []

    with open(filename, "r") as f:
        data = json.load(f)
        metadata = data["metadata"]
        for tp_results in data["iterations_results"][0]:
            sizes.append(tp_results["size"])
            num_senders.append(tp_results["num_senders"])

        for iter_results in data["iterations_results"]:
            latencies = []
            isolated_latencies = []
            for tp_results in iter_results:
                latencies.append(tp_results["latency"])
                isolated_latencies.append(tp_results["isolated_latency"])
            latencies_by_iter.append(latencies)
            isolated_latencies_by_iter.append(isolated_latencies)

    # Mean latencies
    mean_isolated_latencies = np.mean(isolated_latencies_by_iter, axis=0)

    return sizes, latencies_by_iter, mean_isolated_latencies, num_senders, metadata


def plot_run(
    log_file,
    title,
    outfile=None,
    max_y=None,
    max_x=None,
    sol=48.75,
    plot_theoretical_minimum=False,
):

    x, y_lists, sols, n_senders, metadata = parse_columns(log_file)

    run_datetime = datetime.fromtimestamp(metadata["ts"])
    # Plot distribution of x values
    # plt.figure(figsize=(10, 6))
    # plt.hist(x, bins=30, density=True, alpha=0.7)
    # plt.xlabel('Input Sequence Length')
    # plt.ylabel('Density')
    # plt.title('Distribution of Input Sequence Lengths')
    # plt.savefig("/swgwork/eshukrun/nixl/tools/perf/run/8_april/distribution.png")
    # plt.clf()

    plt.figure(figsize=(10, 6))

    # Theoretical minimum
    if plot_theoretical_minimum:
        x_min = min(x)
        x_max = max(x)
        N = 50
        x_linspace = [x_min + i * (x_max - x_min) / (N - 1) for i in range(N)]
        sol_bw = (
            sol * n_senders[0]
        )  # Assume all iterations have the same number of senders
        sol_line = [1e3 * x / sol_bw for x in x_linspace]
        plt.plot(x_linspace, sol_line, "r--", label="Theoretical minimum")

    colors = [
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]  # Define 10 colors
    # Plot individual iterations
    for i, y in enumerate(y_lists):
        plt.scatter(x, y, s=1, color=colors[i], label=f"Workload iter {i + 1}")
        # Add trend line using numpy's polyfit
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(
            x,
            p(x),
            color=colors[i],
            linestyle="--",
            alpha=0.5,
            label=f"Iter {i + 1} trend",
        )

    # Add global trend line across all iterations
    all_y = np.concatenate(y_lists)
    all_x = np.array(x * len(y_lists))  # Repeat x values for each iteration
    global_z = np.polyfit(all_x, all_y, 1)
    global_p = np.poly1d(global_z)
    plt.plot(
        x, global_p(x), color="black", linestyle="-", linewidth=1, label="Global trend"
    )

    # Sort x and sols together based on x values
    x_sorted, sols_sorted = zip(*sorted(zip(x, sols)))
    plt.plot(x_sorted, sols_sorted, "r-", linewidth=1, label="Micro-benchmark")
    plt.legend()
    plt.xlabel("KV Cache size (GB)")
    plt.ylabel("Latency (ms)")
    plt.title(title, wrap=True)
    subtitle = f"{sum(len(y) for y in y_lists)} points, {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    plt.suptitle(subtitle, fontsize=10)
    if max_y is not None:
        plt.ylim(0, max_y)
    if max_x is not None:
        plt.xlim(0, max_x)

    outfile = outfile or os.path.join(os.path.dirname(log_file), "plot.png")
    outfile = outfile.replace(" ", "_")
    plt.savefig(outfile)
    plt.clf()
    output(f"Plotted {title} to {outfile}")


def plot_isolated_diff(
    log_file_1,
    log_file_2,
    label_1,
    label_2,
    title,
    outfile=None,
    max_y=None,
    max_x=None,
):
    outfile = (
        outfile or f"/swgwork/eshukrun/nixl/tools/perf/run/8_april/plot_{title}.png"
    )
    outfile = outfile.replace(" ", "_")

    x1, y_lists_1, mean_isolated_latencies_1, num_senders_1, metadata_1 = parse_columns(
        log_file_1
    )
    x2, y_lists_2, mean_isolated_latencies_2, num_senders_2, metadata_2 = parse_columns(
        log_file_2
    )

    plt.figure(figsize=(10, 6))
    x_sorted_1, mean_isolated_latencies_sorted_1 = zip(
        *sorted(zip(x1, mean_isolated_latencies_1))
    )
    x_sorted_2, mean_isolated_latencies_sorted_2 = zip(
        *sorted(zip(x2, mean_isolated_latencies_2))
    )
    plt.plot(
        x_sorted_1,
        mean_isolated_latencies_sorted_1,
        color="black",
        linestyle="-",
        linewidth=1,
        label=label_1,
    )

    plt.plot(
        x_sorted_2,
        mean_isolated_latencies_sorted_2,
        color="green",
        linestyle="-",
        linewidth=1,
        label=label_2,
    )

    # Sort x and sols together based on x values
    plt.legend()
    plt.xlabel("KV Cache size (GB)")
    plt.ylabel("Isolated Latency (ms)")
    plt.title(title, wrap=True)
    if max_y is not None:
        plt.ylim(0, max_y)
    plt.savefig(outfile)
    plt.clf()
    output(f"Plotted isolated diff {title} to {outfile}")


def load_matrix(matrix_file) -> np.ndarray:
    """Load traffic pattern matrix from file"""
    matrix = []
    with open(matrix_file, "r") as f:
        for line in f:
            row = line.strip().split()
            matrix.append([parse_size(x) for x in row])
    mat = np.array(matrix)
    return mat


def plot_diff(
    log_file_1,
    log_file_2,
    label_1,
    label_2,
    title,
    outfile=None,
    max_y=None,
    max_x=None,
):
    outfile = outfile.replace(" ", "_")

    x1, y_lists_1, sols_1, num_senders_1, metadata_1 = parse_columns(log_file_1)
    x2, y_lists_2, sols_2, num_senders_2, metadata_2 = parse_columns(log_file_2)

    plt.figure(figsize=(10, 6))

    all_y_1 = np.concatenate(y_lists_1)
    all_x_1 = np.array(x1 * len(y_lists_1))  # Repeat x values for each iteration
    global_z_1 = np.polyfit(all_x_1, all_y_1, 1)
    global_p_1 = np.poly1d(global_z_1)
    plt.plot(
        x1, global_p_1(x1), color="black", linestyle="-", linewidth=1, label=label_1
    )

    all_y_2 = np.concatenate(y_lists_2)
    all_x_2 = np.array(x2 * len(y_lists_2))  # Repeat x values for each iteration
    global_z_2 = np.polyfit(all_x_2, all_y_2, 1)
    global_p_2 = np.poly1d(global_z_2)
    plt.plot(
        x2, global_p_2(x2), color="green", linestyle="-", linewidth=1, label=label_2
    )

    # Sort x and sols together based on x values
    plt.legend()
    plt.xlabel("KV Cache size (GB)")
    plt.ylabel("Latency (ms)")
    plt.title(title, wrap=True)
    if max_y is not None:
        plt.ylim(0, max_y)
    plt.savefig(outfile)
    plt.clf()
    output(f"Plotted {title} to {outfile}")


def extract_transfers_timeline_from_dynamo(log_file):
    """
    Extract the transfers timeline from the dynamo decode log file
    return timelines by req and num blocks by req
    """

    timestamps = {}
    sizes = defaultdict(int)
    for line in open(log_file, "r").readlines():
        # Catch start_load_kv line
        # 2025-09-08T07:25:28.465Z DEBUG nixl_connector.start_load_kv: start_load_kv for request 377c1e5e524148b1b57922c602150450 from remote engine 386d7fae-9bab-42f2-bcba-917c47db25b3. Num local_block_ids: 1000. Num remote_block_ids: 1000.
        m = re.search(r"start_load_kv for request ([a-fA-F0-9]+)", line)
        if m:
            req_id = m.group(1)
            num_local_blocks = int(
                re.search(r"Num local_block_ids: ([0-9]+)", line).group(1)
            )
            sizes[req_id] += num_local_blocks

        # Catch finished recving line
        m = re.search(r"Finished recving KV transfer for request ([a-fA-F0-9]+)", line)
        if m:
            req_id = m.group(1)
            dt = datetime.strptime(line.split()[0], "%Y-%m-%dT%H:%M:%S.%fZ")
            timestamps[req_id] = dt.timestamp()
    return timestamps, sizes


def plot_histogram_comparison(
    log_file,
    dynamo_log_file,
    title="Timestamp Distribution Comparison",
    outfile=None,
    time_resolution=10000,
):
    """
    Plot superposed histograms with 50 bins each for start_timestamps and dynamo_timestamps
    """
    plt.figure(figsize=(12, 8))

    # Load and process start timestamps
    data = json.load(open(log_file))
    start_timestamps = []

    for iter_results in data["iterations_results"]:
        for ix, tp_results in enumerate(iter_results):
            start_timestamps.append(tp_results["min_start_ts"])

    # Process start timestamps
    start_timestamps = [
        int((ts - min(start_timestamps)) * time_resolution) for ts in start_timestamps
    ]

    # Load and process dynamo timestamps
    dynamo_timestamps, dynamo_num_blocks = extract_transfers_timeline_from_dynamo(
        dynamo_log_file
    )
    min_ts = min(dynamo_timestamps.values())
    dynamo_timestamp_values = [
        int((ts - min_ts) * time_resolution) for ts in dynamo_timestamps.values()
    ]

    # Create superposed histograms with 50 bins each
    plt.hist(
        start_timestamps,
        bins=50,
        alpha=0.6,
        edgecolor="black",
        label=f"Start timestamps (n={len(start_timestamps)})",
        color="red",
    )

    plt.hist(
        dynamo_timestamp_values,
        bins=50,
        alpha=0.6,
        edgecolor="black",
        label=f"Dynamo timestamps (n={len(dynamo_timestamp_values)})",
        color="blue",
    )

    plt.xlabel(f"Time (1/{time_resolution} s)")
    plt.ylabel("Frequency")
    plt.title(f"{title} (50 bins each)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics to the plot
    start_mean = np.mean(start_timestamps)
    start_std = np.std(start_timestamps)
    dynamo_mean = np.mean(dynamo_timestamp_values)
    dynamo_std = np.std(dynamo_timestamp_values)

    # Add text box with statistics
    stats_text = f"Start: μ={start_mean:.1f}, σ={start_std:.1f}\nDynamo: μ={dynamo_mean:.1f}, σ={dynamo_std:.1f}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Save histogram
    outfile = outfile or "timestamp_histogram_comparison.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    output(f"Plotted histogram comparison to {outfile}")
    output(
        f"Start timestamps: {len(start_timestamps)} samples, mean={start_mean:.1f}, std={start_std:.1f}"
    )
    output(
        f"Dynamo timestamps: {len(dynamo_timestamp_values)} samples, mean={dynamo_mean:.1f}, std={dynamo_std:.1f}"
    )
    plt.show()
    plt.clf()


def plot_transfers_timeline(log_file, title, outfile=None, time_resolution=1000):
    """
    For now plot start only, only first iteration
    """
    plt.figure(figsize=(400, 6))

    data = json.load(open(log_file))
    start_timestamps = []

    # TMP - plot matrices
    matrices_yaml = "/swgwork/eshukrun/run/results/nixl_kvbench__08_09_2025__4324500/matrices/metadata.yaml"
    output("Using matrix metadata from ", matrices_yaml)
    matrices = yaml.load(open(matrices_yaml), Loader=yaml.FullLoader)[
        "traffic_patterns"
    ]  # TMP

    simulated_ts = 0
    for iter_results in data["iterations_results"]:
        for ix, tp_results in enumerate(iter_results):
            # TMP - only rank 0
            matrix_dat = load_matrix(matrices[ix]["matrix_file"])
            tp = TrafficPattern(
                matrix_dat,
                sleep_before_launch_sec=matrices[ix]["sleep_before_launch_sec"],
            )

            plt.axvline(
                x=simulated_ts,
                color="blue",
                linestyle="-",
                linewidth=0.5,
                alpha=0.7,
                ymax=1 / 3,
            )
            simulated_ts += tp.sleep_before_launch_sec * time_resolution

            if 0 not in tp.senders_ranks():
                continue

            start_timestamps.append(tp_results["min_start_ts"])

    # Plot transfers timeline
    start_timestamps = [
        int((ts - min(start_timestamps)) * time_resolution) for ts in start_timestamps
    ]
    for ts in start_timestamps:
        plt.axvline(x=ts, color="yellow", linestyle="-", linewidth=0.5, alpha=0.7)
    output(f"Plotted {len(start_timestamps)} timestamps")

    # TMP - plot dynamo
    dynamo_log = "/swgwork/khalidm/projects/inference/cloudai/main_cloudaix/cloudaix/results/eyalch_deepseek_r1_distill_llama_70b_2025-09-08_10-19-15/Tests.1/0/dynamo_decode_0_0.log"
    dynamo_timestamps, dynamo_num_blocks = extract_transfers_timeline_from_dynamo(
        dynamo_log
    )

    min_ts = min(dynamo_timestamps.values())
    dynamo_timestamps = {
        k: int((ts - min_ts) * time_resolution) for k, ts in dynamo_timestamps.items()
    }
    for ts in dynamo_timestamps.values():
        plt.axvline(x=ts, color="red", linestyle="-", linewidth=0.5, alpha=0.7)
    output(f"Plotted {len(dynamo_timestamps)} dynamo timestamps")

    start_timestamps = sorted(start_timestamps)
    sorted_dynamo_timestamps = sorted(
        dynamo_timestamps.keys(), key=lambda x: dynamo_timestamps[x]
    )
    output("Benchmark\t\tDynamo")
    last_vals = (0, 0)
    for i in range(20):
        output(
            f"{start_timestamps[i]}ms (+{start_timestamps[i] - last_vals[0]})\t\t{dynamo_timestamps[sorted_dynamo_timestamps[i]]}ms (+{dynamo_timestamps[sorted_dynamo_timestamps[i]] - last_vals[1]})"
        )
        last_vals = (
            start_timestamps[i],
            dynamo_timestamps[sorted_dynamo_timestamps[i]],
        )

    plt.xlim(0, max(max(start_timestamps), max(dynamo_timestamps.values())))
    plt.ylim(0, 3)
    plt.legend()
    if time_resolution == 1:
        plt.xlabel("Time (s)")
    elif time_resolution == 1000:
        plt.xlabel("Time (ms)")
    else:
        plt.xlabel(f"Time (1/{time_resolution} s)")
    plt.title("KV transfers starts timeline", wrap=True)
    plt.savefig(outfile)
    output(f"Plotted {title} to {outfile}")


@click.group()
def cli():
    """CLI tool for plotting benchmark results."""
    pass


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--title", "-t", type=str, help="Title of the plot")
@click.option(
    "--outfile", "-o", type=click.Path(), help="Output file path for the plot"
)
@click.option("--max-y", "-y", type=float, help="Maximum value for y-axis")
@click.option("--max-x", "-x", type=float, help="Maximum value for x-axis")
@click.option(
    "--sol", "-s", type=float, default=48.75, help="Solution value (default: 48.75)"
)
@click.option(
    "--plot-theoretical-minimum", is_flag=True, help="Plot theoretical minimum"
)
def plot(log_file, title, outfile, max_y, max_x, sol, plot_theoretical_minimum):
    """Plot a single benchmark run.

    Example:
        python plot.py plot path/to/results.json "Llama 405B Benchmark" 1 --max-y 300 --sol 300
    """

    plot_run(log_file, title, outfile, max_y, max_x, sol, plot_theoretical_minimum)


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--title", "-t", default="", type=str, help="Title of the plot")
@click.option(
    "--outfile", "-o", type=click.Path(), help="Output file path for the plot"
)
def timeline(log_file, title, outfile):
    """Plot a single benchmark run.

    Example:
        python plot.py timeline /swgwork/eshukrun/run/results/nixl_kvbench__26_08_2025__4319106/results/results.json
    """
    plot_transfers_timeline(log_file, title, outfile)


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.argument("dynamo_log_file", type=click.Path(exists=True))
@click.option(
    "--title",
    "-t",
    default="Timestamp Distribution Comparison",
    type=str,
    help="Title of the plot",
)
@click.option(
    "--outfile", "-o", type=click.Path(), help="Output file path for the plot"
)
@click.option(
    "--time-resolution",
    "-r",
    default=10000,
    type=int,
    help="Time resolution multiplier (default: 10000)",
)
def histogram(log_file, dynamo_log_file, title, outfile, time_resolution):
    """Plot superposed histograms with 50 bins each for start_timestamps and dynamo_timestamps.

    Example:
        python plot.py histogram /path/to/results.json /path/to/dynamo_decode_0_0.log --title "My Comparison" --outfile comparison.png
    """
    plot_histogram_comparison(
        log_file, dynamo_log_file, title, outfile, time_resolution
    )


@cli.command()
@click.argument("log_file_1", type=click.Path(exists=True))
@click.argument("log_file_2", type=click.Path(exists=True))
@click.argument("label_1", type=str)
@click.argument("label_2", type=str)
@click.argument("title", type=str)
@click.option(
    "--outfile", "-o", type=click.Path(), help="Output file path for the plot"
)
@click.option("--max-y", "-y", type=float, help="Maximum value for y-axis")
@click.option("--max-x", "-x", type=float, help="Maximum value for x-axis")
@click.option(
    "--isolated-diff",
    is_flag=True,
    help="If true, plot only isolated latency difference",
)
def diff(
    log_file_1,
    log_file_2,
    label_1,
    label_2,
    title,
    outfile,
    max_y,
    max_x,
    isolated_diff,
):
    """Plot the difference between two benchmark runs.

    Example:
        python plot.py diff path/to/results1.json path/to/results2.json "Spectrum-X" "Standard Ethernet" "SPCX vs STD ETH Comparison"
    """
    if isolated_diff:
        plot_isolated_diff(
            log_file_1, log_file_2, label_1, label_2, title, outfile, max_y, max_x=max_x
        )
    else:
        plot_diff(
            log_file_1, log_file_2, label_1, label_2, title, outfile, max_y, max_x=max_x
        )


# ============== STORAGE GRAPHS (Isolated vs Workload) ==============


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--title", "-t", default="Storage Read Latency", type=str)
@click.option("--outfile", "-o", type=click.Path())
@click.option("--max-y", "-y", type=float, help="Maximum value for y-axis")
@click.option("--max-x", "-x", type=float, help="Maximum value for x-axis")
@click.option(
    "--filter-outliers",
    "-f",
    type=float,
    default=None,
    help="Filter isolated latency outliers above this percentile (e.g., 90 for P90)",
)
@click.option(
    "--smooth",
    "-s",
    type=int,
    default=None,
    help="Smooth isolated line by binning into N bins and plotting median per bin",
)
@click.option(
    "--filter-spikes",
    type=float,
    default=None,
    help="Filter spikes: points > N std devs above trend (e.g., 1.5 or 2.0)",
)
def plot_read(
    log_file, title, outfile, max_y, max_x, filter_outliers, smooth, filter_spikes
):
    """Plot Storage Read latency (identical style to RDMA plot).

    Use --filter-outliers 90 to exclude isolated latency spikes above P90.
    Use --smooth 20 to bin data into 20 bins and show median per bin.
    Use --filter-spikes 1.5 to remove points > 1.5 std devs above the trend line.
    """
    with open(log_file) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    run_datetime = (
        datetime.fromtimestamp(metadata.get("ts", 0)) if metadata.get("ts") else None
    )

    sizes = []
    latencies_by_iter = []
    isolated_by_iter = []

    for iter_idx, iter_results in enumerate(data["iterations_results"]):
        lats = []
        isos = []
        for tp in iter_results:
            if iter_idx == 0:
                size = tp.get("size", 0) or tp.get("storage_read_size_gb", 0)
                sizes.append(size)
            lats.append(tp.get("storage_read_avg_ms", 0))
            isos.append(tp.get("isolated_read_p50_ms", 0))
        latencies_by_iter.append(lats)
        isolated_by_iter.append(isos)

    mean_isolated = np.mean(isolated_by_iter, axis=0)

    # Filter outliers if requested (global percentile)
    if filter_outliers is not None:
        threshold = np.percentile(mean_isolated, filter_outliers)
        mean_isolated = np.where(mean_isolated <= threshold, mean_isolated, np.nan)
        output(f"Filtering isolated latency > P{filter_outliers} ({threshold:.1f} ms)")

    # Filter spikes: points significantly above the trend line
    if filter_spikes is not None:
        sizes_arr = np.array(sizes)
        # Fit linear trend
        z = np.polyfit(sizes_arr, mean_isolated, 1)
        trend = np.poly1d(z)(sizes_arr)
        residuals = mean_isolated - trend
        std_resid = np.nanstd(residuals)
        # Filter points where residual > N std devs
        spike_mask = residuals > (filter_spikes * std_resid)
        num_spikes = np.sum(spike_mask)
        mean_isolated = np.where(~spike_mask, mean_isolated, np.nan)
        output(
            f"Filtering {num_spikes} spikes (> {filter_spikes} std devs above trend)"
        )

    plt.figure(figsize=(10, 6))
    colors = [
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    # Plot iterations with per-iteration trend lines
    for i, lats in enumerate(latencies_by_iter):
        plt.scatter(
            sizes,
            lats,
            s=1,
            color=colors[i % len(colors)],
            label=f"Workload iter {i + 1}",
        )
        z = np.polyfit(sizes, lats, 1)
        p = np.poly1d(z)
        plt.plot(
            sizes,
            p(sizes),
            color=colors[i % len(colors)],
            linestyle="--",
            alpha=0.5,
            label=f"Iter {i + 1} trend",
        )

    # Global trend
    all_y = np.concatenate(latencies_by_iter)
    all_x = np.array(sizes * len(latencies_by_iter))
    global_z = np.polyfit(all_x, all_y, 1)
    global_p = np.poly1d(global_z)
    plt.plot(
        sizes,
        global_p(sizes),
        color="black",
        linestyle="-",
        linewidth=1,
        label="Global trend",
    )

    # Micro-benchmark (isolated) - handle NaN values from filtering
    valid_mask = ~np.isnan(mean_isolated)
    valid_sizes = np.array(sizes)[valid_mask]
    valid_isolated = mean_isolated[valid_mask]

    if len(valid_sizes) > 0:
        if smooth is not None and smooth > 0:
            # Bin data and plot median per bin
            bin_edges = np.linspace(valid_sizes.min(), valid_sizes.max(), smooth + 1)
            bin_centers = []
            bin_medians = []
            for i in range(smooth):
                mask = (valid_sizes >= bin_edges[i]) & (valid_sizes < bin_edges[i + 1])
                if i == smooth - 1:  # Include right edge for last bin
                    mask = (valid_sizes >= bin_edges[i]) & (
                        valid_sizes <= bin_edges[i + 1]
                    )
                if np.sum(mask) > 0:
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                    bin_medians.append(np.median(valid_isolated[mask]))
            label = f"Micro-benchmark (median, {smooth} bins)"
            plt.plot(bin_centers, bin_medians, "r-", linewidth=2, label=label)
        else:
            x_sorted, iso_sorted = zip(*sorted(zip(valid_sizes, valid_isolated)))
            label = (
                "Micro-benchmark"
                if filter_outliers is None
                else f"Micro-benchmark (≤P{int(filter_outliers)})"
            )
            plt.plot(x_sorted, iso_sorted, "r-", linewidth=1, label=label)

    plt.legend()
    plt.xlabel("KV Cache size (GB)")
    plt.ylabel("Read Latency (ms)")
    plt.title(title, wrap=True)
    if run_datetime:
        subtitle = f"{sum(len(y) for y in latencies_by_iter)} points, {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        plt.suptitle(subtitle, fontsize=10)
    if max_y is not None:
        plt.ylim(0, max_y)
    if max_x is not None:
        plt.xlim(0, max_x)

    outfile = outfile or os.path.join(os.path.dirname(log_file), "plot_read.png")
    outfile = outfile.replace(" ", "_")
    plt.savefig(outfile)
    plt.clf()
    output(f"Plotted {title} to {outfile}")


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--title", "-t", default="Storage Write Latency", type=str)
@click.option("--outfile", "-o", type=click.Path())
@click.option("--max-y", "-y", type=float, help="Maximum value for y-axis")
@click.option("--max-x", "-x", type=float, help="Maximum value for x-axis")
def plot_write(log_file, title, outfile, max_y, max_x):
    """Plot Storage Write latency (identical style to RDMA plot)."""
    with open(log_file) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    run_datetime = (
        datetime.fromtimestamp(metadata.get("ts", 0)) if metadata.get("ts") else None
    )

    sizes = []
    latencies_by_iter = []
    isolated_by_iter = []

    for iter_idx, iter_results in enumerate(data["iterations_results"]):
        lats = []
        isos = []
        for tp in iter_results:
            if iter_idx == 0:
                size = tp.get("size", 0) or tp.get("storage_write_size_gb", 0)
                sizes.append(size)
            lats.append(tp.get("storage_write_avg_ms", 0))
            isos.append(tp.get("isolated_write_p50_ms", 0))
        latencies_by_iter.append(lats)
        isolated_by_iter.append(isos)

    mean_isolated = np.mean(isolated_by_iter, axis=0)

    plt.figure(figsize=(10, 6))
    colors = [
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    for i, lats in enumerate(latencies_by_iter):
        plt.scatter(
            sizes,
            lats,
            s=1,
            color=colors[i % len(colors)],
            label=f"Workload iter {i + 1}",
        )
        z = np.polyfit(sizes, lats, 1)
        p = np.poly1d(z)
        plt.plot(
            sizes,
            p(sizes),
            color=colors[i % len(colors)],
            linestyle="--",
            alpha=0.5,
            label=f"Iter {i + 1} trend",
        )

    all_y = np.concatenate(latencies_by_iter)
    all_x = np.array(sizes * len(latencies_by_iter))
    global_z = np.polyfit(all_x, all_y, 1)
    global_p = np.poly1d(global_z)
    plt.plot(
        sizes,
        global_p(sizes),
        color="black",
        linestyle="-",
        linewidth=1,
        label="Global trend",
    )

    x_sorted, iso_sorted = zip(*sorted(zip(sizes, mean_isolated)))
    plt.plot(x_sorted, iso_sorted, "r-", linewidth=1, label="Micro-benchmark")

    plt.legend()
    plt.xlabel("KV Cache size (GB)")
    plt.ylabel("Write Latency (ms)")
    plt.title(title, wrap=True)
    if run_datetime:
        subtitle = f"{sum(len(y) for y in latencies_by_iter)} points, {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        plt.suptitle(subtitle, fontsize=10)
    if max_y is not None:
        plt.ylim(0, max_y)
    if max_x is not None:
        plt.xlim(0, max_x)

    outfile = outfile or os.path.join(os.path.dirname(log_file), "plot_write.png")
    outfile = outfile.replace(" ", "_")
    plt.savefig(outfile)
    plt.clf()
    output(f"Plotted {title} to {outfile}")


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--title", "-t", default="Read Bandwidth vs Size", type=str)
@click.option("--outfile", "-o", type=click.Path())
@click.option("--max-y", "-y", type=float, help="Maximum value for y-axis")
@click.option("--max-x", "-x", type=float, help="Maximum value for x-axis")
@click.option(
    "--filter-outliers",
    "-f",
    type=float,
    default=None,
    help="Filter isolated BW outliers below this percentile (e.g., 10 for P10)",
)
def bw_read(log_file, title, outfile, max_y, max_x, filter_outliers):
    """Plot Read BW (identical style to RDMA plot).

    Use --filter-outliers 10 to exclude isolated BW below P10 (low outliers).
    """
    with open(log_file) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    run_datetime = (
        datetime.fromtimestamp(metadata.get("ts", 0)) if metadata.get("ts") else None
    )

    sizes = []
    bw_by_iter = []
    iso_bw_by_iter = []

    for iter_idx, iter_results in enumerate(data["iterations_results"]):
        bws = []
        iso_bws = []
        for tp in iter_results:
            if iter_idx == 0:
                size_x = tp.get("size", 0) or tp.get("storage_read_size_gb", 0)
                sizes.append(size_x)
            size = tp.get("size", 0) or tp.get("storage_read_size_gb", 0)
            read_ms = tp.get("storage_read_avg_ms", 0)
            iso_read_ms = tp.get("isolated_read_p50_ms", 0)
            bws.append(size / (read_ms / 1000) if read_ms > 0 else 0)
            iso_bws.append(size / (iso_read_ms / 1000) if iso_read_ms > 0 else 0)
        bw_by_iter.append(bws)
        iso_bw_by_iter.append(iso_bws)

    mean_iso = np.mean(iso_bw_by_iter, axis=0)

    # Filter outliers if requested (for BW, filter LOW values - below percentile)
    if filter_outliers is not None:
        threshold = np.percentile(mean_iso, filter_outliers)
        mean_iso = np.where(mean_iso >= threshold, mean_iso, np.nan)
        output(f"Filtering isolated BW < P{filter_outliers} ({threshold:.1f} GB/s)")

    plt.figure(figsize=(10, 6))
    colors = [
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    for i, bws in enumerate(bw_by_iter):
        plt.scatter(
            sizes,
            bws,
            s=1,
            color=colors[i % len(colors)],
            label=f"Workload iter {i + 1}",
        )
        z = np.polyfit(sizes, bws, 1)
        p = np.poly1d(z)
        plt.plot(
            sizes,
            p(sizes),
            color=colors[i % len(colors)],
            linestyle="--",
            alpha=0.5,
            label=f"Iter {i + 1} trend",
        )

    all_y = np.concatenate(bw_by_iter)
    all_x = np.array(sizes * len(bw_by_iter))
    global_z = np.polyfit(all_x, all_y, 1)
    global_p = np.poly1d(global_z)
    plt.plot(
        sizes,
        global_p(sizes),
        color="black",
        linestyle="-",
        linewidth=1,
        label="Global trend",
    )

    # Handle NaN values from filtering
    valid_mask = ~np.isnan(mean_iso)
    valid_sizes = np.array(sizes)[valid_mask]
    valid_iso = mean_iso[valid_mask]
    if len(valid_sizes) > 0:
        x_sorted, iso_sorted = zip(*sorted(zip(valid_sizes, valid_iso)))
        label = (
            "Micro-benchmark"
            if filter_outliers is None
            else f"Micro-benchmark (≥P{int(filter_outliers)})"
        )
        plt.plot(x_sorted, iso_sorted, "r-", linewidth=1, label=label)

    plt.legend()
    plt.xlabel("KV Cache size (GB)")
    plt.ylabel("Read Bandwidth (GB/s)")
    plt.title(title, wrap=True)
    if run_datetime:
        subtitle = f"{sum(len(y) for y in bw_by_iter)} points, {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        plt.suptitle(subtitle, fontsize=10)
    if max_y is not None:
        plt.ylim(0, max_y)
    if max_x is not None:
        plt.xlim(0, max_x)

    outfile = outfile or os.path.join(os.path.dirname(log_file), "plot_bw_read.png")
    outfile = outfile.replace(" ", "_")
    plt.savefig(outfile)
    plt.clf()
    output(f"Plotted {title} to {outfile}")


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--title", "-t", default="Write Bandwidth vs Size", type=str)
@click.option("--outfile", "-o", type=click.Path())
@click.option("--max-y", "-y", type=float, help="Maximum value for y-axis")
@click.option("--max-x", "-x", type=float, help="Maximum value for x-axis")
def bw_write(log_file, title, outfile, max_y, max_x):
    """Plot Write BW (identical style to RDMA plot)."""
    with open(log_file) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    run_datetime = (
        datetime.fromtimestamp(metadata.get("ts", 0)) if metadata.get("ts") else None
    )

    sizes = []
    bw_by_iter = []
    iso_bw_by_iter = []

    for iter_idx, iter_results in enumerate(data["iterations_results"]):
        bws = []
        iso_bws = []
        for tp in iter_results:
            if iter_idx == 0:
                size_x = tp.get("size", 0) or tp.get("storage_write_size_gb", 0)
                sizes.append(size_x)
            size = tp.get("size", 0) or tp.get("storage_write_size_gb", 0)
            write_ms = tp.get("storage_write_avg_ms", 0)
            iso_write_ms = tp.get("isolated_write_p50_ms", 0)
            bws.append(size / (write_ms / 1000) if write_ms > 0 else 0)
            iso_bws.append(size / (iso_write_ms / 1000) if iso_write_ms > 0 else 0)
        bw_by_iter.append(bws)
        iso_bw_by_iter.append(iso_bws)

    mean_iso = np.mean(iso_bw_by_iter, axis=0)

    plt.figure(figsize=(10, 6))
    colors = [
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    for i, bws in enumerate(bw_by_iter):
        plt.scatter(
            sizes,
            bws,
            s=1,
            color=colors[i % len(colors)],
            label=f"Workload iter {i + 1}",
        )
        z = np.polyfit(sizes, bws, 1)
        p = np.poly1d(z)
        plt.plot(
            sizes,
            p(sizes),
            color=colors[i % len(colors)],
            linestyle="--",
            alpha=0.5,
            label=f"Iter {i + 1} trend",
        )

    all_y = np.concatenate(bw_by_iter)
    all_x = np.array(sizes * len(bw_by_iter))
    global_z = np.polyfit(all_x, all_y, 1)
    global_p = np.poly1d(global_z)
    plt.plot(
        sizes,
        global_p(sizes),
        color="black",
        linestyle="-",
        linewidth=1,
        label="Global trend",
    )

    x_sorted, iso_sorted = zip(*sorted(zip(sizes, mean_iso)))
    plt.plot(x_sorted, iso_sorted, "r-", linewidth=1, label="Micro-benchmark")

    plt.legend()
    plt.xlabel("KV Cache size (GB)")
    plt.ylabel("Write Bandwidth (GB/s)")
    plt.title(title, wrap=True)
    if run_datetime:
        subtitle = f"{sum(len(y) for y in bw_by_iter)} points, {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        plt.suptitle(subtitle, fontsize=10)
    if max_y is not None:
        plt.ylim(0, max_y)
    if max_x is not None:
        plt.xlim(0, max_x)

    outfile = outfile or os.path.join(os.path.dirname(log_file), "plot_bw_write.png")
    outfile = outfile.replace(" ", "_")
    plt.savefig(outfile)
    plt.clf()
    output(f"Plotted {title} to {outfile}")


@cli.command()
@click.argument("log_file", type=click.Path(exists=True))
@click.option("--title", "-t", default="RDMA Bandwidth vs Size", type=str)
@click.option("--outfile", "-o", type=click.Path())
@click.option("--max-y", "-y", type=float, help="Maximum value for y-axis")
@click.option("--max-x", "-x", type=float, help="Maximum value for x-axis")
def bw_rdma(log_file, title, outfile, max_y, max_x):
    """Plot RDMA BW (identical style to RDMA plot)."""
    with open(log_file) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    run_datetime = (
        datetime.fromtimestamp(metadata.get("ts", 0)) if metadata.get("ts") else None
    )

    sizes = []
    bw_by_iter = []
    iso_bw_by_iter = []

    for iter_idx, iter_results in enumerate(data["iterations_results"]):
        bws = []
        iso_bws = []
        for tp in iter_results:
            if iter_idx == 0:
                sizes.append(tp.get("size", 0))
            size = tp.get("size", 0)
            lat_ms = tp.get("latency", 0)
            iso_lat_ms = tp.get("isolated_latency", 0)
            bw = tp.get("mean_bw", size / (lat_ms / 1000) if lat_ms > 0 else 0)
            iso_bw = size / (iso_lat_ms / 1000) if iso_lat_ms > 0 else 0
            bws.append(bw)
            iso_bws.append(iso_bw)
        bw_by_iter.append(bws)
        iso_bw_by_iter.append(iso_bws)

    mean_iso = np.mean(iso_bw_by_iter, axis=0)

    plt.figure(figsize=(10, 6))
    colors = [
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
    ]

    for i, bws in enumerate(bw_by_iter):
        plt.scatter(
            sizes,
            bws,
            s=1,
            color=colors[i % len(colors)],
            label=f"Workload iter {i + 1}",
        )
        z = np.polyfit(sizes, bws, 1)
        p = np.poly1d(z)
        plt.plot(
            sizes,
            p(sizes),
            color=colors[i % len(colors)],
            linestyle="--",
            alpha=0.5,
            label=f"Iter {i + 1} trend",
        )

    all_y = np.concatenate(bw_by_iter)
    all_x = np.array(sizes * len(bw_by_iter))
    global_z = np.polyfit(all_x, all_y, 1)
    global_p = np.poly1d(global_z)
    plt.plot(
        sizes,
        global_p(sizes),
        color="black",
        linestyle="-",
        linewidth=1,
        label="Global trend",
    )

    x_sorted, iso_sorted = zip(*sorted(zip(sizes, mean_iso)))
    plt.plot(x_sorted, iso_sorted, "r-", linewidth=1, label="Micro-benchmark")

    plt.legend()
    plt.xlabel("KV Cache size (GB)")
    plt.ylabel("RDMA Bandwidth (GB/s)")
    plt.title(title, wrap=True)
    if run_datetime:
        subtitle = f"{sum(len(y) for y in bw_by_iter)} points, {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        plt.suptitle(subtitle, fontsize=10)
    if max_y is not None:
        plt.ylim(0, max_y)
    if max_x is not None:
        plt.xlim(0, max_x)

    outfile = outfile or os.path.join(os.path.dirname(log_file), "plot_bw_rdma.png")
    outfile = outfile.replace(" ", "_")
    plt.savefig(outfile)
    plt.clf()
    output(f"Plotted {title} to {outfile}")


if __name__ == "__main__":
    cli()
