# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Results Reporter for NIXL EP Tests

Saves test results to CSV with metadata for tracking over time.
"""

import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Default results directory (can be overridden via env var)
DEFAULT_RESULTS_DIR = os.environ.get("NIXL_TEST_RESULTS_DIR", "./results")


def get_git_info(repo_path: str) -> Dict[str, str]:
    """
    Get git repository information.

    Returns:
        {
            'repo': 'nixl_ep',
            'branch': 'main',
            'sha': 'abc1234',
            'sha_short': 'abc1234',
            'commit_msg': 'Fix bug in dispatch',
            'commit_date': '2025-12-01 10:30:00',
            'dirty': False
        }
    """
    try:
        cwd = os.getcwd()
        os.chdir(repo_path)

        # Get repo name from remote or directory
        try:
            remote = (
                subprocess.check_output(
                    ["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            repo_name = remote.split("/")[-1].replace(".git", "")
        except:
            repo_name = os.path.basename(repo_path)

        # Get branch
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )

        # Get SHA
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

        sha_short = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )

        # Get commit message (first line)
        commit_msg = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%s"])
            .decode()
            .strip()
        )

        # Get commit date
        commit_date = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%ci"])
            .decode()
            .strip()
        )

        # Check if dirty
        dirty = (
            subprocess.call(
                ["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL
            )
            != 0
        )

        os.chdir(cwd)

        return {
            "repo": repo_name,
            "branch": branch,
            "sha": sha,
            "sha_short": sha_short,
            "commit_msg": commit_msg,
            "commit_date": commit_date,
            "dirty": dirty,
        }
    except Exception as e:
        os.chdir(cwd)
        return {
            "repo": "unknown",
            "branch": "unknown",
            "sha": "unknown",
            "sha_short": "unknown",
            "commit_msg": f"Error: {e}",
            "commit_date": "unknown",
            "dirty": False,
        }


def get_environment_info() -> Dict[str, str]:
    """Get environment information."""
    import platform

    info = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # Try to get CUDA info
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except:
        pass

    # Get SLURM info if available
    slurm_vars = ["SLURM_JOB_ID", "SLURM_NNODES", "SLURM_NTASKS"]
    for var in slurm_vars:
        if var in os.environ:
            info[var.lower()] = os.environ[var]

    return info


class ResultsReporter:
    """
    Reports test results to CSV files.

    Usage:
        reporter = ResultsReporter()
        reporter.add_result(
            test_name='P-CONN-01',
            category='connection',
            metric='latency_ms',
            value=123.45,
            params={'num_ranks': 8, 'backend': 'nixl'}
        )
        reporter.save()
    """

    def __init__(
        self,
        results_dir: str = DEFAULT_RESULTS_DIR,
        nixl_ep_repo: str = None,
        run_id: Optional[str] = None,
    ):
        # Default to the repo containing this file
        if nixl_ep_repo is None:
            nixl_ep_repo = str(Path(__file__).parent.parent.parent.parent.parent)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Generate run ID
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = datetime.now().isoformat()

        # Get metadata
        self.git_info = get_git_info(nixl_ep_repo)
        self.env_info = get_environment_info()

        # Results storage
        self.results: List[Dict[str, Any]] = []

    def add_result(
        self,
        test_name: str,
        category: str,
        metric: str,
        value: float,
        unit: str = "",
        params: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Add a single test result."""
        result = {
            # Identifiers
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "test_name": test_name,
            "category": category,
            # Result
            "metric": metric,
            "value": value,
            "unit": unit,
            # Git info
            "repo": self.git_info["repo"],
            "branch": self.git_info["branch"],
            "sha_short": self.git_info["sha_short"],
            "commit_msg": self.git_info["commit_msg"][:80],  # Truncate
            "dirty": self.git_info["dirty"],
            # Environment
            "hostname": self.env_info.get("hostname", ""),
            "gpu_count": self.env_info.get("gpu_count", ""),
            "slurm_job_id": self.env_info.get("slurm_job_id", ""),
            "slurm_nnodes": self.env_info.get("slurm_nnodes", ""),
        }

        # Add parameters as columns
        if params:
            for k, v in params.items():
                result[f"param_{k}"] = v

        # Add extra fields
        if extra:
            for k, v in extra.items():
                result[f"extra_{k}"] = v

        self.results.append(result)

    def add_benchmark_results(
        self, benchmark_json_path: str, category: str = "benchmark"
    ):
        """
        Import results from pytest-benchmark JSON output.

        Args:
            benchmark_json_path: Path to benchmark JSON file
            category: Category to assign to these results
        """
        with open(benchmark_json_path) as f:
            data = json.load(f)

        for bench in data.get("benchmarks", []):
            test_name = bench.get("name", "unknown")
            stats = bench.get("stats", {})

            # Add main metrics
            for metric in ["mean", "median", "min", "max", "stddev"]:
                if metric in stats:
                    self.add_result(
                        test_name=test_name,
                        category=category,
                        metric=metric,
                        value=stats[metric],
                        unit="seconds",
                        params=bench.get("params", {}),
                        extra={"rounds": stats.get("rounds", 0)},
                    )

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save results to CSV.

        Returns:
            Path to saved CSV file
        """
        if not self.results:
            return ""

        # Generate filename
        if filename is None:
            filename = f"results_{self.run_id}.csv"

        filepath = self.results_dir / filename

        # Get all unique keys across all results
        all_keys = set()
        for r in self.results:
            all_keys.update(r.keys())

        # Sort keys for consistent column order
        # Priority columns first, then alphabetical
        priority = [
            "run_id",
            "timestamp",
            "test_name",
            "category",
            "metric",
            "value",
            "unit",
            "repo",
            "branch",
            "sha_short",
            "commit_msg",
        ]
        other_keys = sorted(all_keys - set(priority))
        fieldnames = [k for k in priority if k in all_keys] + other_keys

        # Write CSV
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.results)

        logger.info("Results saved to: %s", filepath)
        return str(filepath)

    def save_summary(self, filename: Optional[str] = None) -> str:
        """Save a human-readable summary."""
        if filename is None:
            filename = f"summary_{self.run_id}.txt"

        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("NIXL EP Test Results Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")

            f.write("Git Info:\n")
            f.write(f"  Repo: {self.git_info['repo']}\n")
            f.write(f"  Branch: {self.git_info['branch']}\n")
            f.write(f"  SHA: {self.git_info['sha']}\n")
            f.write(f"  Commit: {self.git_info['commit_msg']}\n")
            f.write(f"  Dirty: {self.git_info['dirty']}\n\n")

            f.write("Environment:\n")
            for k, v in self.env_info.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

            f.write("Results:\n")
            f.write("-" * 60 + "\n")
            for r in self.results:
                f.write(
                    f"{r['test_name']}: {r['metric']} = {r['value']} {r.get('unit', '')}\n"
                )

        logger.info("Summary saved to: %s", filepath)
        return str(filepath)


# =============================================================================
# Pytest plugin integration
# =============================================================================

# Global reporter instance (initialized by conftest.py)
_reporter: Optional[ResultsReporter] = None


def get_reporter() -> ResultsReporter:
    """Get or create the global reporter."""
    global _reporter
    if _reporter is None:
        _reporter = ResultsReporter()
    return _reporter


def init_reporter(**kwargs) -> ResultsReporter:
    """Initialize the global reporter with custom settings."""
    global _reporter
    _reporter = ResultsReporter(**kwargs)
    return _reporter


# =============================================================================
# CLI for standalone use
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIXL EP Results Reporter")
    parser.add_argument(
        "--import-benchmark", type=str, help="Import pytest-benchmark JSON"
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    reporter = ResultsReporter(results_dir=args.output_dir)

    if args.import_benchmark:
        reporter.add_benchmark_results(args.import_benchmark)
        reporter.save()
        reporter.save_summary()
    else:
        # Demo
        reporter.add_result(
            test_name="P-CONN-01",
            category="connection",
            metric="latency_ms",
            value=123.45,
            unit="ms",
            params={"num_ranks": 8, "backend": "nixl"},
        )
        reporter.save()
        reporter.save_summary()
