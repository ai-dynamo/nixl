#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL EP Performance Results Collector

Collects, stores, and exports performance test results in a CI/CD-friendly format.

Features:
- JSON storage for individual runs (machine-readable)
- CSV/Excel export for analysis
- Metadata capture (git, hardware, timestamp)
- Incremental results aggregation
- CI/CD integration ready

Usage:
    # From test scripts:
    from results_collector import ResultsCollector
    collector = ResultsCollector()
    collector.record_result("data_plane", "dispatch", config, metrics)
    collector.save()

    # CLI: Export to CSV
    python results_collector.py export --format csv --output results.csv

    # CLI: View recent results
    python results_collector.py list --last 10
"""

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class TestMetadata:
    """Metadata for a test run"""

    timestamp: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    hostname: str
    num_gpus: int
    cuda_version: str
    test_suite_version: str = "1.0.0"
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            # Generate unique run ID from timestamp + commit
            hash_input = f"{self.timestamp}_{self.git_commit}_{self.hostname}"
            self.run_id = hashlib.sha256(hash_input.encode()).hexdigest()[:12]


@dataclass
class TestConfig:
    """Configuration for a test"""

    test_type: str  # "control_plane" or "data_plane"
    test_name: str  # "init", "connect", "dispatch", "e2e", etc.
    num_processes: int
    num_tokens: int
    hidden_dim: int
    num_experts_per_rank: int
    topk: int
    nvlink_backend: str
    num_nodes: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Results from a single test"""

    metadata: TestMetadata
    config: TestConfig
    metrics: Dict[str, float]  # metric_name -> value
    passed: bool
    error_message: str = ""
    raw_output: str = ""


# ============================================================================
# Helper Functions
# ============================================================================


def get_git_info() -> Tuple[str, str, bool]:
    """Get current git commit, branch, and dirty status"""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()[:12]
        )
    except Exception:
        commit = "unknown"

    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        branch = "unknown"

    try:
        # Check if working directory is dirty
        result = subprocess.run(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL)
        dirty = result.returncode != 0
    except Exception:
        dirty = False

    return commit, branch, dirty


def get_cuda_version() -> str:
    """Get CUDA version"""
    try:
        import torch

        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


def get_num_gpus() -> int:
    """Get number of GPUs"""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def create_metadata() -> TestMetadata:
    """Create metadata for current environment"""
    commit, branch, dirty = get_git_info()
    return TestMetadata(
        timestamp=datetime.utcnow().isoformat() + "Z",
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        hostname=socket.gethostname(),
        num_gpus=get_num_gpus(),
        cuda_version=get_cuda_version(),
    )


# ============================================================================
# Results Collector
# ============================================================================


class ResultsCollector:
    """
    Collects and manages performance test results.

    Results are stored as JSON files in results/raw/ directory.
    Can export to CSV for analysis.

    Directory priority:
    1. Explicit results_dir parameter
    2. NIXL_RESULTS_DIR environment variable
    3. Default: tests/perf/results/
    """

    def __init__(self, results_dir: str | None = None):
        results_dir_str = results_dir
        if results_dir_str is None:
            # Check environment variable
            results_dir_str = os.environ.get("NIXL_RESULTS_DIR")

        if results_dir_str is None:
            # Default to tests/perf/results/
            self.results_dir: Path = Path(__file__).parent / "results"
        else:
            self.results_dir = Path(results_dir_str)

        self.raw_dir = self.results_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = create_metadata()
        self.results: List[TestResult] = []

    def record_result(
        self,
        test_type: str,
        test_name: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        passed: bool = True,
        error_message: str = "",
        raw_output: str = "",
    ):
        """
        Record a test result.

        Args:
            test_type: "control_plane" or "data_plane"
            test_name: e.g., "init", "connect", "dispatch", "e2e"
            config: Test configuration dict
            metrics: Dict of metric_name -> value
            passed: Whether test passed
            error_message: Error message if failed
            raw_output: Raw test output for debugging
        """
        test_config = TestConfig(
            test_type=test_type,
            test_name=test_name,
            num_processes=config.get("num_processes", 8),
            num_tokens=config.get("num_tokens", 512),
            hidden_dim=config.get("hidden_dim", 7168),
            num_experts_per_rank=config.get("num_experts_per_rank", 8),
            topk=config.get("topk", 2),
            nvlink_backend=config.get("nvlink_backend", "ipc"),
            num_nodes=config.get("num_nodes", 1),
            extra={
                k: v
                for k, v in config.items()
                if k
                not in [
                    "num_processes",
                    "num_tokens",
                    "hidden_dim",
                    "num_experts_per_rank",
                    "topk",
                    "nvlink_backend",
                    "num_nodes",
                ]
            },
        )

        result = TestResult(
            metadata=self.metadata,
            config=test_config,
            metrics=metrics,
            passed=passed,
            error_message=error_message,
            raw_output=raw_output,
        )

        self.results.append(result)
        return result

    def save(self, filename: str | None = None) -> str:
        """
        Save results to JSON file.

        Returns:
            Path to saved file
        """
        if filename is None:
            # Generate filename from timestamp and run_id
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{self.metadata.run_id}.json"

        filepath = self.raw_dir / filename

        # Convert results to serializable format
        data = {
            "metadata": asdict(self.metadata),
            "results": [
                {
                    "config": asdict(r.config),
                    "metrics": r.metrics,
                    "passed": r.passed,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        sys.stderr.write(
            f"[ResultsCollector] Saved {len(self.results)} results to {filepath}\n"
        )
        return str(filepath)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected results"""
        return {
            "run_id": self.metadata.run_id,
            "timestamp": self.metadata.timestamp,
            "num_results": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
        }


# ============================================================================
# Results Aggregator (for CI/CD)
# ============================================================================


class ResultsAggregator:
    """
    Aggregates results from multiple runs for analysis and CI/CD.
    """

    def __init__(self, results_dir: str | None = None):
        if results_dir is None:
            self.results_dir: Path = Path(__file__).parent / "results"
        else:
            self.results_dir = Path(results_dir)

        self.raw_dir = self.results_dir / "raw"

    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all results from raw directory"""
        results: List[Dict[str, Any]] = []

        if not self.raw_dir.exists():
            return results

        for filepath in sorted(self.raw_dir.glob("*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    # Flatten for easier analysis
                    for result in data.get("results", []):
                        flat = {
                            # Metadata
                            "run_id": data["metadata"]["run_id"],
                            "timestamp": data["metadata"]["timestamp"],
                            "git_commit": data["metadata"]["git_commit"],
                            "git_branch": data["metadata"]["git_branch"],
                            "hostname": data["metadata"]["hostname"],
                            "num_gpus": data["metadata"]["num_gpus"],
                            # Config
                            "test_type": result["config"]["test_type"],
                            "test_name": result["config"]["test_name"],
                            "num_processes": result["config"]["num_processes"],
                            "num_tokens": result["config"]["num_tokens"],
                            "hidden_dim": result["config"]["hidden_dim"],
                            "num_experts_per_rank": result["config"][
                                "num_experts_per_rank"
                            ],
                            "topk": result["config"]["topk"],
                            "nvlink_backend": result["config"]["nvlink_backend"],
                            "num_nodes": result["config"]["num_nodes"],
                            # Status
                            "passed": result["passed"],
                            "error_message": result.get("error_message", ""),
                        }
                        # Add all metrics as separate columns
                        for metric_name, metric_value in result.get(
                            "metrics", {}
                        ).items():
                            flat[f"metric_{metric_name}"] = metric_value

                        results.append(flat)
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to load {filepath}: {e}\n")

        return results

    def export_csv(
        self, output_path: Union[str, Path] | None = None, append: bool = False
    ) -> str:
        """
        Export all results to CSV.

        Args:
            output_path: Output file path (default: results/history.csv)
            append: If True, append to existing file

        Returns:
            Path to CSV file
        """
        csv_path: Path
        if output_path is None:
            csv_path = self.results_dir / "history.csv"
        else:
            csv_path = Path(output_path)

        results = self.load_all_results()

        if not results:
            sys.stderr.write("No results to export\n")
            return str(csv_path)

        # Get all unique columns
        all_columns: Set[str] = set()
        for r in results:
            all_columns.update(r.keys())

        # Order columns logically
        priority_columns = [
            "timestamp",
            "run_id",
            "git_commit",
            "git_branch",
            "hostname",
            "test_type",
            "test_name",
            "num_processes",
            "num_tokens",
            "hidden_dim",
            "num_experts_per_rank",
            "topk",
            "nvlink_backend",
            "num_nodes",
            "passed",
        ]
        metric_columns = sorted([c for c in all_columns if c.startswith("metric_")])
        other_columns = sorted(
            [
                c
                for c in all_columns
                if c not in priority_columns and c not in metric_columns
            ]
        )

        columns = (
            [c for c in priority_columns if c in all_columns]
            + other_columns
            + metric_columns
        )

        mode = "a" if append and csv_path.exists() else "w"
        write_header = not (append and csv_path.exists())

        with open(csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(results)

        sys.stderr.write(f"Exported {len(results)} results to {csv_path}\n")
        return str(csv_path)

    def list_runs(self, last_n: int | None = None) -> List[Dict[str, Any]]:
        """List all runs with summary"""
        runs: List[Dict[str, Any]] = []

        if not self.raw_dir.exists():
            return runs

        for filepath in sorted(self.raw_dir.glob("*.json"), reverse=True):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    num_passed = sum(1 for r in data["results"] if r["passed"])
                    num_total = len(data["results"])
                    runs.append(
                        {
                            "file": filepath.name,
                            "run_id": data["metadata"]["run_id"],
                            "timestamp": data["metadata"]["timestamp"],
                            "git_commit": data["metadata"]["git_commit"],
                            "hostname": data["metadata"]["hostname"],
                            "results": f"{num_passed}/{num_total} passed",
                        }
                    )
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to load {filepath}: {e}\n")

        if last_n:
            runs = runs[:last_n]

        return runs

    def get_latest_metrics(
        self, test_type: str | None = None, test_name: str | None = None
    ) -> Dict[str, Any]:
        """Get latest metrics for comparison"""
        results = self.load_all_results()

        if test_type:
            results = [r for r in results if r["test_type"] == test_type]
        if test_name:
            results = [r for r in results if r["test_name"] == test_name]

        if not results:
            return {}

        # Sort by timestamp and get latest
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results[0]


# ============================================================================
# CI/CD Integration
# ============================================================================


def ci_check_regression(
    baseline_commit: str | None = None,
    threshold_percent: float = 10.0,
    results_dir: str | None = None,
) -> bool:
    """
    Check for performance regression against baseline.

    Returns True if no regression detected, False otherwise.
    Suitable for CI/CD pipelines.
    """
    agg = ResultsAggregator(results_dir)
    results = agg.load_all_results()

    if not results:
        sys.stderr.write("No results to check\n")
        return True

    # Group by test
    from collections import defaultdict

    by_test = defaultdict(list)
    for r in results:
        key = (
            r["test_type"],
            r["test_name"],
            r["num_tokens"],
            r["topk"],
            r["nvlink_backend"],
        )
        by_test[key].append(r)

    regressions = []

    for key, test_results in by_test.items():
        # Sort by timestamp
        test_results.sort(key=lambda x: x["timestamp"])

        if len(test_results) < 2:
            continue

        latest = test_results[-1]

        # Find baseline (specific commit or previous run)
        if baseline_commit:
            baseline = next(
                (r for r in test_results if r["git_commit"] == baseline_commit), None
            )
        else:
            baseline = test_results[-2]  # Previous run

        if not baseline:
            continue

        # Compare metrics
        for metric_key in latest:
            if not metric_key.startswith("metric_"):
                continue

            latest_val = latest.get(metric_key)
            baseline_val = baseline.get(metric_key)

            if latest_val is None or baseline_val is None or baseline_val == 0:
                continue

            # For throughput/bandwidth, higher is better
            # For latency, lower is better
            is_latency = (
                "latency" in metric_key or "_ms" in metric_key or "_us" in metric_key
            )

            if is_latency:
                change = (latest_val - baseline_val) / baseline_val * 100
                is_regression = change > threshold_percent
            else:
                change = (baseline_val - latest_val) / baseline_val * 100
                is_regression = change > threshold_percent

            if is_regression:
                regressions.append(
                    {
                        "test": key,
                        "metric": metric_key,
                        "baseline": baseline_val,
                        "current": latest_val,
                        "change_percent": change,
                    }
                )

    if regressions:
        sys.stderr.write(f"⚠️  Performance regressions detected ({len(regressions)}):\n")
        for reg in regressions:
            sys.stderr.write(
                f"  - {reg['test']}: {reg['metric']} changed by {reg['change_percent']:.1f}%\n"
            )
            sys.stderr.write(
                f"    Baseline: {reg['baseline']}, Current: {reg['current']}\n"
            )
        return False

    sys.stderr.write("✅ No performance regressions detected\n")
    return True


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="NIXL EP Performance Results Manager")

    # Global arguments
    parser.add_argument(
        "--results-dir",
        "-d",
        type=str,
        default=None,
        help="Results directory (default: NIXL_RESULTS_DIR env or perf/results/)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export results to CSV/Excel")
    export_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.add_argument(
        "--append", action="store_true", help="Append to existing file"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List recent runs")
    list_parser.add_argument(
        "--last", "-n", type=int, default=10, help="Show last N runs"
    )

    # Check command (for CI/CD)
    check_parser = subparsers.add_parser("check", help="Check for regressions (CI/CD)")
    check_parser.add_argument(
        "--baseline", help="Baseline git commit to compare against"
    )
    check_parser.add_argument(
        "--threshold", type=float, default=10.0, help="Regression threshold percentage"
    )

    # Summary command (no arguments needed)
    subparsers.add_parser("summary", help="Show summary of latest run")

    args = parser.parse_args()

    # Determine results directory
    results_dir = args.results_dir or os.environ.get("NIXL_RESULTS_DIR")
    agg = ResultsAggregator(results_dir)

    if args.command == "export":
        output = args.output
        if args.format == "csv":
            agg.export_csv(output, append=args.append)
        else:
            # JSON export - just copy raw files
            results = agg.load_all_results()
            output = output or "results_export.json"
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            sys.stderr.write(f"Exported {len(results)} results to {output}\n")

    elif args.command == "list":
        runs = agg.list_runs(args.last)
        if runs:
            sys.stderr.write(f"\n{'='*80}\n")
            sys.stderr.write(f"Recent Test Runs (last {len(runs)})\n")
            sys.stderr.write(f"{'='*80}\n")
            for run in runs:
                sys.stderr.write(
                    f"  {run['timestamp'][:19]}  {run['git_commit']}  {run['hostname']:15}  {run['results']}\n"
                )
            sys.stderr.write("\n")
        else:
            sys.stderr.write("No runs found\n")

    elif args.command == "check":
        success = ci_check_regression(
            baseline_commit=args.baseline,
            threshold_percent=args.threshold,
            results_dir=results_dir,
        )
        sys.exit(0 if success else 1)

    elif args.command == "summary":
        runs = agg.list_runs(1)
        if runs:
            sys.stderr.write(f"\nLatest Run: {runs[0]['run_id']}\n")
            sys.stderr.write(f"  Timestamp: {runs[0]['timestamp']}\n")
            sys.stderr.write(f"  Commit: {runs[0]['git_commit']}\n")
            sys.stderr.write(f"  Results: {runs[0]['results']}\n")
        else:
            sys.stderr.write("No runs found\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
