#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Performance Report Generator for NIXL EP Tests.

Analyzes test results and generates formatted reports with:
- Summary tables
- Throughput/latency comparisons
- Scaling analysis
- Markdown/HTML output

Usage:
    # Generate report from JSON results
    python3 report_generator.py --input results.json --output report.md

    # Run tests and generate report in one command
    python3 report_generator.py --run-tests --num-processes=8 --output report.md

    # Compare two runs
    python3 report_generator.py --compare baseline.json current.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PerformanceMetric:
    """A single performance metric with statistics."""

    name: str
    unit: str
    avg: float
    min: float
    max: float
    std: float = 0.0


@dataclass
class TestResult:
    """Parsed test result."""

    test_name: str
    config: Dict[str, Any]
    passed: int
    total: int
    metrics: Dict[str, PerformanceMetric]


def parse_json_results(data: Dict) -> List[TestResult]:
    """Parse JSON results into TestResult objects."""
    results = []

    for config_key, test_data in data.get("results", {}).items():
        rank_results = test_data.get("results", [])
        passed = sum(1 for r in rank_results if r.get("passed"))
        total = len(rank_results)

        # Aggregate metrics across ranks
        metrics: Dict[str, PerformanceMetric] = {}
        metric_values: Dict[str, List[float]] = dict()

        for r in rank_results:
            if r.get("passed") and r.get("metrics"):
                for key, value in r["metrics"].items():
                    if isinstance(value, (int, float)):
                        if key not in metric_values:
                            metric_values[key] = []
                        metric_values[key].append(value)

        for key, values in metric_values.items():
            if values:
                import statistics

                metrics[key] = PerformanceMetric(
                    name=key,
                    unit=get_metric_unit(key),
                    avg=sum(values) / len(values),
                    min=min(values),
                    max=max(values),
                    std=statistics.stdev(values) if len(values) > 1 else 0.0,
                )

        results.append(
            TestResult(
                test_name=config_key,
                config=test_data.get("config", {}),
                passed=passed,
                total=total,
                metrics=metrics,
            )
        )

    return results


def get_metric_unit(metric_name: str) -> str:
    """Get the unit for a metric."""
    units = {
        "tokens_per_sec": "tok/s",
        "avg_latency_us": "μs",
        "avg_latency_ms": "ms",
        "bandwidth_gbps": "GB/s",
        "init_ms": "ms",
        "connect_ms": "ms",
        "disconnect_ms": "ms",
        "destroy_ms": "ms",
        "total_ms": "ms",
        "reconnect_ms": "ms",
    }
    return units.get(metric_name, "")


def format_number(value: float, precision: int = 2) -> str:
    """Format a number with appropriate precision."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.{precision}f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.{precision}f}K"
    elif value >= 1:
        return f"{value:.{precision}f}"
    else:
        return f"{value:.{precision + 2}f}"


def generate_markdown_report(
    results: List[TestResult], config: Dict[str, Any], output_path: Optional[str] = None
) -> str:
    """Generate a Markdown performance report."""
    lines = []

    # Header
    lines.append("# NIXL EP Performance Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration
    lines.append("## Test Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in config.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # Summary Table
    lines.append("## Summary")
    lines.append("")

    # Group by test type
    test_types: Dict[str, List[TestResult]] = dict()
    for result in results:
        test_type = result.test_name.split("_")[0]
        if test_type not in test_types:
            test_types[test_type] = []
        test_types[test_type].append(result)

    # Control plane results
    ctrl_results = [
        r
        for r in results
        if "ctrl" in r.test_name.lower()
        or any(m in r.metrics for m in ["init_ms", "connect_ms"])
    ]
    if ctrl_results:
        lines.append("### Control Plane Latency")
        lines.append("")
        lines.append(
            "| Test | Experts/Rank | Total | Init (ms) | Connect (ms) | Disconnect (ms) | Destroy (ms) | Status |"
        )
        lines.append(
            "|------|--------------|-------|-----------|--------------|-----------------|--------------|--------|"
        )

        for r in ctrl_results:
            experts_per_rank = r.config.get(
                "num_experts_per_rank", r.config.get("num_experts", "?")
            )
            world_size = r.config.get("world_size", r.total)
            total_experts = (
                experts_per_rank * world_size
                if isinstance(experts_per_rank, int)
                else "?"
            )
            init = r.metrics.get("init_ms", PerformanceMetric("", "", 0, 0, 0))
            conn = r.metrics.get("connect_ms", PerformanceMetric("", "", 0, 0, 0))
            disc = r.metrics.get("disconnect_ms", PerformanceMetric("", "", 0, 0, 0))
            dest = r.metrics.get("destroy_ms", PerformanceMetric("", "", 0, 0, 0))
            status = "✅" if r.passed == r.total else f"❌ ({r.passed}/{r.total})"

            lines.append(
                f"| {r.test_name} | {experts_per_rank} | {total_experts} | "
                f"{init.avg:.1f} | {conn.avg:.1f} | {disc.avg:.1f} | {dest.avg:.1f} | {status} |"
            )
        lines.append("")

    # Data plane results
    dp_results = [
        r
        for r in results
        if "dispatch" in r.test_name.lower()
        or "combine" in r.test_name.lower()
        or "e2e" in r.test_name.lower()
    ]
    if dp_results:
        lines.append("### Data Plane Throughput")
        lines.append("")
        lines.append(
            "| Test | Tokens | Hidden | Throughput (tok/s) | Latency (μs) | BW (GB/s) | Status |"
        )
        lines.append(
            "|------|--------|--------|-------------------|--------------|-----------|--------|"
        )

        for r in dp_results:
            tokens = r.config.get("num_tokens", "?")
            hidden = r.config.get("hidden", "?")
            throughput = r.metrics.get(
                "tokens_per_sec", PerformanceMetric("", "", 0, 0, 0)
            )
            latency = r.metrics.get(
                "avg_latency_us", PerformanceMetric("", "", 0, 0, 0)
            )
            bandwidth = r.metrics.get(
                "bandwidth_gbps", PerformanceMetric("", "", 0, 0, 0)
            )
            status = "✅" if r.passed == r.total else f"❌ ({r.passed}/{r.total})"

            lines.append(
                f"| {r.test_name} | {tokens} | {hidden} | "
                f"{format_number(throughput.avg)} | {latency.avg:.1f} | "
                f"{bandwidth.avg:.2f} | {status} |"
            )
        lines.append("")

    # Detailed Metrics
    lines.append("## Detailed Metrics")
    lines.append("")

    for result in results:
        lines.append(f"### {result.test_name}")
        lines.append("")
        lines.append(f"**Status**: {result.passed}/{result.total} ranks passed")
        lines.append("")

        if result.config:
            lines.append("**Configuration**:")
            for key, value in result.config.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        if result.metrics:
            lines.append("**Metrics**:")
            lines.append("")
            lines.append("| Metric | Avg | Min | Max | Std |")
            lines.append("|--------|-----|-----|-----|-----|")
            for name, metric in result.metrics.items():
                lines.append(
                    f"| {name} ({metric.unit}) | "
                    f"{format_number(metric.avg)} | "
                    f"{format_number(metric.min)} | "
                    f"{format_number(metric.max)} | "
                    f"{format_number(metric.std)} |"
                )
            lines.append("")

    # Known Issues
    lines.append("## Known Issues")
    lines.append("")
    lines.append(
        "See `tests/bugs/README.md` for details on known bugs that may affect results:"
    )
    lines.append(
        "- **BUG-01**: Segfault on repeated buffer creation (use single round)"
    )
    lines.append("- **BUG-02**: UCX rcache assertion (~30% with 16 experts/rank)")
    lines.append("- **BUG-03**: gdr_copy warnings (cosmetic)")
    lines.append("- **BUG-04**: invalidateRemoteMD warnings (cosmetic)")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        sys.stderr.write(f"Report written to: {output_path}\n")

    return report


def compare_results(baseline: Dict, current: Dict) -> str:
    """Compare two result sets and generate a diff report."""
    lines = []

    lines.append("# Performance Comparison Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    baseline_results = {r.test_name: r for r in parse_json_results(baseline)}
    current_results = {r.test_name: r for r in parse_json_results(current)}

    all_tests = set(baseline_results.keys()) | set(current_results.keys())

    lines.append("## Comparison Summary")
    lines.append("")
    lines.append("| Test | Metric | Baseline | Current | Change |")
    lines.append("|------|--------|----------|---------|--------|")

    for test_name in sorted(all_tests):
        base = baseline_results.get(test_name)
        curr = current_results.get(test_name)

        if base and curr:
            # Compare key metrics
            for metric_name in ["tokens_per_sec", "avg_latency_us", "connect_ms"]:
                if metric_name in base.metrics and metric_name in curr.metrics:
                    base_val = base.metrics[metric_name].avg
                    curr_val = curr.metrics[metric_name].avg

                    if base_val > 0:
                        change_pct = ((curr_val - base_val) / base_val) * 100
                        # For latency, lower is better; for throughput, higher is better
                        is_better = (
                            change_pct < 0
                            if "latency" in metric_name
                            else change_pct > 0
                        )
                        indicator = (
                            "🟢" if is_better else "🔴" if abs(change_pct) > 5 else "🟡"
                        )

                        lines.append(
                            f"| {test_name} | {metric_name} | "
                            f"{format_number(base_val)} | {format_number(curr_val)} | "
                            f"{indicator} {change_pct:+.1f}% |"
                        )
        elif curr and not base:
            lines.append(f"| {test_name} | - | NEW | - | - |")
        elif base and not curr:
            lines.append(f"| {test_name} | - | - | REMOVED | - |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance reports for NIXL EP tests"
    )

    parser.add_argument("--input", type=str, help="Input JSON results file")
    parser.add_argument(
        "--output",
        type=str,
        default="perf_report.md",
        help="Output report file (Markdown)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CURRENT"),
        help="Compare two result files",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format",
    )

    args = parser.parse_args()

    if args.compare:
        # Compare mode
        with open(args.compare[0]) as f:
            baseline = json.load(f)
        with open(args.compare[1]) as f:
            current = json.load(f)

        report = compare_results(baseline, current)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            sys.stderr.write(f"Comparison report written to: {args.output}\n")
        else:
            sys.stderr.write(report + "\n")

    elif args.input:
        # Generate report from existing results
        with open(args.input) as f:
            data = json.load(f)

        results = parse_json_results(data)
        config = data.get("config", {})

        report = generate_markdown_report(results, config, args.output)

        if not args.output:
            sys.stderr.write(report + "\n")

    else:
        parser.print_help()
        sys.stderr.write("\nError: Either --input or --compare is required\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
