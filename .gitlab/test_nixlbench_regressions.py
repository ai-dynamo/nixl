#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run non-ETCD regressions and require the pipeline coverage to execute."""
import argparse
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED_SUITES = {
    "PrintStatsTest",
    "RawCliDispatchTest",
    "PluginMetadataDiscoveryTest",
    "RawPosixParserTest",
    "RawPlanTest",
    "RawRequestConversionTest",
}
REQUIRED_POSIX = {
    "ShortPipelinesPassReadWriteConsistency",
    "CappedPipelinePreservesBatchSize",
    "LargeBlocksWithShortWarmupPassConsistency",
    "PipelineWarningUsesDescriptorDepthAndOneThreadPerPhase",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    benchmark = os.environ.get("NIXLBENCH_BINARY")
    if not benchmark or not os.access(benchmark, os.X_OK):
        parser.error("NIXLBENCH_BINARY must name the installed benchmark executable")
    # Never accept an XML file left by an earlier invocation.
    with tempfile.TemporaryDirectory(prefix="nixlbench-regressions-") as temporary:
        report = Path(temporary) / "results.xml"
        result = subprocess.run(
            [
                str(args.binary.resolve()),
                "--gtest_filter=-EtcdReductionTest.*",
                "--gtest_output=xml:" + str(report),
            ],
            timeout=180,
        )
        if not report.exists():
            raise RuntimeError("NIXLBench regressions produced no XML report")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(report.read_bytes())
        cases = ET.parse(report).findall(".//testcase")
        required = [
            case
            for case in cases
            if case.get("classname") in REQUIRED_SUITES
            or (
                case.get("classname") == "PosixIntegrationTest"
                and case.get("name") in REQUIRED_POSIX
            )
        ]
        suites = {case.get("classname") for case in required}
        posix = {
            case.get("name")
            for case in required
            if case.get("classname") == "PosixIntegrationTest"
        }
        if (
            result.returncode != 0
            or not REQUIRED_SUITES <= suites
            or posix != REQUIRED_POSIX
            or any(
                case.find("failure") is not None or case.find("error") is not None
                for case in cases
            )
            or any(
                case.get("status") != "run"
                or case.get("result") != "completed"
                or case.find("skipped") is not None
                for case in required
            )
        ):
            raise RuntimeError(
                "Required NIXLBench regressions did not all execute and pass"
            )
    print(
        f"NIXLBench regressions: {len(cases)} cases reported; required coverage passed"
    )


if __name__ == "__main__":
    main()
