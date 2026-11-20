#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the service-backed reduction suite and reject missing or skipped cases."""
import argparse
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

EXPECTED = {
    "OneSenderDoesNotWaitForTarget",
    "FastContributorsCannotOverwriteLaterRounds",
    "CollectorWaitsForDelayedSender",
    "InterruptedMissingContributionDoesNotPublishPartialSum",
    "LostPeerFailsWithoutPartialSum",
    "MalformedContributionFailsWithoutPartialSum",
    "LegacyPairwiseParticipantCountIsPreserved",
    "LegacyTwoRankCallIncludesBothRanks",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    endpoint = os.environ.get("NIXL_ETCD_ENDPOINTS")
    if not endpoint:
        parser.error("NIXL_ETCD_ENDPOINTS must name the isolated test service")
    env = dict(os.environ, NIXLBENCH_ETCD_TEST_ENDPOINT=endpoint)
    # A fresh path prevents a stale successful report from masking a missing suite.
    with tempfile.TemporaryDirectory(prefix="nixlbench-etcd-tests-") as temporary:
        report = Path(temporary) / "results.xml"
        result = subprocess.run(
            [
                str(args.binary.resolve()),
                "--gtest_filter=EtcdReductionTest.*",
                "--gtest_output=xml:" + str(report),
            ],
            env=env,
            timeout=120,
        )
        if not report.exists():
            raise RuntimeError("ETCD tests produced no XML report")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(report.read_bytes())
        cases = ET.parse(report).findall(".//testcase")
        names = [case.get("name") for case in cases]
        if (
            result.returncode != 0
            or len(cases) != len(EXPECTED)
            or set(names) != EXPECTED
            or any(
                case.get("classname") != "EtcdReductionTest"
                or case.get("status") != "run"
                or case.get("result") != "completed"
                or case.find("skipped") is not None
                or case.find("failure") is not None
                or case.find("error") is not None
                for case in cases
            )
        ):
            raise RuntimeError(
                "Expected eight executed ETCD reduction passes and zero skips"
            )
    print("ETCD reduction tests: 8 executed passes, 0 skips")


if __name__ == "__main__":
    main()
