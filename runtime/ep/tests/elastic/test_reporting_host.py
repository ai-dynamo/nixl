# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Reporting host tests; destination tensor checks optionally use CPU Torch."""

import argparse
import ast
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

SOURCE = Path(__file__).with_name("test_adaptive_elastic.py")


def load_helpers():
    names = {
        "load_plan",
        "remote_payload_rows",
        "pair_bytes",
        "benchmark_options",
        "run_perf",
        "save_results",
        "PairCase",
        "add_common_arguments",
        "make_parser",
    }
    module = ast.parse(SOURCE.read_text())
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    nodes += [node for node in module.body if getattr(node, "name", None) in names]
    testing_source = SOURCE.parents[2] / "nixl_ep_v2/utils/testing.py"
    nodes += [
        node
        for node in ast.parse(testing_source.read_text()).body
        if getattr(node, "name", None) == "bench_kineto"
    ]
    namespace = dict(argparse=argparse, json=json, os=os)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class ReportingHostTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_helpers()

    def test_inline_and_file_plan_preserve_world_ranks(self):
        plan = [[4, 5], [0, 1, 4, 5], [0, 1]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(self.ns["load_plan"](str(path)), plan)
            self.assertEqual(self.ns["load_plan"](json.dumps(plan)), plan)
        example = self.ns["load_plan"](str(SOURCE.parent / "plans/two_domains.json"))
        self.assertIn([4, 5, 6, 7], example)
        self.assertEqual(example[-1], list(range(8)))

    def test_actual_parser_keeps_timing_opt_in_and_skip_independent(self):
        parser = self.ns["make_parser"]()
        scenarios = [
            ([], set(), (False, False)),
            (["--benchmark"], set(), (True, False)),
            (["--kineto"], set(), (False, True)),
            (["--benchmark", "--kineto"], set(), (True, True)),
            ([], {0}, (True, True)),
            (["--skip-perf-test"], {0}, (False, True)),
            (["--benchmark", "--skip-perf-test", "--kineto"], set(), (False, True)),
        ]
        for argv, phases, expected in scenarios:
            with self.subTest(argv=argv, phases=phases):
                args = parser.parse_args(argv)
                self.assertEqual(
                    self.ns["benchmark_options"](args, 0, phases), expected
                )
        self.assertEqual(
            parser.parse_args(["--save-output"]).save_output,
            "adaptive_elastic_results.json",
        )
        self.assertEqual(
            parser.parse_args(["--save-output", "run.json"]).save_output, "run.json"
        )
        self.assertIn("--kineto", parser.format_help())

    def test_remote_rows_exclude_local_domain_and_deduplicate(self):
        count = self.ns["remote_payload_rows"]
        self.assertEqual(count([[0, 1, 4, 7, -1]], 4, 2, 0), (0, 0))
        self.assertEqual(count([[0, 8, 9, 12], [8, 12, 13], [-1, -1]], 4, 2, 0), (2, 4))
        # Active domain/position holes retain their extent coordinates; source domain is not necessarily zero.
        self.assertEqual(count([[10, 26, 27, 30, 31, -1]], 2, 4, 1), (1, 2))
        self.assertEqual(count([[32, 35, 60, -1]], 4, 8, 1), (0, 0))

    def test_combine_destinations_filter_ignored_entries_before_deduplication(self):
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            self.skipTest("requires PyTorch for CPU tensor accounting checks")

        source = SOURCE.with_name("test_ep.py")
        helper = next(
            node
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_unique_and_valid_dst_count"
        )
        namespace = dict(
            torch=torch,
            Optional=Optional,
            num_experts=8,
            args=SimpleNamespace(ignore_local_traffic=True),
        )
        exec(
            compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        count = namespace["get_unique_and_valid_dst_count"]
        for rows, excluded, expected in (
            ([[-1, -1]], (None, None), 0),
            ([[0, 1]], (0, 2), 0),
            ([[0, 2]], (0, 2), 1),
            ([[-1, 2]], (None, None), 1),
            ([[2, 2], [2, 2]], (None, None), 2),
        ):
            with self.subTest(rows=rows, excluded=excluded):
                destinations = torch.tensor(rows, dtype=torch.int64, device="cpu")
                original = destinations.clone()
                self.assertEqual(count(destinations, *excluded), expected)
                self.assertTrue(torch.equal(destinations, original))

    def test_return_payload_counts_follow_reduction_mode(self):
        buffer = SimpleNamespace(
            num_experts_per_rank=2, scaleout_rank_idx=1, allow_multiple_reduction=True
        )
        case = SimpleNamespace(buffer=buffer, extent=lambda phase: (4, 4))
        pair = SimpleNamespace(
            dispatch_args={"topk_idx": Rows([[10, 26, 27, 30, -1]])},
            num_recv=3,
            dispatch_bytes_per_token=10,
            combine_bytes_per_token=20,
        )
        sizes = self.ns["pair_bytes"](case, pair, [5, 7, 13, 15])
        self.assertEqual(
            sizes,
            dict(
                dispatch_received_payload=30,
                combine_input_payload=60,
                dispatch_remote_payload=10,
                combine_remote_payload=20,
            ),
        )
        buffer.allow_multiple_reduction = False
        self.assertEqual(
            self.ns["pair_bytes"](case, pair, [])["combine_remote_payload"], 40
        )

    def test_kineto_only_never_runs_event_benchmark(self):
        self.check_benchmark(events=False, kernels=True)

    def test_external_tools_refuse_kineto_but_allow_event_only_runs(self):
        with patch.dict(os.environ, {"NIXL_EP_USE_NVIDIA_TOOLS": "1"}):
            for names in ("barrier", ("dispatch_impl", "dispatch_copy_epilogue_impl")):
                with self.subTest(names=names), self.assertRaisesRegex(
                    RuntimeError,
                    "Kineto timing conflicts with NIXL_EP_USE_NVIDIA_TOOLS: disable Kineto",
                ):
                    self.ns["bench_kineto"](
                        lambda: self.fail("refused profiling must not run the workload"),
                        names,
                    )
            args = self.ns["make_parser"]().parse_args([])
            self.assertEqual(
                self.ns["benchmark_options"](args, 0, set()), (False, False)
            )
            self.check_benchmark(events=True, kernels=False)

    def test_event_only_never_runs_kineto_and_measures_pair_once(self):
        self.check_benchmark(events=True, kernels=False)

    def test_combined_timing_does_not_repeat_pair_benchmark(self):
        self.check_benchmark(events=True, kernels=True)

    def check_benchmark(self, events, kernels):
        calls = dict(events=0, kernels=0)
        buffer = SimpleNamespace(
            world_rank=0,
            scaleout_rank_idx=0,
            num_experts_per_rank=2,
            allow_multiple_reduction=True,
            barrier=lambda: None,
            dispatch=lambda **kw: None,
            combine=lambda **kw: None,
        )
        case = SimpleNamespace(buffer=buffer, extent=lambda phase: (2, 2))
        pair = SimpleNamespace(
            num_tokens=1,
            num_recv=1,
            dispatch_args={"topk_idx": Rows([[4, 5]])},
            combine_args={},
            dispatch_bytes_per_token=10,
            combine_bytes_per_token=20,
            pair=lambda: None,
        )

        def bench(fn, **kwargs):
            calls["events"] += 1
            self.assertIs(kwargs["barrier"], buffer.barrier)
            return 2e-6, 1e-6, 3e-6

        def kineto(fn, **kwargs):
            calls["kernels"] += 1
            self.assertIs(kwargs["barrier"], buffer.barrier)
            return 1e-6, 0.5e-6

        self.ns.update(
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
            PairCase=lambda *args: pair,
            bench=bench,
            bench_kineto=kineto,
        )
        with redirect_stdout(io.StringIO()):
            record = self.ns["run_perf"](
                case,
                0,
                list(range(4)),
                "hybrid",
                SimpleNamespace(perf_tag="host"),
                measure_events=events,
                measure_kernels=kernels,
            )
        self.assertEqual(
            calls, dict(events=3 if events else 0, kernels=2 if kernels else 0)
        )
        self.assertEqual("events" in record, events)
        self.assertEqual("kineto" in record, kernels)
        self.assertEqual("pair_payload_gbps" in record, events)
        if events:
            self.assertEqual(
                record["events"]["pair"], dict(avg_us=2.0, min_us=1.0, max_us=3.0)
            )
            self.assertEqual(record["remote_payload_gbps"], 0.015)
        json.dumps(record, allow_nan=False)

    def test_pair_preparation_performs_no_communication(self):
        calls = []

        class Tensor:
            def __init__(self, *shape):
                self.shape = shape

            def numel(self):
                return self.shape[0] * self.shape[1]

            def element_size(self):
                return 2

            def __getitem__(self, index):
                return self

        buffer = SimpleNamespace(
            world_rank=0,
            dispatch=lambda **kw: calls.append("dispatch")
            or (None, None, None, SimpleNamespace(num_recv_tokens=3), None),
            combine=lambda **kw: calls.append("combine"),
        )
        case = SimpleNamespace(
            buffer=buffer,
            num_max_tokens_per_rank=4,
            num_experts=8,
            num_topk=2,
            hidden=16,
            active_expert_mask=lambda phase: None,
            dispatch_kwargs=lambda **kw: {},
        )
        self.ns.update(
            torch=SimpleNamespace(
                Generator=lambda **kw: SimpleNamespace(manual_seed=lambda seed: None),
                randn=lambda *shape, **kw: Tensor(*shape),
                bfloat16="bf16",
            ),
            sample_routing=lambda *args: (Rows([[0, 1]]), None),
        )
        pair = self.ns["PairCase"](
            case, 0, [0, 1], SimpleNamespace(perf_tokens=0, seed=0, perf_fp8=0)
        )
        self.assertEqual(calls, [])
        self.assertEqual(pair.dispatch_bytes_per_token, 32)
        pair.pair()
        self.assertEqual(calls, ["dispatch", "combine"])
        self.assertEqual(pair.num_recv, 3)

    def test_json_contains_all_ranks_and_null_absent_measurements(self):
        mine = dict(
            world_rank=4,
            phases=[dict(performance=None)],
            transitions=[dict(gpu_gap_ms=None, first_us=[])],
        )
        peer = dict(world_rank=7, phases=[], transitions=[])
        barriers = []
        cohort = SimpleNamespace(
            rank=0,
            _all_gather=lambda name, value: [value, peer],
            barrier=barriers.append,
        )
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp) / "results.json"
            self.ns["save_results"](
                cohort, str(path), {"benchmark": False}, [[4, 7]], mine
            )
            data = json.loads(path.read_text())
            self.assertEqual(
                data,
                dict(
                    schema_version=1,
                    config={"benchmark": False},
                    plan=[[4, 7]],
                    ranks=[mine, peer],
                ),
            )
            self.assertEqual(barriers, ["results-written"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
