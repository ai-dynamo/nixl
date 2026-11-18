# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Host-only regression tests for topology-driven automatic SM selection.

The production Python class is AST-loaded with simulated native/CUDA boundaries.
No torch import, GPU kernel, distributed process, or cluster access is performed.
Set NIXL_EP_AUTO_SM_SOURCE to exercise another checkout's elastic.py.
"""

import ast
import functools
import math
import os
import re
import unittest
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_REPO_SOURCE = Path(__file__).resolve().parents[2] / "python/buffers/elastic.py"
SOURCE = Path(os.environ.get("NIXL_EP_AUTO_SM_SOURCE", str(_REPO_SOURCE)))
_TOPOLOGY_HEADER = SOURCE.parents[1] / "include/nixl_ep_v2/common/topology.cuh"
_MIN_ACTIVE_RANKS_MATCH = re.search(
    r"\bkMinNumActiveRanks\s*=\s*(\d+)\s*;", _TOPOLOGY_HEADER.read_text()
)
assert (
    _MIN_ACTIVE_RANKS_MATCH is not None
), "Missing kMinNumActiveRanks in topology header"
MIN_ACTIVE_RANKS = int(_MIN_ACTIVE_RANKS_MATCH.group(1))


class FakeCuda:
    def __init__(self):
        self.capturing = False
        self.synchronizations = 0

    def is_current_stream_capturing(self):
        return self.capturing

    def get_device_properties(self, device):
        return SimpleNamespace(multi_processor_count=148)

    def synchronize(self):
        self.synchronizations += 1


class FakeRuntime:
    """Only models the native stage/commit boundary, never the estimator."""

    def __init__(self, *args, cuda=None):
        self.cuda = cuda
        self.mode, self.sms = 0, 24
        self.pending = None
        self.staged_views = False
        self.connected = set()
        self.fail_stage = False
        self.fail_commit = False
        self.dispatch_calls = 0
        self.commit_calls = 0
        self.max_domains, self.max_positions = args[7:9]
        self.domain, self.position = args[9:11]
        self.world_rank = self.domain * self.max_positions + self.position
        self.masks = (1 << self.domain, 1 << self.position)
        self.num_experts_per_rank = args[-1]

    def stage_topology_masks(self, mode, domains, position_words, sms):
        positions = sum(word << (64 * i) for i, word in enumerate(position_words))
        if self.fail_stage:
            raise ValueError("simulated stage refusal")
        if not (domains and positions and 2 <= sms <= 24 and sms % 2 == 0):
            raise ValueError("invalid staged topology")
        if (
            domains.bit_length() > self.max_domains
            or positions.bit_length() > self.max_positions
        ):
            raise ValueError("world exceeds declared maximum")
        self.pending = (mode, sms, domains, positions)

    def commit_topology(self):
        if self.cuda.capturing:
            raise RuntimeError("commit inside CUDA graph capture")
        if self.fail_commit:
            raise ValueError("simulated mapping refusal before activation")
        if self.pending is None and not self.staged_views:
            raise ValueError("nothing staged")
        if self.pending is not None:
            self.mode, self.sms, domains, positions = self.pending
            self.masks = (domains, positions)
        self.pending = None
        self.staged_views = False
        self.commit_calls += 1

    def get_topology(self):
        return self.mode, self.sms

    def get_topology_masks(self):
        domains, positions = self.masks
        return domains, [positions]

    def has_staged_topology(self):
        return self.pending is not None

    def get_local_agent_name(self):
        return "local"

    def get_connected_ranks(self):
        return sorted(self.connected)

    def get_local_peer_info(self):
        return 4096, 0, self.world_rank, 2097152

    def connect_ranks(self, ranks, remote_agent_names, peer_info):
        self.connected.update(ranks)
        self.staged_views = True

    def disconnect_ranks(self, ranks):
        self.connected.difference_update(ranks)
        self.staged_views = True

    def dispatch(self, *args):
        self.dispatch_calls += 1
        topk_idx = args[2]
        return (
            object(),
            None,
            object(),
            None,
            topk_idx,
            16,
            [],
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            None,
        )


def load_buffer():
    """Execute original class methods; replace only external boundaries."""
    cuda = FakeCuda()
    calls = {"rdma": 0, "nvlink": 0}

    @functools.lru_cache()
    def rdma_bandwidth():
        calls["rdma"] += 1
        return 50.0

    @functools.lru_cache()
    def nvlink_bandwidth():
        calls["nvlink"] += 1
        return 600.0

    def native_factory(*args):
        return FakeRuntime(*args, cuda=cuda)

    def exchange_records(store, rank, peers, value, **kwargs):
        return nullcontext(
            {peer: (b"peer", (4096, peer, peer, 2097152)) for peer in peers}
        )

    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "EPHandle",
            "ElasticBuffer",
        }:
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_mode_code":
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id
            in {"_MODE_NAMES", "_DEFAULT_SM_READ_GBS", "_DEFAULT_SM_WRITE_GBS"}
            for assigned in node.targets
            for target in ast.walk(assigned)
        ):
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {
        "math": math,
        "os": os,
        "functools": functools,
        "uuid": uuid,
        "weak_lru": functools.lru_cache,
        "align": lambda value, alignment: ((value + alignment - 1) // alignment)
        * alignment,
        "value_or": lambda value, default: default if value is None else value,
        "torch": SimpleNamespace(cuda=cuda),
        "_C": SimpleNamespace(
            ElasticBuffer=native_factory,
            calculate_elastic_buffer_size=Mock(return_value=2097152),
            min_elastic_buffer_bytes=2097152,
            get_adaptive_num_launch_sms=lambda: 24,
            min_num_active_ranks=MIN_ACTIVE_RANKS,
        ),
        "get_rdma_gbs": rdma_bandwidth,
        "get_nvlink_gbs": nvlink_bandwidth,
        "check_torch_deterministic": lambda: None,
        "EventOverlap": lambda event: event,
        "exchange_records": exchange_records,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["ElasticBuffer"], cuda, calls


class AutoSMHostTests(unittest.TestCase):
    def setUp(self):
        self.Buffer, self.cuda, self.probes = load_buffer()
        self.buffer = self.Buffer(
            world_rank=0,
            num_max_tokens_per_rank=16,
            hidden=7168,
            num_topk=8,
            num_experts_per_rank=16,
            max_num_scaleout_ranks=2,
            max_num_scaleup_ranks=16,
            store=object(),
            explicitly_destroy=True,
        )
        peers = list(range(1, 16))
        self.buffer.connect_ranks(peers)
        self.buffer.set_active_ranks(list(range(16)))
        self.cuda.synchronizations = 0

    def test_mode_override_is_confined_to_test_buffer(self):
        source = Path(__file__).with_name("test_adaptive_modes.py")
        node = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "ModeTestBuffer"
        )
        namespace = {"nixl_ep_v2": SimpleNamespace(ElasticBuffer=self.Buffer), "os": os}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        TestBuffer = namespace["ModeTestBuffer"]
        with patch.dict(os.environ, {"NIXL_EP_ADAPTIVE_MODE": "hybrid"}):
            self.assertEqual(self.Buffer._mode_of(1), "direct")
            self.assertEqual(TestBuffer._mode_of(1), "hybrid")
            self.assertEqual(TestBuffer._mode_of(3), "hybrid")
        buffer = TestBuffer.__new__(TestBuffer)
        buffer.__dict__.update(self.buffer.__dict__)
        buffer.set_topology_forced("hybrid")
        self.assertEqual(buffer._get_topology(), ("hybrid", 24))
        buffer._stage_topology(active_ranks=self.ranks([0], range(8)))
        with self.assertRaisesRegex(AssertionError, "Commit the staged world"):
            buffer.set_topology_forced("direct")
        buffer._commit_topology()
        self.assertEqual(buffer._get_topology()[0], "direct")

    @staticmethod
    def ranks(domains, positions):
        return [domain * 16 + position for domain in domains for position in positions]

    def dispatch(self, num_sms=0, num_topk=8):
        return self.buffer.dispatch(
            object(),
            SimpleNamespace(shape=(16, num_topk)),
            num_sms=num_sms,
            do_cpu_sync=False,
        )

    def expected_sms(self, domains, positions, num_topk=8):
        # Existing traffic model is the oracle; this task changes its invocation,
        # not its formula. Supply proposed dimensions explicitly, never read the
        # live descriptor that _stage_topology has not committed yet.
        active_experts = (
            self.buffer.num_experts_per_rank * len(domains) * len(positions)
        )
        estimate = self.buffer._get_theoretical_num_sms(
            active_experts,
            num_topk,
            len(domains),
            len(positions),
            50.0,
            600.0,
            200,
            50,
        )
        return min(24, max(2, ((estimate + 1) // 2) * 2))

    def test_constructor_and_membership_before_first_auto_dispatch_keep_current_count(
        self,
    ):
        self.assertEqual(self.buffer.num_experts_per_rank, 16)
        self.assertEqual(self.buffer.runtime.num_experts_per_rank, 16)
        self.assertEqual(self.buffer._get_topology(), ("direct", 24))
        self.assertEqual(self.probes, {"rdma": 0, "nvlink": 0})
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(self.buffer._get_topology(), ("hybrid", 24))
        self.assertEqual(self.probes, {"rdma": 0, "nvlink": 0})
        self.assertEqual(self.buffer.runtime.dispatch_calls, 0)

    def test_stage_uses_proposed_masks_and_publishes_only_at_commit(self):
        self.dispatch()
        before = self.buffer._get_topology(), self.buffer._get_topology_masks()
        expected = self.expected_sms([0, 1], range(8))
        self.assertNotEqual(
            before[0][1], expected, "test needs distinguishable SM estimates"
        )
        self.buffer._stage_topology(active_ranks=self.ranks([0, 1], range(8)))
        self.assertEqual(
            (self.buffer._get_topology(), self.buffer._get_topology_masks()), before
        )
        self.assertEqual(self.buffer.runtime.pending[1], expected)
        self.buffer._commit_topology()
        self.assertEqual(self.buffer._get_topology(), ("hybrid", expected))
        self.assertEqual(self.buffer._get_topology_masks(), (3, 255))

    def test_estimate_query_does_not_enable_automatic_retuning(self):
        self.buffer.get_theoretical_num_sms(8)
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(self.buffer._get_topology()[1], 24)

    def test_numeric_debug_flag(self):
        for value, enabled in (("0", False), ("1", True)):
            with self.subTest(value=value):
                self.buffer._get_theoretical_num_sms.cache_clear()
                with patch.dict(os.environ, NIXL_EP_BUFFER_DEBUG=value):
                    with patch("builtins.print") as output:
                        self.buffer.get_theoretical_num_sms(8)
                self.assertEqual(output.called, enabled)

    def test_query_and_staging_reuse_one_numerical_model_cache_entry(self):
        self.dispatch()
        model = self.buffer._get_theoretical_num_sms
        before = model.cache_info()
        self.buffer.get_theoretical_num_sms(8)
        self.buffer._stage_topology(active_ranks=self.ranks([0], range(16)))
        after = model.cache_info()
        self.assertEqual(after.misses, before.misses)
        self.assertGreaterEqual(after.hits - before.hits, 2)

    def test_most_recent_automatic_dispatch_supplies_topk_configuration(self):
        self.dispatch()
        self.dispatch(num_topk=1)
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(
            self.buffer._get_topology()[1],
            self.expected_sms([0, 1], range(8), num_topk=1),
        )
        self.assertNotEqual(
            self.buffer._get_topology()[1],
            self.expected_sms([0, 1], range(8)),
            "test needs distinct estimates for the two MoE configurations",
        )

    def test_auto_retunes_repeated_shrink_expand_and_topology_changes_without_dispatch(
        self,
    ):
        self.dispatch()
        dispatch_calls = self.buffer.runtime.dispatch_calls
        cases = [
            ([0, 1], range(8)),
            ([0], range(8)),
            ([0], range(16)),
            ([0, 1], range(8)),
            ([0], range(4)),
            ([0], range(2)),
            ([0, 1], range(1)),
            ([0], range(16)),
        ]
        for domains, positions in cases:
            with self.subTest(domains=domains, positions=list(positions)):
                self.buffer.set_active_ranks(self.ranks(domains, positions))
                self.assertEqual(
                    self.buffer._get_topology()[1],
                    self.expected_sms(domains, positions),
                )
                self.assertEqual(self.buffer.num_experts_per_rank, 16)
        self.assertEqual(self.buffer.runtime.dispatch_calls, dispatch_calls)
        self.assertEqual(self.cuda.synchronizations, 0)
        self.assertEqual(self.probes, {"rdma": 1, "nvlink": 1})

    def test_holes_do_not_change_active_slot_count_or_estimate(self):
        estimates = []
        with patch.object(
            self.buffer,
            "_get_theoretical_num_sms",
            wraps=self.buffer._get_theoretical_num_sms,
        ) as model:
            for domains, positions in (([0], [0, 1]), ([0], [0, 15]), ([1], [0, 15])):
                self.buffer.set_active_ranks(self.ranks(domains, positions))
                estimates.append(self.buffer.get_theoretical_num_sms(8))
            self.assertEqual(
                [call.args[0] for call in model.call_args_list], [32, 32, 32]
            )
        self.assertEqual(estimates, [estimates[0]] * len(estimates))
        self.assertEqual(self.probes, {"rdma": 0, "nvlink": 1})

    def test_one_position_per_domain_does_not_probe_nvlink(self):
        self.buffer.set_active_ranks(self.ranks([0, 1], [0]))

        def no_nvlink():
            raise RuntimeError("No NVLink links present")

        with patch.dict(self.Buffer.dispatch.__globals__, get_nvlink_gbs=no_nvlink):
            self.dispatch()
        self.assertEqual(self.buffer.runtime.dispatch_calls, 1)
        self.assertEqual(self.buffer._get_topology()[1], self.expected_sms([0, 1], [0]))
        self.assertEqual(self.probes, {"rdma": 1, "nvlink": 0})

    def test_explicit_dispatch_disables_retune_until_automatic_dispatch_resumes(self):
        self.dispatch()
        self.dispatch(num_sms=10)
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(self.buffer._get_topology()[1], 10)
        self.dispatch(num_sms=0)
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0, 1], range(8))
        )
        self.buffer.set_active_ranks(self.ranks([0], range(16)))
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0], range(16))
        )

    def test_explicit_staged_count_is_preserved_until_automatic_dispatch(self):
        self.dispatch()
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)), num_active_sms=12)
        self.assertEqual(self.buffer._get_topology()[1], 12)
        self.buffer.set_active_ranks(self.ranks([0], range(16)))
        self.assertEqual(self.buffer._get_topology()[1], 12)
        self.dispatch()
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0], range(16))
        )

    def test_explicit_unchanged_dispatch_count_still_selects_manual_policy(self):
        self.dispatch()
        current = self.buffer._get_topology()[1]
        self.dispatch(num_sms=current)
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(self.buffer._get_topology()[1], current)

    def test_explicit_setter_pins_count_until_auto_dispatch_resumes(self):
        self.dispatch()
        self.buffer.set_num_active_sms(10)
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(self.buffer._get_topology()[1], 10)
        self.dispatch()
        self.buffer.set_active_ranks(self.ranks([0], range(16)))
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0], range(16))
        )

    def test_failed_explicit_stage_does_not_disable_automatic_policy(self):
        self.dispatch()
        before = self.buffer._get_topology(), self.buffer._get_topology_masks()
        self.buffer.runtime.fail_stage = True
        with self.assertRaisesRegex(ValueError, "stage refusal"):
            self.buffer._stage_topology(
                active_ranks=self.ranks([0, 1], range(8)), num_active_sms=10
            )
        self.assertEqual(
            (self.buffer._get_topology(), self.buffer._get_topology_masks()), before
        )
        self.buffer.runtime.fail_stage = False
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0, 1], range(8))
        )

    def test_failed_restage_keeps_previous_pending_transaction(self):
        self.dispatch()
        self.buffer._stage_topology(active_ranks=self.ranks([0, 1], range(8)))
        pending = self.buffer.runtime.pending
        self.buffer.runtime.fail_stage = True
        with self.assertRaisesRegex(ValueError, "stage refusal"):
            self.buffer._stage_topology(
                active_ranks=self.ranks([0], range(4)), num_active_sms=10
            )
        self.assertEqual(self.buffer.runtime.pending, pending)
        self.buffer.runtime.fail_stage = False
        self.buffer._commit_topology()
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0, 1], range(8))
        )
        self.buffer.set_active_ranks(self.ranks([0], range(16)))
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0], range(16))
        )

    def test_refused_commit_does_not_publish_staged_manual_policy(self):
        self.dispatch()
        before = self.buffer._get_topology(), self.buffer._get_topology_masks()
        self.buffer._stage_topology(
            active_ranks=self.ranks([0, 1], range(8)), num_active_sms=10
        )
        self.buffer.runtime.fail_commit = True
        with self.assertRaisesRegex(ValueError, "mapping refusal"):
            self.buffer._commit_topology()
        self.assertEqual(
            (self.buffer._get_topology(), self.buffer._get_topology_masks()), before
        )
        self.buffer.runtime.fail_commit = False
        self.buffer._stage_topology(active_ranks=self.ranks([0, 1], range(8)))
        self.buffer._commit_topology()
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0, 1], range(8))
        )

    def test_disconnect_finishes_retirement_and_preserves_membership_and_sm_policy(
        self,
    ):
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                self.buffer.connect_ranks(list(range(8, 16)))
                self.buffer.set_active_ranks(list(range(16)))
                self.buffer.set_num_active_sms(10)
                if automatic:
                    self.dispatch()
                self.buffer.set_active_ranks(list(range(8)))
                before = self.buffer._get_topology(), self.buffer._auto_sm_config
                commits = self.buffer.runtime.commit_calls
                self.buffer.disconnect_ranks(list(range(8, 16)))
                self.assertEqual(self.buffer.runtime.commit_calls, commits + 1)
                self.assertFalse(self.buffer.runtime.staged_views)
                self.assertEqual(self.buffer.get_connected_ranks(), list(range(1, 8)))
                self.assertEqual(self.buffer.get_active_ranks(), list(range(8)))
                self.assertEqual(
                    (self.buffer._get_topology(), self.buffer._auto_sm_config), before
                )
                self.buffer.set_active_ranks(list(range(4)))
                self.assertEqual(
                    self.buffer._get_topology()[1],
                    self.expected_sms([0], range(4)) if automatic else 10,
                )

    def test_empty_disconnect_does_not_commit_pending_connections(self):
        runtime = self.buffer.runtime
        runtime.disconnect_ranks = Mock(wraps=runtime.disconnect_ranks)
        for prepared in (False, True):
            with self.subTest(prepared=prepared):
                if prepared:
                    self.buffer.connect_ranks([16])
                commits = runtime.commit_calls
                before = self.buffer._get_topology(), self.buffer._auto_sm_config
                self.buffer.disconnect_ranks([])
                self.assertEqual(runtime.commit_calls, commits)
                self.assertEqual(runtime.staged_views, prepared)
                self.assertEqual(
                    (self.buffer._get_topology(), self.buffer._auto_sm_config), before
                )
        runtime.disconnect_ranks.assert_not_called()

    def test_dispatch_policy_change_cannot_overwrite_pending_transaction(self):
        self.dispatch()
        self.buffer._stage_topology(active_ranks=self.ranks([0, 1], range(8)))
        pending = self.buffer.runtime.pending
        dispatch_calls = self.buffer.runtime.dispatch_calls
        with self.assertRaisesRegex(AssertionError, "pending"):
            self.dispatch(num_sms=self.buffer._get_topology()[1])
        self.assertEqual(self.buffer.runtime.pending, pending)
        self.assertEqual(self.buffer.runtime.dispatch_calls, dispatch_calls)
        self.buffer._commit_topology()
        self.buffer.set_active_ranks(self.ranks([0], range(16)))
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0], range(16))
        )

    def test_single_rank_warmup_with_more_topk_than_local_slots(self):
        Buffer, cuda, _ = load_buffer()
        buffer = Buffer(
            world_rank=7,
            max_num_scaleout_ranks=2,
            max_num_scaleup_ranks=4,
            num_experts_per_rank=1,
            num_max_tokens_per_rank=16,
            hidden=256,
            num_topk=8,
            store=object(),
        )
        topk_idx = SimpleNamespace(shape=(16, 8))
        buffer.dispatch(object(), topk_idx, do_cpu_sync=False)
        self.assertEqual(buffer.get_connected_ranks(), [])
        self.assertEqual(buffer.get_active_ranks(), [7])
        self.assertEqual(buffer._get_topology(), ("direct", buffer.num_launch_sms))
        commits = buffer.runtime.commit_calls
        cuda.capturing = True
        buffer.dispatch(object(), topk_idx, do_cpu_sync=False)
        self.assertEqual(buffer.runtime.commit_calls, commits)
        self.assertEqual(buffer.runtime.dispatch_calls, 2)
        cuda.capturing = False
        buffer.connect_ranks(list(range(7)))
        self.assertEqual(buffer.get_active_ranks(), [7])
        buffer.set_active_ranks(list(range(8)))
        self.assertEqual(buffer.get_active_ranks(), list(range(8)))
        self.assertEqual(buffer._get_topology()[0], "hybrid")

    def test_single_rank_retunes_and_inactive_rank_preserves_auto_policy(
        self,
    ):
        self.dispatch()
        self.buffer.set_active_ranks(self.ranks([0], [0]))
        self.assertEqual(self.buffer._get_topology()[1], self.expected_sms([0], [0]))
        current = self.buffer._get_topology()[1]
        model = self.buffer._get_theoretical_num_sms
        with patch.object(
            self.buffer,
            "_get_theoretical_num_sms",
            side_effect=AssertionError("unexpected estimate"),
        ):
            self.buffer.set_active_ranks(self.ranks([1], range(8)))
            self.assertEqual(self.buffer._get_topology()[1], current)
        self.assertEqual(self.buffer._get_theoretical_num_sms, model)
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(
            self.buffer._get_topology()[1], self.expected_sms([0, 1], range(8))
        )

    def test_count_change_during_capture_refuses_before_native_dispatch(self):
        self.dispatch(num_sms=10)
        dispatch_calls = self.buffer.runtime.dispatch_calls
        self.cuda.capturing = True
        with self.assertRaisesRegex(AssertionError, "capture"):
            self.dispatch(num_sms=0)
        self.assertEqual(self.buffer.runtime.dispatch_calls, dispatch_calls)
        self.assertEqual(self.buffer._get_topology()[1], 10)
        self.cuda.capturing = False
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.assertEqual(
            self.buffer._get_topology()[1],
            10,
            "a refused automatic capture must not replace the manual policy",
        )

    def test_default_elastic_plan_covers_single_rank_world(self):
        plan_source = Path(__file__).with_name("test_adaptive_elastic.py")
        tree = ast.parse(plan_source.read_text(), filename=str(plan_source))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"extent_of", "default_plan"}
        ]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future] + functions, type_ignores=[])
        )
        namespace = {
            "nixl_ep_v2": SimpleNamespace(
                _C=SimpleNamespace(min_num_active_ranks=MIN_ACTIVE_RANKS)
            )
        }
        exec(compile(module, str(plan_source), "exec"), namespace)
        self.assertEqual(MIN_ACTIVE_RANKS, 1)
        plan = namespace["default_plan"](8, list(range(16)))
        self.assertIn([0], plan)
        self.assertTrue(all(len(phase) >= MIN_ACTIVE_RANKS for phase in plan))


if __name__ == "__main__":
    unittest.main(verbosity=2)
