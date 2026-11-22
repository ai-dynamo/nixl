# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""Host-only regression tests for fixed constructor SM counts and topology activation.

The production Python class is AST-loaded with simulated native/CUDA boundaries.
No torch import, GPU kernel, distributed process, or cluster access is performed.
Set NIXL_EP_FIXED_SM_SOURCE to exercise another checkout's elastic.py.
"""

import ast
import functools
import importlib.util
import inspect
import os
import unittest
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, Mock, patch

_REPO_SOURCE = Path(__file__).resolve().parents[2] / "python/buffers/elastic.py"
SOURCE = Path(os.environ.get("NIXL_EP_FIXED_SM_SOURCE", str(_REPO_SOURCE)))


class FakeCuda:
    def __init__(self):
        self.capturing = False
        self.synchronizations = 0

    def is_current_stream_capturing(self):
        return self.capturing

    def synchronize(self):
        self.synchronizations += 1


class FakeRuntime:
    """Models native topology activation and prepared transport."""

    def __init__(self, *args, cuda=None):
        self.cuda = cuda
        self.num_launch_sms = args[-1]
        self.mode = 0
        self.staged_views = False
        self.connected = set()
        self.fail_validation = False
        self.fail_activation = False
        self.dispatch_calls = 0
        self.activation_calls = 0
        self.max_domains, self.max_positions = args[7:9]
        self.world_rank = args[9]
        self.prepared_mapping = self.committed_mapping = {self.world_rank: 0}
        self.masks = (1, 1)
        self.num_experts_per_rank = args[-2]

    def update_topology_masks(self, mode, domains, position_words, disconnect_ranks):
        if self.cuda.capturing:
            raise RuntimeError("topology activation inside CUDA graph capture")
        positions = sum(word << (64 * i) for i, word in enumerate(position_words))
        if self.fail_validation:
            raise ValueError("simulated topology validation refusal")
        if not (domains and positions):
            raise ValueError("invalid topology")
        if (
            domains.bit_length() > self.max_domains
            or positions.bit_length() > self.max_positions
        ):
            raise ValueError("world exceeds declared maximum")
        if mode not in (0, 1) or (mode == 0 and domains.bit_count() != 1):
            raise ValueError("invalid topology mode")
        mapping = (
            self.committed_mapping
            if disconnect_ranks and self.world_rank not in self.get_active_ranks()
            else self.prepared_mapping
        )
        active_slots = {
            domain * self.max_positions + position
            for domain in range(self.max_domains)
            for position in range(self.max_positions)
            if domains & (1 << domain) and positions & (1 << position)
        }
        active = {rank for rank, slot in mapping.items() if slot in active_slots}
        if len(active) != len(active_slots):
            raise ValueError("active topology slot has no rank mapping")
        if self.world_rank in active and active - {self.world_rank} - self.connected:
            raise ValueError("active peer is not connected")
        if self.world_rank in active and active.intersection(disconnect_ranks):
            raise ValueError("departing peer remains active in candidate topology")
        seen = set()
        for rank in disconnect_ranks:
            if not 0 <= rank < self.max_domains * self.max_positions:
                raise ValueError("invalid peer world rank")
            if rank == self.world_rank:
                raise ValueError("invalid peer world rank: self")
            if rank not in self.connected:
                raise ValueError("peer is not connected")
            if rank in seen:
                raise ValueError("duplicate peer world rank")
            seen.add(rank)
        # Native validation completes before any logical peer or prepared view changes.
        candidate = (mode, domains, positions, mapping)
        if disconnect_ranks:
            self.connected.difference_update(disconnect_ranks)
            self.staged_views = True
            try:
                self._activate(candidate)
            except Exception as error:
                raise SystemExit("fatal peer retirement failure") from error
        else:
            self._activate(candidate)

    def _activate(self, candidate):
        if self.fail_activation:
            raise ValueError("simulated mapping refusal before activation")
        self.mode, domains, positions, mapping = candidate
        self.masks = (domains, positions)
        self.committed_mapping = dict(mapping)
        self.staged_views = False
        self.activation_calls += 1

    def get_mode(self):
        return self.mode

    def get_topology_masks(self):
        domains, positions = self.masks
        return domains, [
            (positions >> (64 * i)) & ((1 << 64) - 1)
            for i in range(max(1, (positions.bit_length() + 63) // 64))
        ]

    def get_local_agent_name(self):
        return "local"

    def get_connected_ranks(self):
        return sorted(self.connected)

    def get_active_ranks(self):
        domains, positions = self.masks
        return sorted(
            rank
            for rank, slot in self.committed_mapping.items()
            if domains & (1 << (slot // self.max_positions))
            and positions & (1 << (slot % self.max_positions))
        )

    def get_local_peer_info(self):
        return 4096, 0, self.world_rank, 2097152

    def connect_ranks(self, ranks, remote_agent_names, peer_info, world_to_slot):
        self.connected.update(ranks)
        self.prepared_mapping = {
            rank: slot for rank, slot in enumerate(world_to_slot) if slot >= 0
        }
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


def fabric_record(rank, domain):
    return dict(
        gpu_uuid=f"GPU-{rank:032x}",
        host_id="host",
        nvlink_peers=[],
        fabric=dict(
            cluster_uuid="cluster",
            clique_id=domain,
            state=3,
            status=0,
            handle_supported=True,
        ),
    )


def load_buffer(physical_domains=None):
    """Execute original class methods; replace only external boundaries."""
    cuda = FakeCuda()
    local_runtime = None

    def native_factory(*args):
        nonlocal local_runtime
        local_runtime = FakeRuntime(*args, cuda=cuda)
        return local_runtime

    def device_record(rank):
        domain = (
            rank // local_runtime.max_positions
            if physical_domains is None
            else physical_domains[rank]
        )
        return fabric_record(rank, domain)

    def validate_sizing(max_domains, max_positions, num_tokens, hidden, *args):
        if num_tokens <= 0 or hidden <= 0:
            raise ValueError("num_max_tokens_per_rank and hidden must be positive")
        return DEFAULT

    def exchange_records(store, rank, peers, value, **kwargs):
        return nullcontext(
            {
                peer: (
                    "peer",
                    (4096, peer, peer, 2097152),
                    device_record(peer),
                    {},
                    {},
                )
                for peer in peers
            }
        )

    spec = importlib.util.spec_from_file_location(
        "fixed_sm.nixl_fabric", SOURCE.parents[1] / "utils/nixl_fabric.py"
    )
    fabric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fabric)
    spec = importlib.util.spec_from_file_location(
        "fixed_sm.nixl_topology", SOURCE.parents[1] / "utils/nixl_topology.py"
    )
    topology = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"fixed_sm.nixl_fabric": fabric}):
        spec.loader.exec_module(topology)

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
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {
        "os": os,
        "functools": functools,
        "uuid": uuid,
        "torch": SimpleNamespace(cuda=cuda, Tensor=Mock),
        "_C": SimpleNamespace(
            ElasticBuffer=native_factory,
            calculate_elastic_buffer_size=Mock(
                side_effect=validate_sizing, return_value=2097152
            ),
        ),
        "check_torch_deterministic": lambda: None,
        "EventOverlap": lambda event: event,
        "exchange_records": exchange_records,
        "inspect_fabric_device": lambda: device_record(local_runtime.world_rank),
        "build_rank_mapping": topology.build_rank_mapping,
        "_MODE_NAMES": topology._MODE_NAMES,
        "_mode_code": topology._mode_code,
        "masks_of": topology.masks_of,
        "mode_of": topology.mode_of,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["ElasticBuffer"], cuda, namespace


class FixedSMHostTests(unittest.TestCase):
    def setUp(self):
        self.Buffer, self.cuda, self.namespace = load_buffer()
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
        self.buffer.connect_ranks(list(range(1, 16)))
        self.buffer.set_active_ranks(list(range(16)))
        self.cuda.synchronizations = 0

    @staticmethod
    def ranks(domains, positions):
        return [domain * 16 + position for domain in domains for position in positions]

    def dispatch(self, num_topk=8):
        return self.buffer.dispatch(
            object(), SimpleNamespace(shape=(16, num_topk)), do_cpu_sync=False
        )

    def snapshot(self):
        return (
            self.buffer._get_mode(),
            self.buffer._get_topology_masks(),
            self.buffer.num_launch_sms,
        )

    def retirement_snapshot(self):
        return (
            self.snapshot(),
            self.buffer.get_connected_ranks(),
            self.buffer.runtime.staged_views,
            self.buffer.runtime.activation_calls,
        )

    def test_constructor_count_is_forwarded_immutable_and_kept_by_dispatch(self):
        for count in (2, 12, 24, 64):
            with self.subTest(count=count):
                buffer = self.Buffer(
                    0,
                    1,
                    1,
                    16,
                    256,
                    num_experts_per_rank=16,
                    num_topk=32,
                    num_launch_sms=count,
                )
                self.assertEqual(buffer.num_launch_sms, count)
                self.assertEqual(buffer.runtime.num_launch_sms, count)
                self.assertEqual(buffer.runtime.num_experts_per_rank, 16)
                self.assertEqual(buffer._get_mode(), "direct")
                with self.assertRaises(AttributeError):
                    buffer.num_launch_sms = 4
                for num_topk in (1, 8):
                    _, _, _, handle, _ = buffer.dispatch(
                        object(),
                        SimpleNamespace(shape=(16, num_topk)),
                        do_cpu_sync=False,
                    )
                    self.assertFalse(hasattr(handle, "num_sms"))
                    self.assertEqual(buffer.num_launch_sms, count)
                    self.assertEqual(buffer.runtime.num_launch_sms, count)
                self.assertEqual(buffer.runtime.activation_calls, 0)

    def test_invalid_count_fails_before_sizing_or_native_allocation(self):
        native = self.namespace["_C"]
        with (
            patch.object(native, "ElasticBuffer") as construct,
            patch.object(native, "calculate_elastic_buffer_size") as sizing,
        ):
            for count in (0, -2, 1, 3, 66, True, False, 2.0, None):
                with self.subTest(count=count):
                    with self.assertRaisesRegex(ValueError, "num_launch_sms"):
                        self.Buffer(
                            0,
                            1,
                            1,
                            16,
                            256,
                            num_experts_per_rank=16,
                            num_topk=32,
                            num_launch_sms=count,
                        )
            construct.assert_not_called()
            sizing.assert_not_called()
        self.assertEqual(self.cuda.synchronizations, 0)

    def test_fixed_count_has_no_runtime_policy_or_handle_override(self):
        count = inspect.signature(self.Buffer).parameters["num_launch_sms"]
        self.assertEqual(count.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(count.default, 24)
        for method in (
            self.Buffer.dispatch,
            self.Buffer.combine,
            self.Buffer.set_active_ranks,
            self.Buffer._update_topology,
            self.namespace["EPHandle"],
        ):
            self.assertTrue(
                {"num_sms", "num_active_sms"}.isdisjoint(
                    inspect.signature(method).parameters
                )
            )
        for name in (
            "set_num_active_sms",
            "get_theoretical_num_sms",
            "_get_theoretical_num_sms",
            "_get_theoretical_num_sms_for_topology",
            "_auto_sms",
            "_last_num_topk",
            "_clamp_active_sms",
            "_select_active_sms",
            "_adaptive_active_sms",
            "_get_topology",
        ):
            self.assertFalse(hasattr(self.buffer, name), name)
        self.assertNotIn(
            "recv_count_is_exact",
            inspect.signature(self.namespace["EPHandle"]).parameters,
        )

    def test_membership_updates_only_mode_and_masks(self):
        self.buffer.connect_ranks(list(range(16, 24)))
        runtime = self.buffer.runtime
        with patch.object(
            runtime, "update_topology_masks", wraps=runtime.update_topology_masks
        ) as activate:
            self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        activate.assert_called_once_with(1, 3, [255], ())
        self.assertEqual(self.snapshot(), ("hybrid", (3, 255), 24))
        self.assertEqual(runtime.num_launch_sms, 24)
        self.assertEqual(runtime.dispatch_calls, 0)
        self.assertEqual(self.buffer.num_experts_per_rank, 16)

    def test_active_rank_types_are_checked_before_native_activation(self):
        before = self.retirement_snapshot()
        runtime = self.buffer.runtime
        with patch.object(
            runtime, "update_topology_masks", wraps=runtime.update_topology_masks
        ) as activate:
            for ranks in ([False], [0.0], [0, False], [0, 0.0]):
                with self.subTest(ranks=ranks), self.assertRaisesRegex(
                    ValueError, "integers"
                ):
                    self.buffer.set_active_ranks(ranks)
            activate.assert_not_called()
            self.assertEqual(self.retirement_snapshot(), before)
            self.buffer.set_active_ranks([0, 0])
        activate.assert_called_once_with(0, 1, [1], ())

    def test_invalid_connections_are_refused_before_store_exchange(self):
        before = self.retirement_snapshot()
        runtime = self.buffer.runtime
        exchange = Mock(wraps=self.namespace["exchange_records"])
        with patch.dict(self.namespace, exchange_records=exchange), patch.object(
            runtime, "connect_ranks", wraps=runtime.connect_ranks
        ) as connect:
            for ranks in ([16, 16], [0], [-1], [32]):
                with self.subTest(ranks=ranks), self.assertRaises(ValueError):
                    self.buffer.connect_ranks(ranks)
            exchange.assert_not_called()
            connect.assert_not_called()
            self.assertEqual(self.retirement_snapshot(), before)
            self.buffer.store = None
            self.buffer.connect_ranks([1, 1])
            exchange.assert_not_called()
            self.buffer.connect_ranks([1, 1, 16], store=object())
        self.assertEqual(exchange.call_args.args[2], [16])
        self.assertEqual(connect.call_args.args[0], [16])

    def test_repeated_shrink_expand_keeps_count_with_and_without_dispatch(self):
        self.buffer.connect_ranks(list(range(16, 32)))
        cases = [
            ([0, 1], range(8)),
            ([0], range(8)),
            ([0], range(16)),
            ([0], [0]),
            ([1], [0, 15]),
            ([0, 1], [0]),
            ([0], range(16)),
        ]
        for domains, positions in cases:
            with self.subTest(domains=domains, positions=list(positions)):
                self.buffer.set_active_ranks(self.ranks(domains, positions))
                before = self.snapshot()
                activations = self.buffer.runtime.activation_calls
                if 0 in self.buffer.get_active_ranks():
                    self.dispatch(num_topk=1)
                    self.dispatch(num_topk=8)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.buffer.runtime.activation_calls, activations)
                self.assertEqual(self.buffer.num_launch_sms, 24)
                self.assertEqual(self.buffer.runtime.num_launch_sms, 24)
                self.assertEqual(self.buffer.num_experts_per_rank, 16)
        self.assertEqual(self.cuda.synchronizations, 0)

    def test_position_mask_spans_multiple_words_without_a_count_field(self):
        buffer = self.Buffer(
            64,
            2,
            72,
            16,
            256,
            num_experts_per_rank=4,
            num_topk=32,
            num_launch_sms=12,
            store=object(),
        )
        buffer.connect_ranks([rank for rank in range(144) if rank != 64])
        runtime = buffer.runtime
        with patch.object(
            runtime, "update_topology_masks", wraps=runtime.update_topology_masks
        ) as activate:
            buffer.set_active_ranks([64, 71, 136, 143])
        activate.assert_called_once_with(1, 3, [0, 129], ())
        self.assertEqual(buffer.get_active_ranks(), [64, 71, 136, 143])
        self.assertEqual(buffer._get_topology_extent(), (2, 72))
        self.assertEqual(buffer.num_launch_sms, 12)

    def test_capture_dispatch_does_not_activate_topology(self):
        before = self.snapshot()
        activations = self.buffer.runtime.activation_calls
        self.cuda.capturing = True
        try:
            self.dispatch(num_topk=1)
            self.dispatch(num_topk=8)
            with self.assertRaisesRegex(RuntimeError, "capture"):
                self.buffer.set_active_ranks(list(range(8)))
        finally:
            self.cuda.capturing = False
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.buffer.runtime.dispatch_calls, 2)
        self.assertEqual(self.buffer.runtime.activation_calls, activations)

    def test_discovered_domains_determine_mode_independently_of_rank_numbers(self):
        Buffer, _, _ = load_buffer({0: 7, 3: 7, 1: 2, 2: 2})
        buffer = Buffer(
            2, 2, 2, 16, 256, num_experts_per_rank=4, num_topk=32, store=object()
        )
        buffer.connect_ranks([3, 0, 1])
        self.assertEqual(buffer._prepared_mapping, {0: 0, 3: 1, 1: 2, 2: 3})
        self.assertEqual(buffer.get_active_ranks(), [2])
        self.assertEqual(buffer._get_topology_masks(), (1, 1))
        buffer.runtime.fail_activation = True
        with self.assertRaisesRegex(ValueError, "refusal"):
            buffer.set_active_ranks([0, 1, 2, 3])
        self.assertEqual(buffer.get_active_ranks(), [2])
        self.assertEqual(buffer.runtime.committed_mapping, {2: 0})
        buffer.runtime.fail_activation = False
        for ranks, mode in (
            ([0, 1, 2, 3], "hybrid"),
            ([0, 3], "direct"),
            ([1, 2], "direct"),
            ([0, 1], "hybrid"),
        ):
            with self.subTest(ranks=ranks):
                buffer.set_active_ranks(ranks)
                self.assertEqual(buffer.get_active_ranks(), ranks)
                self.assertEqual(buffer._get_mode(), mode)
                self.assertEqual(buffer.num_launch_sms, 24)

    def test_failed_dispatch_leaves_constructor_count_and_topology_unchanged(self):
        before = self.snapshot()
        activations = self.buffer.runtime.activation_calls
        with patch.object(
            self.buffer.runtime,
            "dispatch",
            side_effect=ValueError("simulated dispatch failure"),
        ):
            with self.assertRaisesRegex(ValueError, "dispatch failure"):
                self.dispatch(num_topk=1)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.buffer.runtime.activation_calls, activations)

    def test_expert_alignment_is_checked_before_native_dispatch(self):
        runtime = self.buffer.runtime
        with patch.object(runtime, "dispatch", wraps=runtime.dispatch) as dispatch:
            for alignment in (0, -1):
                with self.subTest(alignment=alignment), self.assertRaisesRegex(
                    ValueError, "expert_alignment must be positive"
                ):
                    self.buffer.dispatch(object(), object(), expert_alignment=alignment)
            dispatch.assert_not_called()
            self.buffer.dispatch(object(), object(), expert_alignment=3)
        self.assertEqual(dispatch.call_args.args[6], 3)

    def test_dispatch_receive_counts_determine_native_combine_exactness(self):
        topk_idx = SimpleNamespace(shape=(16, 8))
        native_result = list(self.buffer.runtime.dispatch(None, None, topk_idx))
        num_experts = self.buffer.num_experts_per_rank
        for do_cpu_sync, recv_count, expert_counts in (
            (True, 0, [0] * num_experts),
            (True, 16, [1] * num_experts),
            (False, 16, []),
        ):
            with self.subTest(do_cpu_sync=do_cpu_sync, recv_count=recv_count):
                native_result[5:7] = [recv_count, expert_counts]
                with (
                    patch.object(
                        self.buffer.runtime, "dispatch", return_value=native_result
                    ) as dispatch,
                    patch.object(
                        self.buffer.runtime,
                        "combine",
                        create=True,
                        return_value=(object(), None),
                    ) as combine,
                ):
                    recv_x, _, _, handle, _ = self.buffer.dispatch(
                        object(), topk_idx, do_cpu_sync=do_cpu_sync
                    )
                    self.assertIs(dispatch.call_args.args[12], do_cpu_sync)
                    self.assertEqual(handle.num_recv_tokens, recv_count)
                    self.assertEqual(
                        handle.num_recv_tokens_per_expert_list, expert_counts
                    )
                    self.assertFalse(hasattr(handle, "recv_count_is_exact"))
                    self.assertFalse(hasattr(handle, "num_sms"))
                    self.buffer.combine(recv_x, handle)
                    self.assertIs(combine.call_args.args[8], do_cpu_sync)
                    self.assertEqual(len(combine.call_args.args), 15)

    def test_refused_topology_preserves_world_count_and_prepared_transport(self):
        for failure in ("fail_validation", "fail_activation"):
            with self.subTest(failure=failure):
                self.buffer.set_active_ranks(list(range(16)))
                self.buffer.connect_ranks(list(range(16, 24)))
                self.buffer.disconnect_ranks(list(range(16, 24)))
                self.buffer.connect_ranks(self.ranks([1], range(8)))
                before = self.snapshot()
                activations = self.buffer.runtime.activation_calls
                setattr(self.buffer.runtime, failure, True)
                with self.assertRaisesRegex(ValueError, "refusal"):
                    self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.buffer.runtime.activation_calls, activations)
                self.assertTrue(self.buffer.runtime.staged_views)
                setattr(self.buffer.runtime, failure, False)
                self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
                self.assertFalse(self.buffer.runtime.staged_views)
                self.assertEqual(self.buffer.runtime.activation_calls, activations + 1)
                self.assertEqual(self.snapshot(), ("hybrid", (3, 255), 24))

    def test_disconnect_finishes_retirement_and_preserves_membership_and_count(self):
        self.buffer.set_active_ranks(list(range(8)))
        before = self.snapshot()
        activations = self.buffer.runtime.activation_calls
        self.buffer.disconnect_ranks(list(range(8, 16)))
        self.assertEqual(self.buffer.runtime.activation_calls, activations + 1)
        self.assertFalse(self.buffer.runtime.staged_views)
        self.assertEqual(self.buffer.get_connected_ranks(), list(range(1, 8)))
        self.assertEqual(self.buffer.get_active_ranks(), list(range(8)))
        self.assertEqual(self.snapshot(), before)
        self.buffer.set_active_ranks(list(range(4)))
        self.assertEqual(self.buffer.num_launch_sms, 24)

    def test_disconnect_shrinks_domains_positions_and_full_world_to_singleton(self):
        full = list(range(32))
        cases = (
            (0, list(range(16)), "direct", (1, 65535)),
            (16, list(range(16, 32)), "direct", (2, 65535)),
            (0, self.ranks([0, 1], range(8)), "hybrid", (3, 255)),
            (0, self.ranks([0, 1], [0, 7, 15]), "hybrid", (3, 32897)),
            (0, [0], "direct", (1, 1)),
            (31, [31], "direct", (2, 32768)),
        )
        for local_rank, remaining, mode, masks in cases:
            with self.subTest(local_rank=local_rank, remaining=remaining):
                buffer = self.Buffer(
                    world_rank=local_rank,
                    max_num_scaleout_ranks=2,
                    max_num_scaleup_ranks=16,
                    num_max_tokens_per_rank=16,
                    hidden=256,
                    num_experts_per_rank=16,
                    num_topk=32,
                    num_launch_sms=12,
                    store=object(),
                )
                buffer.connect_ranks([rank for rank in full if rank != local_rank])
                buffer.set_active_ranks(full)
                leaving = sorted(set(full) - set(remaining))
                runtime = buffer.runtime
                activations = runtime.activation_calls
                with patch.object(
                    runtime,
                    "update_topology_masks",
                    wraps=runtime.update_topology_masks,
                ) as activate:
                    buffer.disconnect_ranks(leaving)
                activate.assert_called_once_with(
                    0 if mode == "direct" else 1, masks[0], [masks[1]], leaving
                )
                self.assertEqual(runtime.activation_calls, activations + 1)
                self.assertEqual(buffer.get_active_ranks(), remaining)
                self.assertEqual(buffer._get_mode(), mode)
                self.assertEqual(buffer._get_topology_masks(), masks)
                self.assertEqual(
                    buffer.get_connected_ranks(),
                    [rank for rank in remaining if rank != local_rank],
                )
                self.assertFalse(runtime.staged_views)
                self.assertEqual(buffer.num_launch_sms, 12)
                self.assertEqual(runtime.num_launch_sms, 12)

    def test_disconnect_derives_mode_for_a_surviving_rank(self):
        self.buffer._update_topology(mode="hybrid")
        self.assertEqual(self.buffer._get_mode(), "hybrid")
        self.buffer.disconnect_ranks(list(range(8, 16)))
        self.assertEqual(self.buffer.get_active_ranks(), list(range(8)))
        self.assertEqual(self.buffer._get_mode(), "direct")
        self.assertEqual(self.buffer.num_launch_sms, 24)

    def test_inactive_disconnect_preserves_descriptor_and_explicit_mode(self):
        self.buffer.connect_ranks(list(range(16, 24)))
        # Rank zero is sitting out a forced-hybrid world in a single other domain.
        self.buffer._update_topology(mode="hybrid", active_ranks=list(range(16, 24)))
        before = self.snapshot()
        runtime = self.buffer.runtime
        activations = runtime.activation_calls
        leaving = self.buffer.get_connected_ranks()
        with patch.object(
            runtime, "update_topology_masks", wraps=runtime.update_topology_masks
        ) as activate:
            self.buffer.disconnect_ranks(leaving)
        activate.assert_called_once_with(1, 2, [255], leaving)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.buffer.get_active_ranks(), list(range(16, 24)))
        self.assertEqual(self.buffer.get_connected_ranks(), [])
        self.assertEqual(runtime.activation_calls, activations + 1)
        self.assertFalse(runtime.staged_views)

    def test_nonproduct_disconnect_preserves_world_peers_and_prepared_views(self):
        self.buffer.connect_ranks(list(range(16, 24)))
        self.buffer.set_active_ranks(self.ranks([0, 1], range(8)))
        self.buffer.connect_ranks([24])
        before = self.retirement_snapshot()
        runtime = self.buffer.runtime
        with patch.object(
            runtime, "update_topology_masks", wraps=runtime.update_topology_masks
        ) as activate:
            with self.assertRaisesRegex(ValueError, "product"):
                self.buffer.disconnect_ranks([1])
        activate.assert_not_called()
        self.assertEqual(self.retirement_snapshot(), before)

    def test_invalid_disconnect_validates_entire_request_before_peer_mutation(self):
        self.buffer.connect_ranks([17])
        cases = (
            ([1, 1], "duplicate"),
            ([1, 16], "not connected"),
            ([1, 0], "self"),
            ([1, -1], "invalid peer"),
            ([1, 32], "invalid peer"),
        )
        for leaving, message in cases:
            with self.subTest(leaving=leaving):
                before = self.retirement_snapshot()
                with self.assertRaisesRegex(ValueError, message):
                    self.buffer.disconnect_ranks(leaving)
                self.assertEqual(self.retirement_snapshot(), before)

    def test_refused_disconnect_topology_preserves_world_peers_and_prepared_views(self):
        self.buffer.connect_ranks([16])
        before = self.retirement_snapshot()
        self.buffer.runtime.fail_validation = True
        try:
            with self.assertRaisesRegex(ValueError, "refusal"):
                self.buffer.disconnect_ranks(list(range(8, 16)))
        finally:
            self.buffer.runtime.fail_validation = False
        self.assertEqual(self.retirement_snapshot(), before)

    def test_disconnect_capture_refusal_precedes_peer_mutation(self):
        self.buffer.connect_ranks([16])
        before = self.retirement_snapshot()
        self.cuda.capturing = True
        try:
            with self.assertRaisesRegex(RuntimeError, "capture"):
                self.buffer.disconnect_ranks(list(range(8, 16)))
        finally:
            self.cuda.capturing = False
        self.assertEqual(self.retirement_snapshot(), before)

    def test_disconnect_activation_failure_is_fatal(self):
        self.buffer.connect_ranks([16])
        self.buffer.runtime.fail_activation = True
        with self.assertRaisesRegex(SystemExit, "fatal peer retirement"):
            self.buffer.disconnect_ranks(list(range(8, 16)))

    def test_empty_disconnect_does_not_activate_prepared_connections(self):
        runtime = self.buffer.runtime
        runtime.update_topology_masks = Mock(wraps=runtime.update_topology_masks)
        for prepared in (False, True):
            with self.subTest(prepared=prepared):
                if prepared:
                    self.buffer.connect_ranks([16])
                before = self.retirement_snapshot()
                self.cuda.capturing = True
                try:
                    self.buffer.disconnect_ranks([])
                finally:
                    self.cuda.capturing = False
                self.assertEqual(runtime.staged_views, prepared)
                self.assertEqual(self.retirement_snapshot(), before)
        runtime.update_topology_masks.assert_not_called()

    def test_single_rank_warmup_with_more_topk_than_local_slots(self):
        buffer = self.Buffer(
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
        self.assertEqual(buffer._get_mode(), "direct")
        activations = buffer.runtime.activation_calls
        self.cuda.capturing = True
        try:
            buffer.dispatch(object(), topk_idx, do_cpu_sync=False)
        finally:
            self.cuda.capturing = False
        self.assertEqual(buffer.runtime.activation_calls, activations)
        self.assertEqual(buffer.runtime.dispatch_calls, 2)
        buffer.connect_ranks(list(range(7)))
        self.assertEqual(buffer.get_active_ranks(), [7])
        buffer.set_active_ranks(list(range(8)))
        self.assertEqual(buffer.get_active_ranks(), list(range(8)))
        self.assertEqual(buffer._get_mode(), "hybrid")
        self.assertEqual(buffer.num_launch_sms, 24)

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
        self.assertEqual(buffer._get_mode(), "hybrid")
        buffer.set_active_ranks(self.ranks([0], range(8)))
        self.assertEqual(buffer._get_mode(), "direct")
        buffer.set_topology_forced("hybrid")
        self.assertEqual(buffer._get_mode(), "hybrid")
        self.assertEqual(buffer.num_launch_sms, 24)

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
        namespace = {}
        exec(compile(module, str(plan_source), "exec"), namespace)
        plan = namespace["default_plan"](8, list(range(16)))
        self.assertIn([0], plan)
        self.assertTrue(all(plan))


if __name__ == "__main__":
    unittest.main(verbosity=2)
