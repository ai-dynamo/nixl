# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise bootstrap coordination with a real CPU Store and mocked GPU discovery."""

import importlib.util
import inspect
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch.distributed as dist
from test_fixed_sm_host import load_buffer


def parallel(functions):
    results, failures = [None] * len(functions), []

    def run(index, function):
        try:
            results[index] = function()
        except BaseException as error:
            failures.append(error)

    threads = [
        threading.Thread(target=run, args=(i, function), daemon=True)
        for i, function in enumerate(functions)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    for thread in threads:
        thread.join(max(0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        raise AssertionError("Bootstrap worker did not finish within five seconds")
    if failures:
        raise failures[0]
    return results


class ClosingStore:
    """Refuse further operations when the process owning a private Store returns."""

    def __init__(self, store):
        self.store, self.closed = store, False

    def __getattr__(self, name):
        method = getattr(self.store, name)

        def call(*args, **kwargs):
            if self.closed:
                raise AssertionError(f"Store.{name} called after owner returned")
            return method(*args, **kwargs)

        return call


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.local = threading.local()
        self.fabric_records = {}

        def validate(records, positions):
            self.fabric_records[self.local.rank] = tuple(r["rank"] for r in records)
            positions = len(records) if positions is None else positions
            return len(records) // positions, positions

        path = Path(__file__).resolve().parents[2] / "python/utils/nixl_bootstrap.py"
        spec = importlib.util.spec_from_file_location("bootstrap_host.bootstrap", path)
        assert spec is not None and spec.loader is not None
        self.module = importlib.util.module_from_spec(spec)
        fabric = SimpleNamespace(
            inspect_fabric_device=lambda: {"rank": self.local.rank},
            validate_fabric_records=validate,
        )
        with patch.dict("sys.modules", {"bootstrap_host.nixl_fabric": fabric}):
            spec.loader.exec_module(self.module)
        self.store = dist.HashStore()

    def exchange(self, rank, peers, value, store=None):
        with self.module.exchange_records(
            self.store if store is None else store,
            rank,
            peers,
            value,
            timeout_seconds=2,
        ) as records:
            return records

    def bootstrap(self, rank, ranks, store=None):
        self.local.rank = ranks[rank]
        return self.module.NixlBootstrap(
            self.store if store is None else store,
            rank,
            len(ranks),
            world_ranks=ranks,
            num_scaleup_ranks=len(ranks),
            timeout_seconds=2,
        )

    def test_disjoint_cohorts_share_store_without_a_bootstrap_id(self):
        cohorts = ((2, 5), (9, 12))

        def construct(index, ranks):
            bootstrap = self.bootstrap(index, ranks)
            records = bootstrap._all_gather("identity", ranks[index])
            bootstrap.barrier("ready")
            return records

        results = parallel(
            [
                lambda index=index, ranks=ranks: construct(index, ranks)
                for ranks in cohorts
                for index in range(2)
            ]
        )
        self.assertEqual(results, [[2, 5], [2, 5], [9, 12], [9, 12]])
        for ranks in cohorts:
            for rank in ranks:
                self.assertEqual(self.fabric_records[rank], ranks)
        self.assertEqual(self.store.num_keys(), 0)

    def test_singleton_at_nonzero_world_rank_needs_no_other_participant(self):
        bootstrap = self.bootstrap(0, [17])
        info = (4096, 0, 17, 2097152)
        self.assertEqual(bootstrap._all_gather("identity", info), [info])
        bootstrap.barrier("ready")
        self.assertEqual(self.fabric_records[17], (17,))
        self.assertEqual(self.store.num_keys(), 0)

    def test_completed_exchanges_reuse_rank_keys_without_stale_metadata(self):
        for generation in range(3):
            results = parallel(
                [
                    lambda rank=rank: self.exchange(
                        rank, [7 - rank], (rank, generation)
                    )
                    for rank in (2, 5)
                ]
            )
            self.assertEqual(results, [{5: (5, generation)}, {2: (2, generation)}])
            self.assertEqual(self.store.num_keys(), 0)

    def test_metadata_survives_until_the_slowest_requested_peer_reads_it(self):
        paused, release, first_peer_done = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        owner_done = threading.Event()
        actual = self.store

        class SlowReader:
            def __getattr__(self, name):
                return getattr(actual, name)

            def compare_set(self, key, expected, desired):
                if key == "NIXL_EP/0":
                    paused.set()
                    if not release.wait(2):
                        raise AssertionError("Test did not release the slow reader")
                return actual.compare_set(key, expected, desired)

        def owner():
            result = self.exchange(0, [1, 2], b"owner")
            owner_done.set()
            return result

        def first_peer():
            result = self.exchange(1, [0], b"first")
            first_peer_done.set()
            return result

        def inspect_while_paused():
            try:
                self.assertTrue(paused.wait(2))
                self.assertTrue(first_peer_done.wait(2))
                self.assertFalse(owner_done.is_set())
                self.assertTrue(actual.check(["NIXL_EP/0"]))
            finally:
                release.set()

        results = parallel(
            [
                owner,
                first_peer,
                lambda: self.exchange(2, [0], b"slow", SlowReader()),
                inspect_while_paused,
            ]
        )
        self.assertEqual(
            results[:3], [{1: b"first", 2: b"slow"}, {0: b"owner"}, {0: b"owner"}]
        )
        self.assertEqual(actual.num_keys(), 0)

    def test_timeout_cleans_up_the_local_publication(self):
        with self.assertRaises(RuntimeError):
            with self.module.exchange_records(
                self.store, 2, [5], b"local", timeout_seconds=0.05
            ):
                self.fail("A missing peer must not complete the exchange")
        self.assertEqual(self.store.num_keys(), 0)

    def test_withdrawn_metadata_fails_without_waiting_for_the_store_timeout(self):
        actual, publisher_done = self.store, threading.Event()
        actual.set_timeout(timedelta(seconds=1))

        class DelayedReader:
            def __getattr__(self, name):
                return getattr(actual, name)

            def wait(self, keys, timeout):
                actual.wait(keys, timeout)
                if keys == ["NIXL_EP/0"] and not publisher_done.wait(2):
                    raise AssertionError("Publisher did not finish timeout cleanup")

        def publisher():
            try:
                with self.assertRaises(RuntimeError):
                    with self.module.exchange_records(
                        actual, 0, [1], b"owner", timeout_seconds=0.1
                    ):
                        self.fail("The delayed reader has not acknowledged the record")
            finally:
                publisher_done.set()

        def reader():
            with self.assertRaisesRegex(
                RuntimeError, "Peer 0 withdrew its Store metadata"
            ):
                self.exchange(1, [0], b"reader", DelayedReader())

        parallel([publisher, reader])
        self.assertEqual(actual.num_keys(), 0)

    def test_context_failure_cleans_up_after_peers_have_read(self):
        def failing_peer():
            with self.assertRaisesRegex(ValueError, "connection refused"):
                with self.module.exchange_records(
                    self.store, 2, [5], b"local", timeout_seconds=2
                ):
                    raise ValueError("connection refused")

        parallel([failing_peer, lambda: self.exchange(5, [2], b"remote")])
        self.assertEqual(self.store.num_keys(), 0)

    def test_final_barrier_allows_the_store_owner_to_return_first(self):
        store = ClosingStore(self.store)

        def participant(index):
            bootstrap = self.bootstrap(index, [3, 8, 11], store)
            bootstrap.barrier("ready")
            if index == 0:
                store.closed = True

        parallel([lambda index=index: participant(index) for index in range(3)])
        self.assertTrue(store.closed)
        self.assertEqual(self.store.num_keys(), 0)

    def test_explicit_store_needs_no_namespace_or_process_group(self):
        self.assertIs(self.module.resolve_bootstrap_store(None, self.store), self.store)

    def test_consecutive_bootstrap_constructions_do_not_mix_fabric_records(self):
        slow_cleanup, fast_boundary, second_started, release = (
            threading.Event() for _ in range(4)
        )
        actual = self.store

        class DelayedCleanup:
            def __getattr__(self, name):
                return getattr(actual, name)

            def set(self, key, value):
                if "/barrier/" in key and key.endswith("/fabric/0/arrived/2"):
                    fast_boundary.set()
                return actual.set(key, value)

            def delete_key(self, key):
                if key == "NIXL_EP/bootstrap/fabric/0/5" and not release.is_set():
                    slow_cleanup.set()
                    if not release.wait(2):
                        raise AssertionError("Test did not release fabric cleanup")
                return actual.delete_key(key)

        def validate(records, positions):
            self.assertEqual(
                {record["generation"] for record in records}, {self.local.generation}
            )
            return 1, positions

        self.module.inspect_fabric_device = lambda: {
            "rank": self.local.rank,
            "generation": self.local.generation,
        }
        self.module.validate_fabric_records = validate
        store = DelayedCleanup()

        def participant(index):
            for generation in range(2):
                self.local.generation = generation
                if index == 0 and generation == 1:
                    second_started.set()
                    fast_boundary.set()
                self.bootstrap(index, [2, 5], store)

        def inspect_while_paused():
            try:
                self.assertTrue(slow_cleanup.wait(2))
                self.assertTrue(fast_boundary.wait(2))
                self.assertFalse(second_started.is_set())
            finally:
                release.set()

        parallel([lambda: participant(0), lambda: participant(1), inspect_while_paused])
        self.assertEqual(actual.num_keys(), 0)

    def test_discovery_retry_does_not_consume_failed_barrier_keys(self):
        failed = [threading.Event(), threading.Event()]
        second_returned = threading.Event()

        def validate(records, positions):
            if self.local.rank == 2:
                if self.local.generation == 0:
                    for event in failed:
                        if not event.wait(2):
                            raise AssertionError(
                                "Expected the first discovery to time out"
                            )
                else:
                    self.assertFalse(
                        second_returned.wait(0.1),
                        "A peer returned using the previous discovery's barrier keys",
                    )
            return 1, positions

        self.module.validate_fabric_records = validate

        def participant(rank, generation):
            self.local.rank, self.local.generation = rank, generation
            if generation == 0 and rank != 2:
                try:
                    with self.assertRaises(RuntimeError):
                        self.module.NixlBootstrap(
                            self.store,
                            rank,
                            3,
                            num_scaleup_ranks=3,
                            timeout_seconds=0.1,
                        )
                finally:
                    failed[rank].set()
            else:
                result = self.bootstrap(rank, [0, 1, 2])
                if generation == 1 and rank != 2:
                    second_returned.set()
                return result.num_scaleout_ranks, result.num_scaleup_ranks

        parallel([lambda rank=rank: participant(rank, 0) for rank in range(3)])
        failed_keys = self.store.num_keys()
        self.assertGreater(failed_keys, 0)
        results = parallel(
            [lambda rank=rank: participant(rank, 1) for rank in range(3)]
        )
        self.assertEqual(results, [(1, 3)] * 3)
        self.assertEqual(self.store.num_keys(), failed_keys)

    def test_constructor_is_local_and_starts_with_only_its_own_rank(self):
        Buffer, cuda, _ = load_buffer()
        namespace = Buffer.__init__.__globals__

        class ForbiddenBoundary:
            def __getattr__(self, name):
                raise AssertionError(f"Constructor accessed distributed state: {name}")

        forbidden = Mock(side_effect=AssertionError("Constructor called bootstrap"))
        namespace["NixlBootstrap"] = forbidden
        namespace["resolve_bootstrap_store"] = forbidden
        namespace["dist"] = ForbiddenBoundary()
        native = namespace["_C"].ElasticBuffer
        namespace["_C"].ElasticBuffer = Mock(wraps=native)
        for store in (None, ForbiddenBoundary()):
            with self.subTest(store=store):
                buffer = Buffer(
                    5,
                    3,
                    4,
                    num_experts_per_rank=1,
                    num_max_tokens_per_rank=1,
                    hidden=256,
                    num_topk=1,
                    store=store,
                    explicitly_destroy=True,
                )
                self.assertIs(buffer.store, store)
                self.assertEqual(buffer.world_rank, 5)
                self.assertFalse(hasattr(buffer, "scaleout_rank_idx"))
                self.assertFalse(hasattr(buffer, "scaleup_rank_idx"))
                self.assertEqual(buffer._get_topology_masks(), (1, 1))
                self.assertEqual(buffer.get_active_ranks(), [5])
                self.assertIn(buffer.world_rank, buffer.get_active_ranks())
                self.assertEqual(buffer.runtime.connected, set())
                self.assertFalse(buffer.runtime.staged_views)
                self.assertEqual(buffer.runtime.activation_calls, 0)
                self.assertEqual(
                    buffer.num_bytes,
                    namespace["_C"].calculate_elastic_buffer_size.return_value,
                )
                self.assertEqual(
                    namespace["_C"].ElasticBuffer.call_args.args[0], buffer.num_bytes
                )
        self.assertEqual(namespace["_C"].ElasticBuffer.call_count, 2)
        self.assertEqual(namespace["_C"].calculate_elastic_buffer_size.call_count, 2)
        self.assertEqual(cuda.synchronizations, 2)
        forbidden.assert_not_called()

        obsolete = {
            "num_bytes",
            "num_nixl_channels",
            "group",
            "rank_idx",
            "num_ranks",
            "num_group_scaleup_ranks",
            "scaleout_rank_idx",
            "scaleup_rank_idx",
            "store_timeout_secs",
            "bootstrap_id",
        }
        self.assertTrue(obsolete.isdisjoint(inspect.signature(Buffer).parameters))
        for method in (
            "publish",
            "stage_topology",
            "commit_topology",
            "get_local_metadata",
            "get_local_agent_name",
            "get_local_peer_info",
        ):
            self.assertFalse(hasattr(Buffer, method))
        self.assertTrue(
            {"remote_mds", "peer_infos"}.isdisjoint(
                inspect.signature(Buffer.connect_ranks).parameters
            )
        )

    def test_constructor_sizes_workload_for_maximum_world_like_planning_query(self):
        Buffer, _, _ = load_buffer()
        native = Buffer.__init__.__globals__["_C"]
        native.ElasticBuffer = Mock(wraps=native.ElasticBuffer)
        native.calculate_elastic_buffer_size.return_value = 8 * 2097152
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                settings = dict(
                    max_num_scaleout_ranks=3,
                    max_num_scaleup_ranks=4,
                    num_max_tokens_per_rank=16,
                    hidden=7168,
                    num_topk=8,
                    use_fp8_dispatch=quantized,
                    allow_multiple_reduction=False,
                )
                buffer = Buffer(world_rank=5, num_experts_per_rank=4, **settings)
                native.calculate_elastic_buffer_size.assert_called_with(
                    3, 4, 16, 7168, 8, quantized, False
                )
                self.assertEqual(buffer.num_bytes, 8 * 2097152)
                self.assertEqual(
                    native.ElasticBuffer.call_args.args[0], buffer.num_bytes
                )
                self.assertEqual(
                    Buffer.get_buffer_size_hint(**settings),
                    buffer.num_bytes,
                )

    def test_expert_capacity_is_required_and_positive_before_native_allocation(self):
        Buffer, cuda, _ = load_buffer()
        native = Buffer.__init__.__globals__["_C"]
        native.ElasticBuffer = Mock()
        with self.assertRaisesRegex(TypeError, "num_experts_per_rank"):
            Buffer(0, 1, 1, 1, 256, num_topk=32)
        for capacity in (0, -1, True, 1.0, None):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(
                    ValueError, "num_experts_per_rank must be a positive integer"
                ):
                    Buffer(0, 1, 1, 1, 256, num_experts_per_rank=capacity, num_topk=32)
        native.ElasticBuffer.assert_not_called()
        native.calculate_elastic_buffer_size.assert_not_called()
        self.assertEqual(cuda.synchronizations, 0)

    def test_topk_capacity_is_required_and_validated_before_native_allocation(self):
        Buffer, cuda, namespace = load_buffer()
        native = namespace["_C"]
        native.ElasticBuffer = Mock(wraps=native.ElasticBuffer)
        settings = dict(
            max_num_scaleout_ranks=1,
            max_num_scaleup_ranks=1,
            num_max_tokens_per_rank=1,
            hidden=256,
        )
        for call in (
            lambda **kw: Buffer(world_rank=0, num_experts_per_rank=4, **kw),
            Buffer.get_buffer_size_hint,
        ):
            with self.assertRaisesRegex(TypeError, "num_topk"):
                call(**settings)
            for num_topk in (0, -1, 33, True, False, 1.0, "1", None):
                with (
                    self.subTest(num_topk=num_topk),
                    self.assertRaisesRegex(
                        ValueError, "num_topk must be an integer between 1 and 32"
                    ),
                ):
                    call(num_topk=num_topk, **settings)
        native.ElasticBuffer.assert_not_called()
        native.calculate_elastic_buffer_size.assert_not_called()
        self.assertEqual(cuda.synchronizations, 0)
        for num_topk in (1, 32):
            buffer = Buffer(
                world_rank=0, num_experts_per_rank=4, num_topk=num_topk, **settings
            )
            self.assertEqual(
                buffer.num_bytes,
                Buffer.get_buffer_size_hint(num_topk=num_topk, **settings),
            )
            native.calculate_elastic_buffer_size.assert_called_with(
                1, 1, 1, 256, num_topk, False, True
            )

    def test_workload_sizes_are_required_before_sizing_or_native_allocation(self):
        Buffer, cuda, namespace = load_buffer()
        native = namespace["_C"]
        native.ElasticBuffer = Mock()
        for missing in ("num_max_tokens_per_rank", "hidden"):
            settings = dict(num_max_tokens_per_rank=1, hidden=256)
            del settings[missing]
            with (
                self.subTest(missing=missing),
                self.assertRaisesRegex(TypeError, missing),
            ):
                Buffer(0, 1, 1, num_experts_per_rank=1, num_topk=32, **settings)
        native.ElasticBuffer.assert_not_called()
        native.calculate_elastic_buffer_size.assert_not_called()
        self.assertEqual(cuda.synchronizations, 0)

    def test_invalid_rank_or_capacity_fails_before_sizing_or_native_allocation(self):
        Buffer, cuda, _ = load_buffer()
        native = Buffer.__init__.__globals__["_C"]
        native.ElasticBuffer = Mock()
        native.calculate_elastic_buffer_size = Mock()
        for world_rank, domains, positions in (
            (-1, 3, 4),
            (12, 3, 4),
            (True, 3, 4),
            (1.0, 3, 4),
            (0, 0, 4),
            (0, -1, 4),
            (0, True, 4),
            (0, 3, 0),
            (0, 3, -1),
            (0, 3, 4.0),
        ):
            with self.subTest(world_rank=world_rank, dimensions=(domains, positions)):
                with self.assertRaisesRegex(ValueError, "declared maximum world"):
                    Buffer(
                        world_rank,
                        domains,
                        positions,
                        1,
                        256,
                        num_experts_per_rank=1,
                        num_topk=32,
                    )
        native.ElasticBuffer.assert_not_called()
        native.calculate_elastic_buffer_size.assert_not_called()
        self.assertEqual(cuda.synchronizations, 0)

    def make_wrapper(
        self,
        rank,
        agent_name=None,
        physical_domains=None,
        max_domains=1,
        max_positions=8,
    ):
        Buffer, _, _ = load_buffer(physical_domains)
        namespace = Buffer.__init__.__globals__
        namespace["exchange_records"] = self.module.exchange_records
        buffer = Buffer(
            rank,
            max_domains,
            max_positions,
            1,
            256,
            num_experts_per_rank=4,
            num_topk=32,
            store=self.store,
            explicitly_destroy=True,
        )
        buffer.runtime.get_local_agent_name = lambda: (
            f"rank-{rank}" if agent_name is None else agent_name
        )
        buffer.runtime.get_local_peer_info = lambda: (4096, 0, rank, 2097152)
        buffer.runtime.connect_ranks = Mock(wraps=buffer.runtime.connect_ranks)
        return buffer

    def test_buffer_connect_exchanges_agent_names_and_region_information(self):
        first_name = "b14a8ca1b892448791ff7e4f1a24ed98_2"
        second_name = "ea59713635e0478cbe56b9f00f820791_5"
        first = self.make_wrapper(2, first_name)
        second = self.make_wrapper(5, second_name)
        parallel([lambda: first.connect_ranks([5]), lambda: second.connect_ranks([2])])
        first.runtime.connect_ranks.assert_called_once_with(
            [5], [second_name], [(4096, 0, 5, 2097152)], [-1, -1, 0, -1, -1, 1, -1, -1]
        )
        second.runtime.connect_ranks.assert_called_once_with(
            [2], [first_name], [(4096, 0, 2, 2097152)], [-1, -1, 0, -1, -1, 1, -1, -1]
        )
        self.assertEqual(self.store.num_keys(), 0)

    def test_buffer_connect_requires_store_and_accepts_per_call_store(self):
        first, second = self.make_wrapper(2), self.make_wrapper(5)
        first.store = second.store = None
        with self.assertRaisesRegex(ValueError, "Store"):
            first.connect_ranks([5])
        first.runtime.connect_ranks.assert_not_called()
        parallel(
            [
                lambda: first.connect_ranks([5], store=self.store),
                lambda: second.connect_ranks([2], store=self.store),
            ]
        )
        self.assertEqual(first.runtime.connected, {5})
        self.assertEqual(second.runtime.connected, {2})
        self.assertEqual(self.store.num_keys(), 0)

    def test_buffer_connect_existing_peers_does_not_exchange_or_restage(self):
        buffer = self.make_wrapper(2)
        buffer.runtime.connected.add(5)
        buffer.store = None
        exchange = Mock(side_effect=AssertionError("Unexpected Store exchange"))
        namespace = type(buffer).connect_ranks.__globals__
        with patch.dict(namespace, exchange_records=exchange):
            for staged in (False, True):
                with self.subTest(staged=staged):
                    buffer.runtime.staged_views = staged
                    for ranks in ([], [5], [5, 5]):
                        buffer.connect_ranks(ranks)
                    self.assertEqual(buffer.runtime.staged_views, staged)
                    self.assertEqual(buffer.get_connected_ranks(), [5])
        exchange.assert_not_called()
        buffer.runtime.connect_ranks.assert_not_called()

    def test_buffer_connect_mixed_peers_only_exchanges_with_new_peers(self):
        first, existing, joining = (
            self.make_wrapper(2),
            self.make_wrapper(5),
            self.make_wrapper(6),
        )
        parallel(
            [lambda: first.connect_ranks([5]), lambda: existing.connect_ranks([2])]
        )
        for buffer in (first, existing):
            buffer.set_active_ranks([2, 5])
            buffer.runtime.connect_ranks.reset_mock()
        parallel(
            [
                lambda: first.connect_ranks([5, 6, 5]),
                lambda: existing.connect_ranks([2, 6]),
                lambda: joining.connect_ranks([2, 5]),
            ]
        )
        mapping = [-1, -1, 0, -1, -1, 1, 2, -1]
        first.runtime.connect_ranks.assert_called_once_with(
            [6], ["rank-6"], [(4096, 0, 6, 2097152)], mapping
        )
        existing.runtime.connect_ranks.assert_called_once_with(
            [6], ["rank-6"], [(4096, 0, 6, 2097152)], mapping
        )
        joining.runtime.connect_ranks.assert_called_once_with(
            [2, 5],
            ["rank-2", "rank-5"],
            [(4096, 0, 2, 2097152), (4096, 0, 5, 2097152)],
            mapping,
        )
        self.assertEqual(first.get_connected_ranks(), [5, 6])
        self.assertEqual(joining.get_connected_ranks(), [2, 5])
        self.assertEqual(first.get_active_ranks(), [2, 5])
        self.assertEqual(existing.get_active_ranks(), [2, 5])
        self.assertEqual(joining.get_active_ranks(), [6])
        for buffer in (first, existing, joining):
            buffer.set_active_ranks([2, 5, 6])
            self.assertEqual(buffer.get_active_ranks(), [2, 5, 6])
            self.assertEqual(buffer.runtime.committed_mapping, {2: 0, 5: 1, 6: 2})
        self.assertEqual(self.store.num_keys(), 0)

    def test_retired_rank_can_rejoin_in_a_different_physical_domain(self):
        physical = {0: 7, 1: 7}

        def make(rank, name):
            return self.make_wrapper(rank, name, physical, 2, 2)

        first, departed = make(0, "survivor"), make(1, "departed")
        parallel(
            [lambda: first.connect_ranks([1]), lambda: departed.connect_ranks([0])]
        )
        for buffer in (first, departed):
            buffer.set_active_ranks([0, 1])
        first.disconnect_ranks([1])
        self.assertEqual(first.get_active_ranks(), [0])
        self.assertEqual(first._prepared_mapping, {0: 0, 1: 1})
        self.assertNotIn(1, first._fabric_records)

        physical[1] = 2
        replacement = make(1, "replacement")
        parallel(
            [lambda: first.connect_ranks([1]), lambda: replacement.connect_ranks([0])]
        )
        for buffer in (first, replacement):
            self.assertEqual(buffer.get_active_ranks(), [buffer.world_rank])
            self.assertEqual(buffer._prepared_mapping, {0: 0, 1: 2})
            buffer.set_active_ranks([0, 1])
            self.assertEqual(buffer.get_active_ranks(), [0, 1])
            self.assertEqual(buffer._get_mode(), "hybrid")
        self.assertEqual(self.store.num_keys(), 0)

    def test_inactive_connected_rank_preserves_mapping_when_a_peer_joins(self):
        buffers = [self.make_wrapper(rank, max_positions=4) for rank in range(4)]
        first, joining, inactive, last = buffers
        parallel([lambda: inactive.connect_ranks([3]), lambda: last.connect_ranks([2])])
        for buffer in (inactive, last):
            buffer.set_active_ranks([2, 3])
        parallel(
            [
                lambda: first.connect_ranks([2, 3]),
                lambda: inactive.connect_ranks([0]),
                lambda: last.connect_ranks([0]),
            ]
        )
        for buffer in (first, inactive, last):
            buffer.set_active_ranks([0, 3])
        parallel(
            [
                lambda: first.connect_ranks([1]),
                lambda: inactive.connect_ranks([1]),
                lambda: last.connect_ranks([1]),
                lambda: joining.connect_ranks([0, 2, 3]),
            ]
        )
        mapping = {0: 2, 1: 3, 2: 0, 3: 1}
        self.assertEqual(inactive.get_active_ranks(), [0, 3])
        for buffer in buffers:
            self.assertEqual(buffer._prepared_mapping, mapping)
            buffer.set_active_ranks([0, 1, 2, 3])
            self.assertEqual(buffer.runtime.committed_mapping, mapping)
        self.assertEqual(self.store.num_keys(), 0)

    def test_indirect_evidence_does_not_keep_a_retired_peer_alive(self):
        physical = {0: 7, 1: 7, 2: 7}

        def make(rank, name):
            return self.make_wrapper(rank, name, physical, 2, 3)

        first, departed, joining = make(0, "zero"), make(1, "old"), make(2, "two")
        parallel(
            [lambda: first.connect_ranks([1]), lambda: departed.connect_ranks([0])]
        )
        for buffer in (first, departed):
            buffer.set_active_ranks([0, 1])
        # Prepare rank 2 while [0, 1] still runs; rank 1 will leave, so 2 never connects to it.
        parallel([lambda: first.connect_ranks([2]), lambda: joining.connect_ranks([0])])
        self.assertEqual(first.get_active_ranks(), [0, 1])
        self.assertEqual(joining.get_connected_ranks(), [0])
        self.assertIn(1, joining._fabric_records)
        for buffer in (first, joining):
            buffer.set_active_ranks([0, 2])
        first.disconnect_ranks([1])

        physical[1] = 8
        replacement = make(1, "replacement")
        parallel(
            [
                lambda: first.connect_ranks([1]),
                lambda: joining.connect_ranks([1]),
                lambda: replacement.connect_ranks([0, 2]),
            ]
        )
        for buffer in (first, joining, replacement):
            self.assertEqual(buffer._prepared_mapping, {0: 0, 1: 3, 2: 2})
            self.assertEqual(buffer._fabric_records[1]["fabric"]["clique_id"], 8)
        self.assertEqual(first.get_active_ranks(), [0, 2])
        self.assertEqual(joining.get_active_ranks(), [0, 2])
        self.assertEqual(replacement.get_active_ranks(), [1])
        self.assertEqual(self.store.num_keys(), 0)


if __name__ == "__main__":
    unittest.main()
