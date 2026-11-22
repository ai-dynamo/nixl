# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""CPU checks of the elastic oracle's fixed expert ownership and replay inputs."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

SOURCE = Path(__file__).with_name("test_elastic_ep.py")


def load_functions(path, names, namespace):
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    nodes += [
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) in names
        or (
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) in names for target in node.targets)
        )
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )


class CpuTorch:
    """Keep the oracle's tensor operations unchanged while allocating only on the CPU."""

    def __getattr__(self, name):
        function = getattr(torch, name)
        if name not in {"tensor", "arange", "empty"}:
            return function

        def allocate(*args, **kwargs):
            kwargs["device"] = "cpu"
            return function(*args, **kwargs)

        return allocate


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class StableExpertIdsTests(unittest.TestCase):
    def setUp(self):
        namespace = dict(torch=CpuTorch(), math=math)
        load_functions(
            SOURCE.parents[2] / "python/utils/refs.py",
            {"generate_pre_combine_data", "ordered_accumulate", "combine"},
            namespace,
        )
        namespace["ref_combine"] = namespace["combine"]
        load_functions(SOURCE, {"combine_recipe"}, namespace)
        load_functions(
            SOURCE.with_name("buffer_setup.py"),
            {"rank_mapping", "rank_coordinates", "topology_experts"},
            namespace,
        )
        load_functions(
            SOURCE,
            {
                "extent_of",
                "Case",
                "GraphCase",
                "NUM_CAPTURE_BARRIERS",
                "COMBINE_SLICE_BYTES",
                "as_nvfp4",
                "payload_parts",
                "payload_rows",
                "cat_payloads",
                "payload_equal",
            },
            namespace,
        )
        self.namespace = namespace
        self.mapping = list(range(8))
        self.buffer = SimpleNamespace(
            max_num_scaleout_ranks=2,
            max_num_scaleup_ranks=4,
            num_experts_per_rank=2,
            world_rank=4,
            runtime=SimpleNamespace(get_rank_mapping=lambda: self.mapping),
            allow_multiple_reduction=True,
            _get_mode=lambda: "hybrid",
        )
        self.case = namespace["Case"](
            self.buffer,
            SimpleNamespace(
                num_tokens=16, hidden=8, num_topk=4, num_experts=16, seed=1
            ),
        )

    def test_replay_restores_captured_routes_and_poisons_output(self):
        graph_type = self.namespace["GraphCase"]
        graph = graph_type.__new__(graph_type)
        capture_phase, phase = [0, 4], [0, 2, 4]
        reroute_seed = graph_type.REROUTE_ROUTING_PHASE + 1
        captured = (torch.tensor([[8, 0]]), torch.tensor([[0.25, 0.75]]))
        rerouted = (torch.tensor([[10, 4]]), torch.tensor([[0.8, 0.2]]))
        routes = {
            (graph_type.CAPTURE_ROUTING_PHASE, tuple(capture_phase)): captured,
            (reroute_seed, tuple(phase)): rerouted,
        }
        graph.case = SimpleNamespace(
            buffer=self.buffer,
            routing=lambda seed, rank, world: routes[seed, tuple(world)],
        )
        graph.capture_phase_of = {4: capture_phase}
        graph.topk_idx, graph.topk_weights = (torch.zeros_like(t) for t in captured)
        addresses = (graph.topk_idx.data_ptr(), graph.topk_weights.data_ptr())
        graph.outputs, graph.stats = torch.ones((1, 8)), torch.ones(2)
        seen = []

        def replay():
            self.assertTrue(torch.isnan(graph.outputs).all().item())
            self.assertEqual(graph.stats.tolist(), [0, 0])
            seen.append((graph.topk_idx.clone(), graph.topk_weights.clone()))
            graph.outputs.fill_(3)
            graph.stats.fill_(4)

        graph.graph = SimpleNamespace(replay=replay)
        graph.check = lambda phase, tag, routing_phase_idx=None: routing_phase_idx
        self.namespace["torch"].cuda = SimpleNamespace(synchronize=lambda: None)
        for seed in (None, reroute_seed, None):
            self.assertEqual(graph.replay_and_check(phase, "host", seed), seed)
            self.assertEqual(
                (graph.topk_idx.data_ptr(), graph.topk_weights.data_ptr()), addresses
            )
        self.assertEqual(len(seen), 3)
        for actual, expected in zip(seen, (captured, rerouted, captured)):
            for received, supplied in zip(actual, expected):
                self.assertTrue(torch.equal(received, supplied))

    def test_payload_rows_preserve_bytes_and_detect_corruption(self):
        rows, parts, equal, cat = (
            self.namespace[name]
            for name in (
                "payload_rows",
                "payload_parts",
                "payload_equal",
                "cat_payloads",
            )
        )
        bf16 = torch.arange(256).reshape(4, 64).to(torch.bfloat16)
        fp8 = (
            torch.arange(256, dtype=torch.uint8)
            .reshape(4, 64)
            .view(torch.float8_e4m3fn),
            torch.arange(1, 5, dtype=torch.float32).reshape(4, 1),
        )
        nvfp4 = self.namespace["as_nvfp4"](bf16)
        nvfp4[1].view(torch.uint8)[0, 0] = 0x7F  # NaN scale bits must survive exactly.
        for kind, payload in (("bf16", bf16), ("fp8", fp8), ("nvfp4", nvfp4)):
            with self.subTest(dtype=kind):
                expected = rows(payload, torch.tensor([3, 0, 2]))
                joined = cat(
                    [
                        rows(payload, slice(0, 0)),
                        rows(payload, torch.tensor([3])),
                        rows(payload, torch.tensor([0, 2])),
                    ]
                )
                self.assertTrue(equal(joined, expected))
                for component in range(len(parts(joined))):
                    damaged = tuple(t.clone() for t in parts(joined))
                    damaged[component].view(torch.uint8)[0, 0] ^= 1
                    damaged = damaged if isinstance(joined, tuple) else damaged[0]
                    self.assertFalse(equal(damaged, expected))

    def test_sliced_combine_matches_single_slice(self):
        self.mapping[:] = [0, 4, 1, 5, 2, 6, 3, 7]
        sources = torch.arange(17)
        sources[-2:] = -1
        local_indices = torch.tensor([[0, -1, 1, -1], [-1] * 4, [0, 1, -1, -1]])
        local_indices = local_indices.repeat(6, 1)[:17]
        local_indices[-2:] = -1
        expected_input = self.case.combine_input(sources, local_indices)
        small_budget = 5 * self.case.num_topk * self.case.hidden * 4
        self.namespace["COMBINE_SLICE_BYTES"] = small_budget
        self.assertEqual(len(self.case.row_slices(17)), 4)
        self.assertTrue(
            torch.equal(self.case.combine_input(sources, local_indices), expected_input)
        )
        empty = self.case.combine_input(
            torch.empty(0, dtype=torch.int64), torch.empty((0, 4), dtype=torch.int64)
        )
        self.assertEqual(empty.shape, (0, self.case.hidden))
        self.assertEqual(empty.dtype, torch.bfloat16)

        routing = torch.tensor([[8, 2, 12, -1], [14, 10, 4, 1], [-1] * 4]).repeat(4, 1)
        for mode, phase in (("direct", [0, 2, 4, 6]), ("hybrid", [0, 1, 4, 5])):
            with self.subTest(mode=mode):
                self.buffer._get_mode = lambda: mode
                self.namespace["COMBINE_SLICE_BYTES"] = 1 << 30
                expected = self.case.combine_reference(phase, routing)
                self.namespace["COMBINE_SLICE_BYTES"] = small_budget
                self.assertTrue(
                    torch.equal(self.case.combine_reference(phase, routing), expected)
                )

    def test_position_shrink_and_nonzero_domain_preserve_expert_owner(self):
        routing = torch.tensor([[8, 9, -1, 4], [4, 5, -1, -1], [10, 11, 0, 1]])
        original = routing.clone()
        for phase in (list(range(8)), [0, 1, 4, 5], [4, 5]):
            with self.subTest(phase=phase):
                active = self.case.active_expert_mask(phase)
                self.assertEqual(
                    active.nonzero().flatten().tolist(),
                    [2 * rank + local for rank in phase for local in range(2)],
                )
                self.assertEqual(
                    self.case.reaches_me(routing, phase).tolist(), [True, False, False]
                )
        self.assertFalse(self.case.reaches_me(routing, [0, 1]).any().item())
        self.assertEqual(self.case.flat_idx(4, [0, 1, 4, 5]), 2)
        self.assertEqual(
            self.case.global_idx(4, [0, 1, 4, 5]).tolist(), list(range(32, 44))
        )
        self.assertTrue(torch.equal(routing, original))

    def test_receive_oracle_does_not_alias_departed_owner_to_live_rank(self):
        phase = [0, 1, 4, 5]
        # External owner 2 is absent; its IDs 4/5 must not alias live world owner 4 (internal rank 2).
        rows = torch.tensor([[4, 5, -1, -1], [8, 9, 10, -1]])
        self.case.routing = lambda phase_idx, rank, active: (
            rows.repeat((self.case.num_tokens_of(rank) + 1) // 2, 1)[
                : self.case.num_tokens_of(rank)
            ],
            torch.ones((self.case.num_tokens_of(rank), 4)),
        )
        source, payload, local, weights = self.case.expected_receive(0, phase)
        expected_source = [
            self.case.flat_idx(rank, phase) * 16 + token
            for rank in phase
            for token in range(1, self.case.num_tokens_of(rank), 2)
        ]
        self.assertEqual(source.tolist(), expected_source)
        self.assertEqual(local.tolist(), [[0, 1, -1, -1]] * len(expected_source))
        payload_ids = torch.tensor(
            [
                rank * 16 + token
                for rank in phase
                for token in range(1, self.case.num_tokens_of(rank), 2)
            ]
        )
        self.assertTrue(torch.equal(payload, self.case.payload(payload_ids)))
        self.assertEqual(weights.tolist(), [[1, 1, 1, 1]] * len(expected_source))

    def test_permuted_owners_use_discovered_coordinates(self):
        self.mapping[:] = [0, 4, 1, 5, 2, 6, 3, 7]
        phase = [0, 2, 4, 6]
        self.assertEqual(self.case.extent(phase), (1, 4))
        self.assertEqual(self.case.flat_idx(4, phase), 2)
        self.assertEqual(self.case.payload_indices(4).tolist(), list(range(64, 76)))
        routing = torch.tensor([[8, 12, 0, -1]]).repeat(12, 1)
        values = (
            torch.tensor([256, 1, -256, 0], dtype=torch.bfloat16)
            .view(1, 4, 1)
            .repeat(12, 1, 8)
        )
        self.case.pre_combine_data = lambda source: values.clone()
        self.assertTrue(
            torch.equal(
                self.case.combine_reference(phase, routing),
                torch.ones((12, 8), dtype=torch.bfloat16),
            )
        )

    def test_combine_filters_inactive_ids_and_uses_physical_domains(self):
        # Owners 4 and 6 share physical domain 1 but straddle an extent stride of three.
        phase = [0, 1, 2, 4, 5, 6]
        routing = torch.tensor([[8, 12, 0, 6, -1]]).repeat(12, 1)
        values = (
            torch.tensor([256, 1, -256, 64, 128], dtype=torch.bfloat16)
            .view(1, 5, 1)
            .repeat(12, 1, 8)
        )
        self.case.pre_combine_data = lambda source: values.clone()
        expected_hybrid = torch.zeros((12, 8), dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(self.case.combine_reference(phase, routing), expected_hybrid)
        )
        self.buffer._get_mode = lambda: "direct"
        expected_direct = torch.full((12, 8), 256, dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(
                self.case.combine_reference([4, 5, 6], routing), expected_direct
            )
        )
        self.assertEqual(routing[0].tolist(), [8, 12, 0, 6, -1])


if __name__ == "__main__":
    unittest.main()
