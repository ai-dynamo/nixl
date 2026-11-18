# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""CPU checks of the elastic oracle's fixed expert ownership and extent-ranked sources."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

SOURCE = Path(__file__).with_name("test_adaptive_elastic.py")


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
        if name not in {"tensor", "arange"}:
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
            SOURCE.parents[2] / "nixl_ep_v2/utils/refs.py",
            {"generate_pre_combine_data", "ordered_accumulate", "combine"},
            namespace,
        )
        namespace["ref_combine"] = namespace["combine"]
        load_functions(
            SOURCE.with_name("test_adaptive_modes.py"), {"combine_recipe"}, namespace
        )
        load_functions(SOURCE, {"extent_of", "Case"}, namespace)
        self.buffer = SimpleNamespace(
            max_num_scaleout_ranks=2,
            max_num_scaleup_ranks=4,
            num_experts_per_rank=2,
            world_rank=4,
            num_launch_sms=4,
            allow_multiple_reduction=True,
            _get_topology=lambda: ("hybrid", 4),
        )
        self.case = namespace["Case"](
            self.buffer,
            SimpleNamespace(
                num_tokens=16, hidden=8, num_topk=4, num_experts=16, num_sms=4, seed=1
            ),
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
        self.assertTrue(torch.equal(payload, self.case.payload(source)))
        self.assertEqual(weights.tolist(), [[1, 1, 1, 1]] * len(expected_source))

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
        self.buffer._get_topology = lambda: ("direct", 4)
        expected_direct = torch.full((12, 8), 256, dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(
                self.case.combine_reference([4, 5, 6], routing), expected_direct
            )
        )
        self.assertEqual(routing[0].tolist(), [8, 12, 0, 6, -1])


if __name__ == "__main__":
    unittest.main()
