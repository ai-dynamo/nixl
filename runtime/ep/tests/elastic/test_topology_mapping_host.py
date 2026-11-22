# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical-domain mapping tests without GPUs, NVML or the native extension."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def fabric_records(groups, *, fabric=True):
    records = {}
    for clique, ranks in enumerate(groups):
        for rank in ranks:
            records[rank] = {
                "gpu_uuid": f"gpu-{rank}",
                "host_id": "host",
                "nvlink_peers": [f"gpu-{other}" for other in ranks if other != rank],
                "fabric": (
                    {
                        "cluster_uuid": "cluster",
                        "clique_id": clique,
                        "state": 3,
                        "status": 0,
                        "handle_supported": True,
                    }
                    if fabric
                    else None
                ),
            }
    return records


class TopologyMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[2] / "python/utils"
        modules = {}
        for name in ("nixl_fabric", "nixl_topology"):
            qualified = "topology_mapping_host." + name
            spec = importlib.util.spec_from_file_location(
                qualified, source / f"{name}.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, modules):
                spec.loader.exec_module(module)
            modules[qualified] = module
        cls.fabric = modules["topology_mapping_host.nixl_fabric"]
        cls.build = staticmethod(
            modules["topology_mapping_host.nixl_topology"].build_rank_mapping
        )

    def test_permuted_world_ids_preserve_physical_domains(self):
        records = fabric_records(((0, 2), (1, 3)))
        mapping = self.build(records, [], 2, 2)
        self.assertEqual(mapping, {0: 0, 1: 2, 2: 1, 3: 3})
        self.assertEqual({mapping[rank] for rank in (0, 2)}, {0, 1})
        self.assertEqual({mapping[rank] for rank in (1, 3)}, {2, 3})
        # Public expert ownership remains rank-based even when internal slots differ.
        self.assertEqual(mapping[(2 * 16 + 7) // 16], 1)

    def test_singleton_uses_internal_zero_for_nonzero_world_id(self):
        self.assertEqual(self.build(fabric_records(((17,),)), [], 4, 8), {17: 0})

    def test_lower_numbered_joiners_do_not_move_established_slots(self):
        records = fabric_records(((0, 2, 4), (1, 3)))
        snapshots = [{2: 0, 4: 1, 3: 3}, {4: 1, 3: 3}]
        before = copy.deepcopy((records, snapshots))
        expected = {0: 2, 1: 4, 2: 0, 3: 3, 4: 1}
        self.assertEqual(self.build(records, snapshots, 2, 3), expected)
        self.assertEqual(
            self.build(dict(reversed(list(records.items()))), snapshots[::-1], 2, 3),
            expected,
        )
        self.assertEqual((records, snapshots), before)

    def test_new_domain_does_not_take_an_established_domain_index(self):
        records = fabric_records(((0, 1), (4, 5)))
        self.assertEqual(
            self.build(records, [{4: 0, 5: 1}], 3, 2),
            {0: 2, 1: 3, 4: 0, 5: 1},
        )

    def test_retired_claims_do_not_reserve_slots_or_move_survivors(self):
        records = fabric_records(((0, 2), (3,)))
        mapping = self.build(records, [{0: 0, 1: 1, 3: 2}], 2, 2)
        self.assertEqual(mapping, {0: 0, 2: 1, 3: 2})
        self.assertNotIn(1, mapping)

    def test_conflicting_established_assignments_are_refused(self):
        cases = (
            (((0, 1),), [{0: 0}, {0: 1}], "one world rank"),
            (((0, 1),), [{0: 0}, {1: 0}], "multiple ranks"),
            (((0, 1),), [{0: 0, 1: 2}], "splits"),
            (((0,), (1,)), [{0: 0, 1: 1}], "combines"),
        )
        for groups, snapshots, message in cases:
            with self.subTest(groups=groups, snapshots=snapshots):
                records = fabric_records(groups)
                before = copy.deepcopy((records, snapshots))
                with self.assertRaisesRegex(ValueError, message):
                    self.build(records, snapshots, 2, 2)
                self.assertEqual((records, snapshots), before)

    def test_topology_capacity_is_checked_before_assignment(self):
        for groups in (((0, 1, 2),), ((0,), (1,), (2,))):
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(ValueError, "topology capacity"):
                    self.build(fabric_records(groups), [], 2, 2)
        with self.assertRaisesRegex(ValueError, "maximum world"):
            self.build(fabric_records(((4,),)), [], 2, 2)
        with self.assertRaisesRegex(ValueError, "slot exceeds"):
            self.build(fabric_records(((0,),)), [{0: 4}], 2, 2)

    def test_domain_positions_above_64_remain_distinct(self):
        records = fabric_records((tuple(range(72)),))
        mapping = self.build(records, [{0: 70}], 1, 128)
        self.assertEqual(mapping[0], 70)
        self.assertEqual(set(mapping.values()), set(range(72)))

    def test_pairwise_nvlink_fallback_accepts_permuted_world_ids(self):
        records = fabric_records(((0, 2), (1, 3)), fabric=False)
        self.assertEqual(self.build(records, [], 2, 2), {0: 0, 1: 2, 2: 1, 3: 3})
        records[2]["nvlink_peers"] = []
        with self.assertRaisesRegex(ValueError, "Asymmetric"):
            self.build(records, [], 2, 2)

    def test_incomplete_cliques_and_fabric_registration_are_refused(self):
        records = fabric_records(((0, 1, 2),), fabric=False)
        records[0]["nvlink_peers"].remove("gpu-2")
        records[2]["nvlink_peers"].remove("gpu-0")
        with self.assertRaisesRegex(ValueError, "complete pairwise"):
            self.build(records, [], 1, 4)
        records = fabric_records(((0, 1),))
        records[1]["fabric"]["state"] = 2
        with self.assertRaisesRegex(ValueError, "registration incomplete"):
            self.build(records, [], 1, 4)

    def test_group_test_discovery_keeps_its_declared_domain_contract(self):
        ordered = list(fabric_records(((0, 1), (2, 3))).values())
        self.assertEqual(self.fabric.validate_fabric_records(ordered), (2, 2))
        self.assertEqual(self.fabric.validate_fabric_records(ordered, 2), (2, 2))
        permuted = fabric_records(((0, 2), (1, 3)))
        self.assertEqual(self.fabric.group_fabric_records(permuted), [[0, 2], [1, 3]])
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            self.fabric.validate_fabric_records([permuted[i] for i in range(4)])
        with self.assertRaisesRegex(ValueError, "positive and divide"):
            self.fabric.validate_fabric_records(ordered, 0)


if __name__ == "__main__":
    unittest.main()
