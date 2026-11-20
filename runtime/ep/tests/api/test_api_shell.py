# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only API contracts; tensor placeholders never establish numerical correctness.

Run with ``python -m unittest discover -s runtime/ep/tests/api -v``. Set
``NIXL_EP_API_MODULE=nixl.runtime.ep`` to exercise the installed public package.
The checked-in signature snapshot comes from production commit 6b11b2c, with
documented API corrections; it is independent of the shell under test.
"""

import ast
import importlib
import importlib.abc
import inspect
import json
import os
import socket
import subprocess
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

EP_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path(__file__).with_name("production_api_6b11b2c.json")


class ForbiddenNativeImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            "nixl_ep_v2_cpp" in fullname
            or fullname.endswith("._C")
            or fullname in {"pynvml", "nvml", "nixl._api"}
        ):
            raise AssertionError(f"API shell attempted native import: {fullname}")
        return None


@contextmanager
def no_execution():
    """Fail before crossing a hardware, transport, process or network boundary."""
    finder = ForbiddenNativeImport()
    sys.meta_path.insert(0, finder)
    try:
        with ExitStack() as stack:
            groups = (
                (
                    torch.cuda,
                    "_lazy_init init set_device current_device synchronize "
                    "current_stream default_stream stream Stream Event device_count "
                    "is_available get_device_properties is_current_stream_capturing",
                ),
                (
                    dist,
                    "init_process_group new_group barrier broadcast broadcast_object_list "
                    "all_gather all_gather_object all_reduce all_to_all_single "
                    "TCPStore FileStore HashStore",
                ),
                (subprocess, "Popen run call check_call check_output"),
                (socket, "socket create_connection"),
                (os, "system"),
            )
            for module, names in groups:
                for name in names.split():
                    stack.enter_context(
                        patch.object(
                            module,
                            name,
                            side_effect=AssertionError(
                                f"API shell called {module.__name__}.{name}"
                            ),
                        )
                    )
            yield
    finally:
        sys.meta_path.remove(finder)


class ForbiddenStore:
    def __getattr__(self, name):
        raise AssertionError(f"API shell accessed Store.{name}")


def source_contract(path, classes=(), functions=()):
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            members = (
                (f"{node.name}.{member.name}", member)
                for member in node.body
                if isinstance(member, ast.FunctionDef)
                and (not member.name.startswith("_") or member.name.startswith("__"))
            )
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            members = [(node.name, node)]
        else:
            continue
        for name, member in members:
            result[name] = {
                "parameters": ast.unparse(member.args),
                "returns": ast.unparse(member.returns) if member.returns else None,
                "decorators": [ast.unparse(item) for item in member.decorator_list],
            }
    return result


class ApiShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_name = os.environ.get("NIXL_EP_API_MODULE", "python")
        if cls.module_name == "python":
            sys.path.insert(0, str(EP_ROOT))
            cls.addClassCleanup(sys.path.remove, str(EP_ROOT))
        with no_execution():
            cls.ep = importlib.import_module(cls.module_name)
        cls.package_root = Path(cls.ep.__file__).parent

    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(no_execution())

    def buffer(self, **kwargs):
        options = dict(
            world_rank=0,
            max_num_scaleout_ranks=2,
            max_num_scaleup_ranks=2,
            num_max_tokens_per_rank=8,
            hidden=256,
            num_topk=2,
            num_experts_per_rank=3,
            explicitly_destroy=True,
            store=ForbiddenStore(),
        )
        options.update(kwargs)
        return self.ep.ElasticBuffer(**options)

    def two_rank_buffer(self, **kwargs):
        buffer = self.buffer(**kwargs)
        buffer.connect_ranks([1], store=ForbiddenStore())
        buffer.set_active_ranks([0, 1])
        return buffer

    def tensor(self, shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    def assert_placeholder(self, value, shape, dtype):
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(value.device.type, "meta")
        self.assertEqual(tuple(value.shape), shape)
        self.assertEqual(value.dtype, dtype)

    def test_production_signatures_defaults_annotations_and_descriptors(self):
        expected = json.loads(SNAPSHOT.read_text())
        actual = {}
        for filename, classes, functions in (
            ("__init__.py", (), ("init_jit",)),
            ("buffers/elastic.py", ("ElasticBuffer", "EPHandle"), ()),
            ("utils/event.py", ("EventOverlap", "EventHandle"), ()),
        ):
            actual.update(
                source_contract(self.package_root / filename, classes, functions)
            )
        self.assertEqual(set(actual), set(expected["callables"]))
        for name, contract in expected["callables"].items():
            with self.subTest(name=name):
                self.assertEqual(actual[name], contract)
                if "." in name:
                    class_name, method_name = name.split(".")
                    cls = getattr(self.ep, class_name)
                    descriptor = inspect.getattr_static(cls, method_name)
                    if contract["decorators"] == ["staticmethod"]:
                        self.assertIsInstance(descriptor, staticmethod)
                    elif contract["decorators"] == ["property"]:
                        self.assertIsInstance(descriptor, property)
                    else:
                        self.assertTrue(inspect.isfunction(descriptor))

    def test_root_exports_helpers_and_submodule_identity(self):
        expected = json.loads(SNAPSHOT.read_text())
        self.assertEqual(self.ep.__all__, expected["exports"])
        namespace = {}
        exec(f"from {self.module_name} import *", namespace)
        self.assertEqual(set(namespace) - {"__builtins__"}, set(expected["exports"]))
        self.assertNotIn("init_jit", namespace)
        self.assertIsNone(self.ep.init_jit())
        self.assertIs(self.ep.topk_idx_t, torch.int64)
        elastic = importlib.import_module(f"{self.module_name}.buffers.elastic")
        event = importlib.import_module(f"{self.module_name}.utils.event")
        self.assertIs(self.ep.ElasticBuffer, elastic.ElasticBuffer)
        self.assertIs(self.ep.EPHandle, elastic.EPHandle)
        self.assertIs(self.ep.EventHandle, event.EventHandle)
        self.assertIs(self.ep.EventOverlap, event.EventOverlap)

    def test_configuration_and_local_helpers_do_not_touch_store_or_hardware(self):
        store = ForbiddenStore()
        buffer = self.buffer(world_rank=1, store=store)
        self.assertEqual(buffer.num_experts_per_rank, 3)
        self.assertEqual(buffer.num_bytes, 0)
        self.assertIs(buffer.store, store)
        self.assertIsNone(buffer.runtime)
        self.assertEqual(buffer.get_active_ranks(), [1])
        self.assertEqual(buffer.get_buffer_size_hint(2, 2, 8, 256, 2), 0)
        self.assertEqual(buffer.read_profile(), {})
        self.assertIs(buffer.get_comm_stream(), buffer.get_comm_stream())
        self.assertIsNone(buffer.barrier(with_cpu_sync=True))

    def test_membership_accepts_caller_rank_ids_without_inferring_topology(self):
        buffer = self.two_rank_buffer()
        buffer.connect_ranks([2, 3], store=ForbiddenStore())
        self.assertEqual(buffer.get_connected_ranks(), [1, 2, 3])
        self.assertEqual(buffer.get_active_ranks(), [0, 1])
        # These IDs do not imply physical domains or internal positions.
        for active in ([3, 2, 1, 0], [0, 3], [0, 1, 3], [1, 2], [2, 3]):
            with self.subTest(active=active):
                buffer.set_active_ranks(active)
                self.assertEqual(buffer.get_active_ranks(), sorted(active))
                self.assertEqual(buffer.num_experts_per_rank, 3)
        self.assertNotIn(buffer.world_rank, buffer.get_active_ranks())
        buffer.disconnect_ranks([1])
        self.assertEqual(buffer.get_connected_ranks(), [2, 3])
        self.assertEqual(buffer.get_active_ranks(), [2, 3])

    def test_inactive_rank_can_retire_peers_without_changing_membership(self):
        for active in ([2, 3], [1], [1, 3]):
            with self.subTest(active=active):
                buffer = self.two_rank_buffer()
                buffer.connect_ranks([2, 3], store=ForbiddenStore())
                buffer.set_active_ranks(active)
                buffer.disconnect_ranks([1, 2, 3])
                self.assertEqual(buffer.get_connected_ranks(), [])
                self.assertEqual(buffer.get_active_ranks(), active)
                self.assertNotIn(buffer.world_rank, buffer.get_active_ranks())

    def test_disconnect_mixed_active_and_inactive_peers_leaves_singleton(self):
        for active, retired in (([0, 1], [2, 1]), ([0, 2], [1, 2])):
            with self.subTest(active=active):
                buffer = self.two_rank_buffer()
                buffer.connect_ranks([2, 3], store=ForbiddenStore())
                buffer.set_active_ranks(active)
                buffer.disconnect_ranks(retired)
                self.assertEqual(buffer.get_connected_ranks(), [3])
                self.assertEqual(buffer.get_active_ranks(), [0])
                received, _, _, handle, _ = buffer.dispatch(
                    self.tensor((5, 256)), self.tensor((5, 2), self.ep.topk_idx_t)
                )
                combined, _ = buffer.combine(received, handle)
                self.assert_placeholder(combined, (5, 256), torch.bfloat16)

    def test_disconnect_rejects_entire_request_before_changing_state(self):
        for retired in ([4, 4], [4, 0], [4, 5], [4, 6], [4, -1]):
            with self.subTest(retired=retired):
                buffer = self.buffer(max_num_scaleout_ranks=3, num_launch_sms=8)
                buffer.connect_ranks([1, 2, 3, 4])
                buffer.set_active_ranks([0, 1, 2, 3])
                with self.assertRaises(ValueError):
                    buffer.disconnect_ranks(retired)
                self.assertEqual(buffer.get_connected_ranks(), [1, 2, 3, 4])
                self.assertEqual(buffer.get_active_ranks(), [0, 1, 2, 3])
                self.assertEqual(buffer.num_launch_sms, 8)

    def test_disconnect_keeps_noncontiguous_rank_ids(self):
        buffer = self.two_rank_buffer()
        buffer.connect_ranks([2, 3])
        buffer.set_active_ranks([0, 1, 2, 3])
        buffer.disconnect_ranks([1, 3])
        self.assertEqual(buffer.get_connected_ranks(), [2])
        self.assertEqual(buffer.get_active_ranks(), [0, 2])

    def test_disconnect_does_not_infer_physical_shape_from_rank_ids(self):
        buffer = self.buffer(max_num_scaleout_ranks=3)
        buffer.connect_ranks([1, 2, 3, 4])
        buffer.set_active_ranks([0, 1, 2, 3])
        buffer.disconnect_ranks([4, 2])
        self.assertEqual(buffer.get_active_ranks(), [0, 1, 3])
        self.assertEqual(buffer.get_connected_ranks(), [1, 3])

    def test_disconnect_preserves_nonzero_caller_rank(self):
        buffer = self.buffer(world_rank=2)
        buffer.connect_ranks([0, 1, 3])
        buffer.set_active_ranks([0, 1, 2, 3])
        buffer.disconnect_ranks([0, 1])
        self.assertEqual(buffer.get_connected_ranks(), [3])
        self.assertEqual(buffer.get_active_ranks(), [2, 3])
        buffer.disconnect_ranks([3])
        self.assertEqual(buffer.get_connected_ranks(), [])
        self.assertEqual(buffer.get_active_ranks(), [2])

    def test_connect_rejects_duplicate_new_ranks_before_mutation(self):
        for ranks in ([2, 2], [1, 2, 2]):
            with self.subTest(ranks=ranks):
                buffer = self.two_rank_buffer()
                with self.assertRaises(ValueError):
                    buffer.connect_ranks(ranks, store=ForbiddenStore())
                self.assertEqual(buffer.get_connected_ranks(), [1])
                self.assertEqual(buffer.get_active_ranks(), [0, 1])

    def test_connect_ignores_existing_peers_without_store_or_state_changes(self):
        for store in (None, ForbiddenStore()):
            for active in ([0, 1], [0]):
                with self.subTest(store=store, active=active):
                    buffer = self.two_rank_buffer(store=store)
                    buffer.set_active_ranks(active)
                    buffer.connect_ranks([1, 1])
                    buffer.connect_ranks([])
                    self.assertEqual(buffer.get_connected_ranks(), [1])
                    self.assertEqual(buffer.get_active_ranks(), active)

    def test_connect_mixed_peers_prepares_only_new_connections(self):
        buffer = self.two_rank_buffer()
        buffer.connect_ranks([2, 1, 1])
        self.assertEqual(buffer.get_connected_ranks(), [1, 2])
        self.assertEqual(buffer.get_active_ranks(), [0, 1])
        buffer.connect_ranks([2, 1])
        self.assertEqual(buffer.get_connected_ranks(), [1, 2])
        self.assertEqual(buffer.get_active_ranks(), [0, 1])
        buffer.set_active_ranks([0, 2])
        self.assertEqual(buffer.get_active_ranks(), [0, 2])

    def test_connect_requires_store_without_changing_active_state(self):
        for active in ([0, 1], [0]):
            with self.subTest(active=active):
                buffer = self.two_rank_buffer(store=None)
                buffer.set_active_ranks(active)
                with self.assertRaisesRegex(ValueError, "explicit Store"):
                    buffer.connect_ranks([2, 3])
                self.assertEqual(buffer.get_connected_ranks(), [1])
                self.assertEqual(buffer.get_active_ranks(), active)
                buffer.connect_ranks([2, 3], store=ForbiddenStore())
                self.assertEqual(buffer.get_connected_ranks(), [1, 2, 3])
                self.assertEqual(buffer.get_active_ranks(), active)

        buffer = self.two_rank_buffer()
        buffer.connect_ranks([2, 3])
        self.assertEqual(buffer.get_connected_ranks(), [1, 2, 3])
        self.assertEqual(buffer.get_active_ranks(), [0, 1])

    def test_disconnect_finishes_retirement_and_allows_reconnect(self):
        buffer = self.two_rank_buffer()
        buffer.connect_ranks([2], store=ForbiddenStore())
        buffer.set_active_ranks([0, 2])
        buffer.set_active_ranks([0, 1])
        for ranks in ([2, 3], [2, 2]):
            with self.subTest(disconnect=ranks):
                with self.assertRaises(ValueError):
                    buffer.disconnect_ranks(ranks)
                self.assertEqual(buffer.get_connected_ranks(), [1, 2])
        buffer.disconnect_ranks([2])
        self.assertEqual(buffer.get_connected_ranks(), [1])
        self.assertEqual(buffer.get_active_ranks(), [0, 1])
        buffer.connect_ranks([2], store=ForbiddenStore())
        buffer.set_active_ranks([0, 2])
        self.assertEqual(buffer.get_connected_ranks(), [1, 2])
        self.assertEqual(buffer.get_active_ranks(), [0, 2])

    def test_disconnect_and_reconnect_preserve_fixed_sm_count(self):
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        for sms in (2, 24, 64):
            with self.subTest(sms=sms):
                buffer = self.two_rank_buffer(num_launch_sms=sms)
                buffer.connect_ranks([2, 3])
                buffer.set_active_ranks([0, 1, 2, 3])
                received, _, _, handle, _ = buffer.dispatch(x, idx)
                buffer.combine(received, handle)
                buffer.disconnect_ranks([2, 3])
                self.assertEqual(buffer.get_connected_ranks(), [1])
                self.assertEqual(buffer.get_active_ranks(), [0, 1])
                self.assertEqual(buffer.num_launch_sms, sms)
                buffer.connect_ranks([2, 3])
                buffer.set_active_ranks([0, 1, 2, 3])
                self.assertEqual(buffer.num_launch_sms, sms)

    def test_empty_disconnect_preserves_connections_membership_and_fixed_sms(self):
        buffer = self.two_rank_buffer(num_launch_sms=8)
        received, _, _, handle, _ = buffer.dispatch(
            self.tensor((5, 256)), self.tensor((5, 2), self.ep.topk_idx_t)
        )
        buffer.connect_ranks([2, 3])
        buffer.disconnect_ranks([])
        self.assertEqual(buffer.get_connected_ranks(), [1, 2, 3])
        self.assertEqual(buffer.get_active_ranks(), [0, 1])
        self.assertEqual(buffer.num_launch_sms, 8)
        combined, _ = buffer.combine(received, handle)
        self.assert_placeholder(combined, (5, 256), torch.bfloat16)

    def test_constructor_initializes_only_local_rank_without_store_access(self):
        for world_rank in (0, 2, 3):
            for store in (None, ForbiddenStore()):
                with self.subTest(world_rank=world_rank, store=store):
                    buffer = self.buffer(world_rank=world_rank, store=store)
                    self.assertEqual(buffer.world_rank, world_rank)
                    self.assertEqual(buffer.get_active_ranks(), [world_rank])
                    self.assertEqual(buffer.get_connected_ranks(), [])
                    self.assertIs(buffer.store, store)
                    self.assertIsNone(buffer.runtime)
                    self.assertEqual(buffer.num_bytes, 0)
                    for name in (
                        "rank_idx",
                        "num_ranks",
                        "scaleout_rank_idx",
                        "scaleup_rank_idx",
                        "_get_mode",
                        "_get_topology_masks",
                        "_get_topology_extent",
                        "_world_rank_of",
                        "_masks_of",
                        "_mode_of",
                    ):
                        self.assertFalse(hasattr(buffer, name), name)

    def test_constructor_rejects_invalid_world_rank_and_maxima(self):
        invalid = (
            {"world_rank": -1},
            {"world_rank": 4},
            {"world_rank": True},
            {"world_rank": 1.0},
            {"max_num_scaleout_ranks": 0},
            {"max_num_scaleout_ranks": -1},
            {"max_num_scaleout_ranks": True},
            {"max_num_scaleout_ranks": 2.0},
            {"max_num_scaleup_ranks": 0},
            {"max_num_scaleup_ranks": -1},
            {"max_num_scaleup_ranks": True},
            {"max_num_scaleup_ranks": 2.0},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.buffer(**options)

    def test_constructor_requires_positive_integer_expert_capacity(self):
        with self.assertRaisesRegex(TypeError, "num_experts_per_rank"):
            self.ep.ElasticBuffer(
                world_rank=0,
                max_num_scaleout_ranks=1,
                max_num_scaleup_ranks=1,
                num_max_tokens_per_rank=1,
                hidden=256,
                num_topk=1,
            )
        for capacity in (0, -1, 1.0, "1", None, True, False):
            with (
                self.subTest(capacity=capacity),
                self.assertRaisesRegex(
                    ValueError, "num_experts_per_rank must be a positive integer"
                ),
            ):
                self.buffer(num_experts_per_rank=capacity)

    def test_constructor_requires_positive_workload_sizes(self):
        for name in ("num_max_tokens_per_rank", "hidden"):
            settings = dict(num_max_tokens_per_rank=1, hidden=256, num_topk=1)
            del settings[name]
            with self.subTest(missing=name), self.assertRaisesRegex(TypeError, name):
                self.ep.ElasticBuffer(0, 1, 1, num_experts_per_rank=1, **settings)
            for value in (0, -1):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, "must be positive"),
                ):
                    self.buffer(**{name: value})

    def test_constructor_and_size_hint_require_supported_topk(self):
        with self.assertRaisesRegex(TypeError, "num_topk"):
            self.ep.ElasticBuffer(0, 1, 1, 1, 256, 1)
        with self.assertRaisesRegex(TypeError, "num_topk"):
            self.ep.ElasticBuffer.get_buffer_size_hint(1, 1, 1, 256)
        for topk in (0, -1, 33, True, False, 1.0, "1", None):
            for construct in (False, True):
                with (
                    self.subTest(topk=topk, construct=construct),
                    self.assertRaisesRegex(
                        ValueError, "num_topk must be an integer between 1 and 32"
                    ),
                ):
                    if construct:
                        self.buffer(num_topk=topk)
                    else:
                        self.ep.ElasticBuffer.get_buffer_size_hint(1, 1, 1, 256, topk)
        for topk in (1, 32):
            with self.subTest(topk=topk):
                self.assertEqual(self.buffer(num_topk=topk).num_bytes, 0)
                self.assertEqual(
                    self.ep.ElasticBuffer.get_buffer_size_hint(1, 1, 1, 256, topk), 0
                )

    def test_fixed_sm_count_is_per_buffer_and_read_only(self):
        with patch.dict(os.environ, {"NIXL_EP_ADAPTIVE_NUM_LAUNCH_SMS": "64"}):
            small = self.buffer(num_launch_sms=8)
            large = self.buffer(num_launch_sms=32)
            default = self.buffer()
        self.assertEqual(default.num_launch_sms, 24)
        for buffer, count in ((small, 8), (large, 32), (default, 24)):
            with self.subTest(count=count):
                with self.assertRaises(AttributeError):
                    buffer.num_launch_sms = 64
                for topk in (1, 2):
                    received, _, _, handle, _ = buffer.dispatch(
                        self.tensor((5, 256)),
                        self.tensor((5, topk), self.ep.topk_idx_t),
                    )
                    self.assertEqual(buffer.num_launch_sms, count)
                    self.assertFalse(hasattr(handle, "num_sms"))
                    buffer.combine(received, handle)
        self.assertEqual(small.num_launch_sms, 8)
        self.assertEqual(large.num_launch_sms, 32)

    def test_constructor_validates_fixed_sm_count(self):
        for count in (0, -2, 1, 3, 65, 66, 2.0, "2", None, True, False):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(
                    ValueError,
                    "num_launch_sms must be an even integer between 2 and 64",
                ),
            ):
                self.buffer(num_launch_sms=count)
        for count in (2, 64):
            with self.subTest(count=count):
                buffer = self.buffer(num_launch_sms=count)
                received, _, _, handle, _ = buffer.dispatch(
                    self.tensor((5, 256)), self.tensor((5, 2), self.ep.topk_idx_t)
                )
                buffer.combine(received, handle)
                self.assertEqual(buffer.num_launch_sms, count)

    def test_sm_configuration_is_only_available_at_construction(self):
        constructor = inspect.signature(self.ep.ElasticBuffer).parameters
        self.assertEqual(constructor["num_launch_sms"].default, 24)
        self.assertIs(
            constructor["num_launch_sms"].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        self.assertNotIn(
            "num_sms", inspect.signature(self.ep.ElasticBuffer.dispatch).parameters
        )
        self.assertNotIn(
            "num_active_sms",
            inspect.signature(self.ep.ElasticBuffer.set_active_ranks).parameters,
        )
        self.assertNotIn("num_sms", inspect.signature(self.ep.EPHandle).parameters)
        buffer = self.two_rank_buffer()
        for name in (
            "set_num_active_sms",
            "get_theoretical_num_sms",
            "_validate_sms",
            "_sms",
            "_auto_sms",
            "_last_num_topk",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(buffer, name))
        before = buffer.get_active_ranks(), buffer.num_launch_sms
        with self.assertRaises(TypeError):
            buffer.dispatch(
                self.tensor((5, 256)),
                self.tensor((5, 2), self.ep.topk_idx_t),
                num_sms=6,
            )
        with self.assertRaises(TypeError):
            buffer.set_active_ranks([0], num_active_sms=6)
        with self.assertRaises(TypeError):
            buffer.set_active_ranks([0], 6)
        self.assertEqual(
            (buffer.get_active_ranks(), buffer.num_launch_sms),
            before,
        )

    def test_single_rank_pairs_before_connection_and_after_shrink(self):
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        for rank, domains, positions in ((0, 1, 1), (3, 2, 2)):
            with self.subTest(rank=rank, maximum=(domains, positions)):
                buffer = self.buffer(
                    world_rank=rank,
                    max_num_scaleout_ranks=domains,
                    max_num_scaleup_ranks=positions,
                    store=None,
                )
                received, _, _, handle, _ = buffer.dispatch(x, idx)
                combined, _ = buffer.combine(received, handle)
                self.assert_placeholder(combined, (5, 256), torch.bfloat16)
                self.assertEqual(buffer.get_connected_ranks(), [])
                if positions == 1:
                    continue
                buffer.connect_ranks([2], store=ForbiddenStore())
                for active in ([2, 3], [3]):
                    buffer.set_active_ranks(active)
                    received, _, _, handle, _ = buffer.dispatch(x, idx)
                    combined, _ = buffer.combine(received, handle)
                    self.assert_placeholder(combined, (5, 256), torch.bfloat16)
                buffer.set_active_ranks([2])
                with self.assertRaisesRegex(ValueError, "this rank to be active"):
                    buffer.dispatch(x, idx)
                with self.assertRaisesRegex(ValueError, "this rank to be active"):
                    buffer.combine(received, handle)

    def test_failed_dispatch_preserves_topology_fixed_sms_and_existing_handle(self):
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        elastic = importlib.import_module(f"{self.module_name}.buffers.elastic")
        for sms in (2, 24, 64):
            with self.subTest(sms=sms):
                buffer = self.two_rank_buffer(num_launch_sms=sms)
                received, _, _, handle, _ = buffer.dispatch(x, idx)
                with self.assertRaises((IndexError, ValueError)):
                    buffer.dispatch(x, self.tensor((5,), self.ep.topk_idx_t))
                # Fail after allocating a receive placeholder.
                allocate = elastic._empty
                calls = 0

                def fail_later(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("placeholder allocation failed")
                    return allocate(*args)

                with patch.object(elastic, "_empty", side_effect=fail_later):
                    with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                        buffer.dispatch(x, self.tensor((5, 1), self.ep.topk_idx_t))
                self.assertEqual(buffer.get_active_ranks(), [0, 1])
                self.assertEqual(buffer.num_launch_sms, sms)
                combined, _ = buffer.combine(received, handle)
                self.assert_placeholder(combined, (5, 256), torch.bfloat16)

    def test_failed_first_dispatch_preserves_fixed_sm_count(self):
        buffer = self.buffer(num_launch_sms=32)
        elastic = importlib.import_module(f"{self.module_name}.buffers.elastic")
        with patch.object(
            elastic, "_empty", side_effect=RuntimeError("allocation failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                buffer.dispatch(
                    self.tensor((5, 256)), self.tensor((5, 2), self.ep.topk_idx_t)
                )
        self.assertEqual(buffer.num_launch_sms, 32)
        self.assertEqual(buffer.get_active_ranks(), [0])

    def test_invalid_topology_update_preserves_membership_fixed_sms_and_handle(self):
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        for sms in (2, 24, 64):
            for active in ([], [-1], [4], [0, 1, 2], [0, 2]):
                with self.subTest(sms=sms, active=active):
                    buffer = self.two_rank_buffer(num_launch_sms=sms)
                    received, _, _, handle, _ = buffer.dispatch(x, idx)
                    with self.assertRaises(ValueError):
                        buffer.set_active_ranks(active)
                    self.assertEqual(buffer.get_active_ranks(), [0, 1])
                    self.assertEqual(buffer.get_connected_ranks(), [1])
                    self.assertEqual(buffer.num_launch_sms, sms)
                    combined, _ = buffer.combine(received, handle)
                    self.assert_placeholder(combined, (5, 256), torch.bfloat16)

    def test_fixed_sms_survive_topk_and_membership_changes(self):
        x = self.tensor((5, 256))
        for sms in (2, 8, 24, 64):
            with self.subTest(sms=sms):
                buffer = self.two_rank_buffer(num_launch_sms=sms)
                buffer.connect_ranks([2, 3])
                for active in ([0, 1, 2, 3], [0, 3], [0, 1, 3], [0], [0, 1]):
                    buffer.set_active_ranks(active)
                    for topk in (1, 2):
                        received, _, _, handle, _ = buffer.dispatch(
                            x, self.tensor((5, topk), self.ep.topk_idx_t)
                        )
                        combined, _ = buffer.combine(received, handle)
                        self.assert_placeholder(combined, (5, 256), torch.bfloat16)
                        self.assertEqual(buffer.num_launch_sms, sms)
                buffer.set_active_ranks([2, 3])
                self.assertEqual(buffer.num_launch_sms, sms)
                with self.assertRaisesRegex(ValueError, "this rank to be active"):
                    buffer.dispatch(x, self.tensor((5, 1), self.ep.topk_idx_t))
                buffer.set_active_ranks([0, 1, 2, 3])
                self.assertEqual(buffer.num_launch_sms, sms)

    def test_fixed_sms_and_destroyed_lifecycle(self):
        buffer = self.two_rank_buffer(num_launch_sms=6)
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        received, _, _, handle, _ = buffer.dispatch(x, idx)
        buffer.combine(received, handle)
        self.assertEqual(buffer.num_launch_sms, 6)
        buffer.destroy()
        for name, action in (
            ("dispatch", lambda: buffer.dispatch(x, idx)),
            ("combine", lambda: buffer.combine(received, handle)),
            ("barrier", buffer.barrier),
            ("set_active_ranks", lambda: buffer.set_active_ranks([0, 1])),
            ("disconnect", lambda: buffer.disconnect_ranks([])),
            ("connect", lambda: buffer.connect_ranks([2])),
            ("active", buffer.get_active_ranks),
            ("connected", buffer.get_connected_ranks),
        ):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                action()

    def test_dispatch_and_combine_placeholders_preserve_wire_format_structure(self):
        for format_name, dtype, columns, scale_dtype, scale_columns in (
            ("bf16", torch.bfloat16, 256, None, None),
            ("fp8", torch.float8_e4m3fn, 256, torch.float32, 2),
            ("nvfp4", torch.uint8, 128, torch.uint8, 16),
        ):
            with self.subTest(format=format_name):
                buffer = self.two_rank_buffer()
                data = self.tensor((5, columns), dtype)
                x = (
                    data
                    if scale_dtype is None
                    else (data, self.tensor((5, scale_columns), scale_dtype))
                )
                idx = self.tensor((5, 2), self.ep.topk_idx_t)
                weights = self.tensor((5, 2), torch.float32)
                result = buffer.dispatch(x, idx, topk_weights=weights)
                self.assertEqual(len(result), 5)
                received, indices, recv_weights, handle, event = result
                if scale_dtype is None:
                    self.assert_placeholder(received, (0, columns), dtype)
                else:
                    self.assertIsInstance(received, tuple)
                    self.assertEqual(len(received), 2)
                    self.assert_placeholder(received[0], (0, columns), dtype)
                    expected_scale = (
                        torch.float8_e4m3fn if format_name == "nvfp4" else scale_dtype
                    )
                    self.assert_placeholder(
                        received[1], (0, scale_columns), expected_scale
                    )
                self.assert_placeholder(indices, (0, 2), self.ep.topk_idx_t)
                self.assert_placeholder(recv_weights, (0, 2), torch.float32)
                self.assertIsInstance(handle, self.ep.EPHandle)
                self.assertIsInstance(event, self.ep.EventOverlap)
                self.assertIsNone(event.event)
                combined = buffer.combine(self.tensor((0, 256)), handle)
                self.assertEqual(len(combined), 2)
                self.assert_placeholder(combined[0], (5, 256), torch.bfloat16)
                self.assertIsInstance(combined[1], self.ep.EventOverlap)

    def test_expanded_optional_weights_handle_ownership_and_device_count(self):
        buffer = self.two_rank_buffer(deterministic=True)
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        weights = self.tensor((5, 2), torch.float32)
        for expand, supplied_weights, copy_handle, cpu_sync in (
            (False, None, False, False),
            (True, None, True, True),
            (True, weights, False, False),
        ):
            with self.subTest(
                expand=expand, weights=supplied_weights is not None, copy=copy_handle
            ):
                received, indices, recv_weights, handle, event = buffer.dispatch(
                    x,
                    idx,
                    topk_weights=supplied_weights,
                    do_expand=expand,
                    do_handle_copy=copy_handle,
                    do_cpu_sync=cpu_sync,
                    expert_alignment=4,
                    async_with_compute_stream=True,
                )
                self.assert_placeholder(received, (0, 256), torch.bfloat16)
                if expand:
                    self.assertIsNone(indices)
                if supplied_weights is None:
                    self.assertIsNone(recv_weights)
                else:
                    self.assert_placeholder(recv_weights, (0,), torch.float32)
                if copy_handle:
                    self.assertIsNot(handle.topk_idx, idx)
                    self.assert_placeholder(handle.topk_idx, (5, 2), self.ep.topk_idx_t)
                else:
                    self.assertIs(handle.topk_idx, idx)
                self.assertFalse(hasattr(handle, "recv_count_is_exact"))
                self.assertEqual(
                    handle.num_recv_tokens_per_expert_list,
                    [0] * buffer.num_experts_per_rank if cpu_sync else [],
                )
                self.assertEqual(handle.do_expand, expand)
                self.assertEqual(handle.expert_alignment, 4)
                self.assertEqual(handle.num_max_tokens_per_rank, 8)
                self.assert_placeholder(handle.num_recv_tokens_device, (), torch.int32)
                self.assert_placeholder(
                    handle.num_unaligned_recv_tokens_per_expert, (3,), torch.int32
                )
                self.assertIsNone(
                    handle.deterministic_sort(received, None, indices, recv_weights)
                )
                self.assertIsInstance(event.event, self.ep.EventHandle)
                combined, completed = buffer.combine(
                    self.tensor((0, 256)), handle, async_with_compute_stream=True
                )
                self.assert_placeholder(combined, (5, 256), torch.bfloat16)
                self.assertIsInstance(completed.event, self.ep.EventHandle)

    def test_event_context_retains_api_but_never_executes_hooks(self):
        event = self.ep.ElasticBuffer.capture()
        self.assertIsInstance(event, self.ep.EventHandle)
        self.assertIsNone(event.current_stream_wait())
        extra = (self.tensor((1,)),)
        overlap = self.ep.EventOverlap(event, extra)
        self.assertIs(overlap.extra_tensors, extra)
        hook = Mock(side_effect=AssertionError("A shell wait must not run user code"))
        overlap.register_hook_after_wait(hook)
        with self.assertRaises(ValueError):
            overlap.register_hook_after_wait(hook)
        self.assertIs(overlap(), overlap)
        with overlap as entered:
            self.assertIs(entered, overlap)
        hook.assert_not_called()
        self.assertIsNone(overlap.hook_after_wait)
        self.assertIs(overlap.event, event)
        self.assertIs(overlap.extra_tensors, extra)
        overlap.register_hook_after_wait(hook)
        with overlap(release_handle=True):
            pass
        hook.assert_not_called()
        self.assertIsNone(overlap.event)
        self.assertIsNone(overlap.extra_tensors)
        with self.ep.EventOverlap() as empty:
            self.assertIsNone(empty.event)
        with self.assertRaisesRegex(RuntimeError, "body failure"):
            with self.ep.EventOverlap(event):
                raise RuntimeError("body failure")


if __name__ == "__main__":
    unittest.main()
