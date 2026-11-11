# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only API contracts; tensor placeholders never establish numerical correctness.

Run with ``python -m unittest discover -s runtime/ep/tests/api -v``. Set
``NIXL_EP_API_MODULE=nixl.runtime.ep`` to exercise the installed public package.
The checked-in signature snapshot comes from production commit 6b11b2c;
it is deliberately independent of the shell implementation under test.
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
from types import SimpleNamespace
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
        cls.module_name = os.environ.get("NIXL_EP_API_MODULE", "nixl_ep_v2")
        if cls.module_name == "nixl_ep_v2":
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
            rank_idx=0,
            num_ranks=2,
            num_group_scaleup_ranks=2,
            max_num_scaleout_ranks=2,
            max_num_scaleup_ranks=2,
            num_max_tokens_per_rank=8,
            hidden=256,
            num_topk=2,
            num_experts_per_rank=3,
            explicitly_destroy=True,
            store=ForbiddenStore(),
            bootstrap_id="api-only",
        )
        options.update(kwargs)
        return self.ep.ElasticBuffer(**options)

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
            ("utils/envs.py", (), ("get_logical_domain_size",)),
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
        # These defaults refer to module constants in the source signatures.
        sms_parameters = inspect.signature(
            self.ep.ElasticBuffer.get_theoretical_num_sms
        ).parameters
        self.assertEqual(sms_parameters["sm_read_gbs"].default, 200)
        self.assertEqual(sms_parameters["sm_write_gbs"].default, 50)

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
        envs = importlib.import_module(f"{self.module_name}.utils.envs")
        self.assertIs(self.ep.ElasticBuffer, elastic.ElasticBuffer)
        self.assertIs(self.ep.EPHandle, elastic.EPHandle)
        self.assertIs(self.ep.EventHandle, event.EventHandle)
        self.assertIs(self.ep.EventOverlap, event.EventOverlap)
        self.assertIs(self.ep.get_logical_domain_size, envs.get_logical_domain_size)

    def test_configuration_and_local_helpers_do_not_touch_store_or_hardware(self):
        group = SimpleNamespace(rank=lambda: 1, size=lambda: 4)
        store = ForbiddenStore()
        buffer = self.buffer(group=group, rank_idx=None, num_ranks=None, store=store)
        self.assertEqual(
            (buffer.rank_idx, buffer.num_ranks, buffer.world_rank), (1, 4, 1)
        )
        self.assertEqual(buffer.num_experts_per_rank, 3)
        self.assertIs(buffer.store, store)
        self.assertIsNone(buffer.runtime)
        self.assertEqual(buffer.get_active_ranks(), [0, 1, 2, 3])
        self.assertEqual(self.ep.get_logical_domain_size(group, store=store), (1, 4))
        self.assertEqual(
            self.ep.get_logical_domain_size(group, num_scaleup_ranks=2, store=store),
            (2, 2),
        )
        self.assertEqual(buffer.get_buffer_size_hint(group, 8, 256, store=store), 0)
        self.assertIsInstance(buffer.get_local_metadata(), bytes)
        peer = buffer.get_local_peer_info()
        self.assertEqual(len(peer), 4)
        self.assertEqual(peer[2], buffer.world_rank)
        self.assertEqual(buffer.read_profile(), {})
        self.assertIs(buffer.get_comm_stream(), buffer.get_comm_stream())
        self.assertIsNone(buffer.barrier(with_cpu_sync=True))
        self.assertIsNone(buffer.publish(store=store))

    def test_topology_stages_then_commits_without_renumbering_slots(self):
        buffer = self.buffer()
        self.assertEqual(buffer.get_topology_masks(), (1, 3))
        self.assertEqual(buffer.get_topology()[0], "direct")
        self.assertEqual(buffer.world_rank_of(1, 1), 3)
        self.assertEqual(buffer.get_connected_ranks(), [1])
        buffer.connect_ranks([2, 3], store=ForbiddenStore())
        self.assertEqual(buffer.get_connected_ranks(), [1, 2, 3])
        self.assertEqual(buffer.get_active_ranks(), [0, 1])
        buffer.stage_topology(active_ranks=[3, 2, 1, 0], num_active_sms=8)
        self.assertEqual(buffer.get_active_ranks(), [0, 1])
        with self.assertRaises(ValueError):
            buffer.set_num_active_sms(4)
        buffer.commit_topology()
        self.assertEqual(buffer.get_topology(), ("hybrid", 8))
        self.assertEqual(buffer.get_topology_masks(), (3, 3))
        self.assertEqual(buffer.get_topology_extent(), (2, 2))
        with self.assertRaises(ValueError):
            buffer.disconnect_ranks([2])
        buffer.set_active_ranks([2, 3])
        self.assertEqual(buffer.get_topology(), ("direct", 8))
        self.assertEqual(buffer.get_topology_masks(), (2, 3))
        self.assertEqual(buffer.get_topology_extent(), (2, 2))
        self.assertFalse(buffer.is_active)
        self.assertEqual(buffer.num_experts_per_rank, 3)
        buffer.disconnect_ranks([1])
        buffer.commit_topology()
        self.assertEqual(buffer.get_connected_ranks(), [2, 3])
        with self.assertRaises(ValueError):
            buffer.masks_of([0, 1, 2])

    def test_inactive_rank_can_retire_peers_from_committed_and_staged_worlds(self):
        buffer = self.buffer()
        buffer.connect_ranks([2, 3], store=ForbiddenStore())
        buffer.set_active_ranks([2, 3])
        buffer.stage_topology(active_ranks=[1])
        self.assertFalse(buffer.is_active)
        buffer.disconnect_ranks([1, 2, 3])
        self.assertEqual(buffer.get_connected_ranks(), [])
        self.assertEqual(buffer.get_active_ranks(), [2, 3])
        buffer.commit_topology()
        self.assertEqual(buffer.get_active_ranks(), [1])
        self.assertFalse(buffer.is_active)

    def test_disconnect_protects_each_world_only_when_local_rank_is_active(self):
        for committed, staged in (([0, 1], [2, 3]), ([2, 3], [0, 1])):
            with self.subTest(committed=committed, staged=staged):
                buffer = self.buffer()
                buffer.connect_ranks([2, 3], store=ForbiddenStore())
                buffer.set_active_ranks(committed)
                buffer.stage_topology(active_ranks=staged)
                with self.assertRaises(ValueError):
                    buffer.disconnect_ranks([2, 1])
                self.assertEqual(buffer.get_connected_ranks(), [1, 2, 3])
                buffer.disconnect_ranks([2, 3])
                self.assertEqual(buffer.get_connected_ranks(), [1])
                self.assertEqual(buffer.get_active_ranks(), committed)
                buffer.commit_topology()
                self.assertEqual(buffer.get_active_ranks(), staged)

    def test_connect_rejects_duplicate_and_connected_ranks_before_mutation(self):
        for ranks in ([2, 2], [2, 1]):
            with self.subTest(ranks=ranks):
                buffer = self.buffer()
                with self.assertRaises(ValueError):
                    buffer.connect_ranks(ranks, store=ForbiddenStore())
                self.assertEqual(buffer.get_connected_ranks(), [1])
                self.assertEqual(buffer.get_active_ranks(), [0, 1])
                with self.assertRaises(ValueError):
                    buffer.commit_topology()

    def test_connect_rejects_incomplete_metadata_without_changing_staged_state(self):
        metadata = dict(
            remote_mds=[b"peer-2", b"peer-3"],
            peer_infos=[(0, 0, 2, 0), (0, 0, 3, 0)],
        )
        invalid = [
            {key: value for key, value in metadata.items() if key != missing}
            for missing in metadata
        ]
        for key, values in metadata.items():
            for wrong_length in ([], values[:1], values + values[:1]):
                invalid.append({**metadata, key: wrong_length})
        invalid.append({key: values[:1] for key, values in metadata.items()})
        for incomplete in invalid:
            with self.subTest(metadata=incomplete):
                buffer = self.buffer()
                with self.assertRaises(ValueError):
                    buffer.connect_ranks([2, 3], **incomplete, store=ForbiddenStore())
                self.assertEqual(buffer.get_connected_ranks(), [1])
                self.assertEqual(buffer.get_active_ranks(), [0, 1])
                with self.assertRaises(ValueError):
                    buffer.commit_topology()

                buffer.stage_topology(active_ranks=[0], num_active_sms=8)
                with self.assertRaises(ValueError):
                    buffer.connect_ranks([2, 3], **incomplete, store=ForbiddenStore())
                self.assertEqual(buffer.get_connected_ranks(), [1])
                self.assertEqual(buffer.get_active_ranks(), [0, 1])
                buffer.commit_topology()
                self.assertEqual(buffer.get_active_ranks(), [0])
                self.assertEqual(buffer.get_topology(), ("direct", 8))

        buffer = self.buffer()
        buffer.connect_ranks([2, 3], **metadata, store=ForbiddenStore())
        self.assertEqual(buffer.get_connected_ranks(), [1, 2, 3])
        buffer.commit_topology()
        self.assertEqual(buffer.get_active_ranks(), [0, 1])

    def test_disconnected_rank_can_reconnect_only_after_commit(self):
        buffer = self.buffer()
        buffer.connect_ranks([2], store=ForbiddenStore())
        buffer.commit_topology()
        for ranks in ([2, 3], [2, 2]):
            with self.subTest(disconnect=ranks):
                with self.assertRaises(ValueError):
                    buffer.disconnect_ranks(ranks)
                self.assertEqual(buffer.get_connected_ranks(), [1, 2])
                with self.assertRaises(ValueError):
                    buffer.commit_topology()
        buffer.disconnect_ranks([2])
        self.assertEqual(buffer.get_connected_ranks(), [1])
        with self.assertRaises(ValueError):
            buffer.connect_ranks([3, 2], store=ForbiddenStore())
        self.assertEqual(buffer.get_connected_ranks(), [1])
        buffer.commit_topology()
        buffer.connect_ranks([2], store=ForbiddenStore())
        self.assertEqual(buffer.get_connected_ranks(), [1, 2])
        buffer.commit_topology()
        self.assertEqual(buffer.get_active_ranks(), [0, 1])

    def test_sm_updates_commit_pending_disconnects_when_they_recommit_topology(self):
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        for method, sms in (("set", 4), ("set", 6), ("dispatch", 4), ("dispatch", 6)):
            with self.subTest(method=method, sms=sms):
                buffer = self.buffer()
                buffer.set_num_active_sms(4)
                buffer.connect_ranks([2], store=ForbiddenStore())
                buffer.commit_topology()
                buffer.disconnect_ranks([2])
                if method == "set":
                    buffer.set_num_active_sms(sms)
                else:
                    buffer.dispatch(x, idx, num_sms=sms)
                self.assertEqual(buffer.get_topology(), ("direct", sms))
                if method == "dispatch" and sms == 4:
                    with self.assertRaises(ValueError):
                        buffer.connect_ranks([2], store=ForbiddenStore())
                    buffer.commit_topology()
                buffer.connect_ranks([2], store=ForbiddenStore())
                self.assertEqual(buffer.get_connected_ranks(), [1, 2])
                self.assertEqual(buffer.get_active_ranks(), [0, 1])

    def test_constructor_rejects_coordinates_outside_each_axis(self):
        for domain, position in ((-1, 0), (2, 0), (0, -1), (0, 2), (1, -1), (2, -1)):
            with self.subTest(domain=domain, position=position):
                with self.assertRaises(ValueError):
                    self.buffer(
                        num_ranks=1,
                        num_group_scaleup_ranks=1,
                        scaleout_rank_idx=domain,
                        scaleup_rank_idx=position,
                    )
        # Valid local coordinates cannot let the cohort wrap into another domain.
        with self.assertRaises(ValueError):
            self.buffer(
                rank_idx=0,
                num_ranks=2,
                num_group_scaleup_ranks=2,
                max_num_scaleout_ranks=2,
                max_num_scaleup_ranks=1,
                scaleout_rank_idx=0,
                scaleup_rank_idx=0,
            )
        # A singleton at the final valid cell establishes the upper boundary.
        buffer = self.buffer(
            num_ranks=1,
            num_group_scaleup_ranks=1,
            scaleout_rank_idx=1,
            scaleup_rank_idx=1,
        )
        self.assertEqual(buffer.world_rank, 3)
        self.assertEqual(buffer.get_active_ranks(), [3])

    def test_failed_dispatch_preserves_manual_sms_and_existing_handle(self):
        buffer = self.buffer()
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        received, _, _, handle, _ = buffer.dispatch(x, idx, num_sms=4)
        malformed_idx = self.tensor((5,), self.ep.topk_idx_t)
        with self.assertRaises((IndexError, ValueError)):
            buffer.dispatch(x, malformed_idx)
        self.assertEqual(buffer.get_topology(), ("direct", 4))
        combined, _ = buffer.combine(received, handle)
        self.assert_placeholder(combined, (5, 256), torch.bfloat16)
        buffer.connect_ranks([2, 3], store=ForbiddenStore())
        buffer.stage_topology(active_ranks=[0, 1, 2, 3])
        buffer.commit_topology()
        self.assertEqual(buffer.get_topology(), ("hybrid", 4))

    def test_staged_topology_preserves_automatic_dispatch_topk_policy(self):
        buffer = self.buffer()
        x = self.tensor((5, 256))
        idx_one = self.tensor((5, 1), self.ep.topk_idx_t)
        idx_two = self.tensor((5, 2), self.ep.topk_idx_t)
        buffer.dispatch(x, idx_one)
        buffer.connect_ranks([2, 3], store=ForbiddenStore())
        buffer.stage_topology(active_ranks=[0, 1, 2, 3])
        with self.assertRaises(ValueError):
            buffer.dispatch(x, idx_two)
        buffer.dispatch(x, idx_one)
        self.assertEqual(buffer.get_active_ranks(), [0, 1])
        buffer.commit_topology()
        self.assertEqual(buffer.get_active_ranks(), [0, 1, 2, 3])
        _, _, _, handle, _ = buffer.dispatch(x, idx_two)
        self.assert_placeholder(handle.topk_idx, (5, 2), self.ep.topk_idx_t)

    def test_manual_sms_and_destroyed_lifecycle(self):
        buffer = self.buffer()
        buffer.set_num_active_sms(6)
        buffer.stage_topology(active_ranks=[0, 1])
        buffer.commit_topology()
        self.assertEqual(buffer.get_topology(), ("direct", 6))
        x = self.tensor((5, 256))
        idx = self.tensor((5, 2), self.ep.topk_idx_t)
        received, _, _, handle, _ = buffer.dispatch(x, idx, num_sms=6)
        self.assertEqual(handle.num_sms, 6)
        buffer.set_num_active_sms(4)
        with self.assertRaises(ValueError):
            buffer.combine(received, handle)
        buffer.destroy()
        for name, action in (
            ("dispatch", lambda: buffer.dispatch(x, idx)),
            ("combine", lambda: buffer.combine(received, handle)),
            ("barrier", buffer.barrier),
            ("stage", buffer.stage_topology),
            ("connect", lambda: buffer.connect_ranks([2])),
            ("topology", buffer.get_topology),
            ("active", lambda: buffer.is_active),
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
                buffer = self.buffer()
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
        buffer = self.buffer(deterministic=True)
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
                self.assertEqual(handle.recv_count_is_exact, cpu_sync)
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
        overlap.register_hook_after_wait(hook)
        with overlap(release_handle=True):
            pass
        hook.assert_not_called()
        self.assertIsNone(overlap.event)
        with self.ep.EventOverlap() as empty:
            self.assertIsNone(empty.event)
        with self.assertRaisesRegex(RuntimeError, "body failure"):
            with self.ep.EventOverlap(event):
                raise RuntimeError("body failure")


if __name__ == "__main__":
    unittest.main()
