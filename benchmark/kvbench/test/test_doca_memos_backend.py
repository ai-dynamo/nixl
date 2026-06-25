# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mock-based behavioral tests for ``DocaMemosBackend``.

NIXL is a C++ binding that is not importable in CI, so we install stub
``nixl``, ``nixl._api``, ``nixl.logging`` and ``torch`` modules *before*
importing the module under test, then drive ``DocaMemosBackend`` against a
fully recording mock NIXL agent.

These are mock tests: they exercise the Python orchestration of the backend
(key building, OBJ registration, transfer wiring) and assert against the
recorded mock calls. They do NOT exercise the runtime DOCA hardware path.

The assertions are real: they fail if the backend behavior is wrong. The
backend source is never modified to make them pass.
"""

import os
import sys
import types

import pytest  # type: ignore

p = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# Stub nixl / torch before importing storage_backend (ordering-safe, hermetic:
# only installs a stub when the real module is absent, as in CI).
# --------------------------------------------------------------------------- #
def _install_stubs():
    if "nixl" not in sys.modules:
        nixl_pkg = types.ModuleType("nixl")

        api_mod = types.ModuleType("nixl._api")

        class nixl_agent:  # noqa: N801 - mirrors the real binding's name
            pass

        api_mod.nixl_agent = nixl_agent
        nixl_pkg._api = api_mod

        logging_mod = types.ModuleType("nixl.logging")
        import logging as _logging

        logging_mod.get_logger = lambda name: _logging.getLogger(name)
        nixl_pkg.logging = logging_mod

        sys.modules["nixl"] = nixl_pkg
        sys.modules["nixl._api"] = api_mod
        sys.modules["nixl.logging"] = logging_mod

    # storage_backend._prestore_read_object does `import torch; torch.zeros(...)`.
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.uint8 = "uint8"

        class _FakeTensor:
            def __init__(self, n):
                self.n = n

        torch_mod.zeros = lambda n, dtype=None: _FakeTensor(n)
        sys.modules["torch"] = torch_mod


_install_stubs()

import storage_backend as sb  # noqa: E402 - must follow stub installation

DocaMemosBackend = sb.DocaMemosBackend


# --------------------------------------------------------------------------- #
# Recording mock NIXL agent.
# --------------------------------------------------------------------------- #
class MockRegDescs:
    def __init__(self, reg_list, mem_type, backends):
        self.reg_list = reg_list
        self.mem_type = mem_type
        self.backends = backends


class MockXferDescs:
    def __init__(self, descs, mem_type):
        self.descs = descs
        self.mem_type = mem_type


class MockXferHandle:
    def __init__(self, op, local, remote, remote_agent, backends):
        self.op = op
        self.local = local
        self.remote = remote
        self.remote_agent = remote_agent
        self.backends = backends


class MockAgent:
    name = "mock_agent"

    def __init__(self):
        self.create_backend_calls = []
        self.register_calls = []
        self.deregister_calls = []
        self.xfer_desc_calls = []
        self.initialize_xfer_calls = []
        self.transfer_calls = []
        self.release_calls = []

    def create_backend(self, name, params):
        self.create_backend_calls.append((name, dict(params)))
        return object()

    def register_memory(self, reg_list, mem_type, backends=None):
        self.register_calls.append((list(reg_list), mem_type, list(backends or [])))
        return MockRegDescs(reg_list, mem_type, backends)

    def deregister_memory(self, reg_descs, backends=None):
        self.deregister_calls.append((reg_descs, list(backends or [])))

    def get_xfer_descs(self, descs, mem_type=None):
        self.xfer_desc_calls.append((descs, mem_type))
        return MockXferDescs(descs, mem_type)

    def initialize_xfer(self, op, local, remote, remote_agent, backends=None):
        h = MockXferHandle(op, local, remote, remote_agent, list(backends or []))
        self.initialize_xfer_calls.append(h)
        return h

    def transfer(self, handle):
        self.transfer_calls.append(handle)
        return "DONE"  # synchronous success

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.release_calls.append(handle)


# --------------------------------------------------------------------------- #
# Fixtures / helpers.
# --------------------------------------------------------------------------- #
@pytest.fixture
def agent():
    return MockAgent()


def make_backend(agent, **kw):
    backend = DocaMemosBackend(agent=agent, device_name="/dev/nvme0n1", **kw)
    return backend


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
def test_create_backend_params_minimal(agent):
    """Only device_name + query_mem_mode by default; optional keys absent."""
    make_backend(agent)
    assert len(agent.create_backend_calls) == 1
    name, params = agent.create_backend_calls[0]
    assert name == "DOCA_MEMOS"
    assert params == {
        "device_name": "/dev/nvme0n1",
        "query_mem_mode": "assume_success",
    }
    assert "num_tasks" not in params
    assert "nguid" not in params
    assert "ignore_read_not_found" not in params


def test_create_backend_params_all_optionals(agent):
    """Optional params appear (as strings) only when provided/true."""
    make_backend(
        agent,
        num_tasks=4096,
        nguid="0123456789abcdef0123456789abcdef",
        query_mem_mode="actual",
        ignore_read_not_found=True,
    )
    name, params = agent.create_backend_calls[0]
    assert params["device_name"] == "/dev/nvme0n1"
    assert params["query_mem_mode"] == "actual"
    assert params["num_tasks"] == "4096"  # stringified
    assert params["nguid"] == "0123456789abcdef0123456789abcdef"
    assert params["ignore_read_not_found"] == "true"


def test_empty_device_name_rejected():
    with pytest.raises(ValueError):
        DocaMemosBackend(agent=MockAgent(), device_name="")


def test_obj_key_16_byte_limit_fires_on_overflow(agent):
    backend = make_backend(agent, key_prefix="kvbench")
    # "kvbench0_3" = 10 bytes -> OK
    assert backend._obj_key(0, 3) == "kvbench0_3"
    # Long prefix pushing the key past 16 bytes must raise.
    big = make_backend(MockAgent(), key_prefix="thisprefixistoolong")
    with pytest.raises(ValueError) as ei:
        big._obj_key(0, 0)
    assert "16-byte" in str(ei.value)
    # Short prefix is fine.
    b2 = make_backend(MockAgent(), key_prefix="p")
    assert b2._obj_key(0, 3) == "p0_3"
    # Boundary: a key encoding to exactly 15 bytes is allowed.
    ok15 = make_backend(MockAgent(), key_prefix="x" * 12)
    assert len(ok15._obj_key(0, 3).encode()) == 15
    # 14-char prefix + "0_3" = 17 bytes -> must raise.
    bad17 = make_backend(MockAgent(), key_prefix="y" * 14)
    with pytest.raises(ValueError):
        bad17._obj_key(0, 3)


def test_prepare_read_only_single_key_and_obj_register_args(agent):
    """read-only op: single shared key (no suffix), OBJ register 4-tuple."""
    backend = make_backend(agent)
    handle = backend.prepare(tp_idx=0, rank=2, read_size=1024, write_size=0)
    assert handle.backend_data["read_key"] == "kvbench0_2"  # no _r suffix
    assert "write_key" not in handle.backend_data
    # Exactly one OBJ registration.
    assert len(agent.register_calls) == 1
    reg_list, mem_type, backends = agent.register_calls[0]
    assert mem_type == "OBJ"
    assert backends == ["DOCA_MEMOS"]
    assert len(reg_list) == 1
    tup = reg_list[0]
    assert len(tup) == 4  # (addr, len, devId, metaInfo)
    addr, length, dev_id, key = tup
    assert addr == 0
    assert length == 1024
    assert isinstance(dev_id, int)
    assert key == "kvbench0_2"


def test_prepare_write_only_single_key(agent):
    backend = make_backend(agent)
    handle = backend.prepare(tp_idx=1, rank=0, read_size=0, write_size=2048)
    assert handle.backend_data["write_key"] == "kvbench1_0"  # no _w suffix
    assert "read_key" not in handle.backend_data
    # write-only: NO pre-store transfer should be issued.
    assert agent.transfer_calls == []


def test_prepare_rw_distinct_suffixed_keys(agent):
    """Both sizes non-zero -> distinct _r / _w keys, two OBJ registrations."""
    backend = make_backend(agent)
    handle = backend.prepare(tp_idx=0, rank=5, read_size=512, write_size=256)
    assert handle.backend_data["read_key"] == "kvbench0_5_r"
    assert handle.backend_data["write_key"] == "kvbench0_5_w"
    assert handle.backend_data["read_dev_id"] != handle.backend_data["write_dev_id"]
    # Two OBJ registrations, both 4-tuple/OBJ/DOCA_MEMOS, sizes match each op.
    assert len(agent.register_calls) == 2
    sizes = {}
    for reg_list, mem_type, backends in agent.register_calls:
        assert mem_type == "OBJ"
        assert backends == ["DOCA_MEMOS"]
        addr, length, dev_id, key = reg_list[0]
        sizes[key] = length
    assert sizes["kvbench0_5_r"] == 512
    assert sizes["kvbench0_5_w"] == 256


def test_prepare_read_triggers_prestore_store(agent):
    """A read op pre-stores: a WRITE (STORE) transfer is issued in prepare()."""
    backend = make_backend(agent)
    backend.prepare(tp_idx=0, rank=0, read_size=4096, write_size=0)
    # initialize_xfer was called with WRITE (the pre-store STORE).
    store_xfers = [h for h in agent.initialize_xfer_calls if h.op == "WRITE"]
    assert len(store_xfers) == 1, "expected exactly one pre-store STORE"
    pre = store_xfers[0]
    assert pre.backends == ["DOCA_MEMOS"]
    assert pre.remote_agent == agent.name
    # transfer() actually submitted, then released.
    assert len(agent.transfer_calls) == 1
    assert len(agent.release_calls) == 1


def test_get_read_handle_initialize_xfer_args(agent):
    backend = make_backend(agent)
    handle = backend.prepare(tp_idx=0, rank=1, read_size=1024, write_size=0)
    agent.initialize_xfer_calls.clear()  # drop the pre-store call
    buf = object()
    rh = backend.get_read_handle(handle, buf)
    assert rh.op == "READ"
    assert rh.backends == ["DOCA_MEMOS"]
    assert rh.remote_agent == agent.name
    # remote side is an OBJ xfer dlist (3-tuple) matched to the read dev_id.
    assert rh.remote.mem_type == "OBJ"
    assert len(rh.remote.descs[0]) == 3  # (addr, len, devId)
    assert rh.remote.descs[0][1] == 1024
    assert rh.remote.descs[0][2] == handle.backend_data["read_dev_id"]


def test_get_write_handle_initialize_xfer_args(agent):
    backend = make_backend(agent)
    handle = backend.prepare(tp_idx=0, rank=1, read_size=0, write_size=2048)
    agent.initialize_xfer_calls.clear()
    buf = object()
    wh = backend.get_write_handle(handle, buf)
    assert wh.op == "WRITE"
    assert wh.backends == ["DOCA_MEMOS"]
    assert wh.remote.mem_type == "OBJ"
    assert len(wh.remote.descs[0]) == 3
    assert wh.remote.descs[0][1] == 2048
    assert wh.remote.descs[0][2] == handle.backend_data["write_dev_id"]


def test_get_handles_return_none_for_zero_size(agent):
    backend = make_backend(agent)
    h_read_only = backend.prepare(tp_idx=2, rank=0, read_size=100, write_size=0)
    assert backend.get_write_handle(h_read_only, object()) is None
    h_write_only = backend.prepare(tp_idx=3, rank=0, read_size=0, write_size=100)
    assert backend.get_read_handle(h_write_only, object()) is None


def test_close_deregisters_all_objs(agent):
    backend = make_backend(agent)
    backend.prepare(tp_idx=0, rank=0, read_size=64, write_size=64)  # 2 objs
    assert len(agent.register_calls) == 2
    backend.close()
    assert len(agent.deregister_calls) == 2
    for _, backends in agent.deregister_calls:
        assert backends == ["DOCA_MEMOS"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
