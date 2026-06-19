# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

try:
    import nixl_cu13._utils as utils
    from nixl_cu13._api import nixl_agent, nixl_agent_config
except ModuleNotFoundError:
    import nixl._utils as utils
    from nixl._api import nixl_agent, nixl_agent_config


def _require_cuda_fabric():
    if not utils.has_cuda_support():
        pytest.skip("NIXL was built without CUDA support")
    if utils.cuda_device_count() == 0:
        pytest.skip("No CUDA devices available")
    if not utils.cuda_fabric_supported(0):
        pytest.skip("CUDA fabric VMM is not supported on device 0")


def test_cuda_fabric_support_query_is_safe():
    if not utils.has_cuda_support():
        assert utils.cuda_device_count() == 0
        assert not utils.cuda_fabric_supported(0)
        assert not utils.is_cuda_fabric_vmm_supported(0)
        return
    assert utils.cuda_device_count() >= 0
    assert isinstance(utils.cuda_fabric_supported(0), bool)
    assert utils.is_cuda_fabric_vmm_supported(0) == utils.cuda_fabric_supported(0)


def test_cuda_fabric_allocation_roundtrip():
    _require_cuda_fabric()
    size = 4096
    src = utils.cuda_fabric_vmm_alloc(size, device_id=0, require_fabric=True)
    dst = utils.cuda_fabric_vmm_alloc(size, device_id=0, require_fabric=True)
    try:
        utils.cuda_memset(src, 0xBA, size)
        utils.cuda_memset(dst, 0xBA, size)
        utils.cuda_verify_transfer(src, dst, size)
    finally:
        utils.cuda_fabric_vmm_free(src)
        utils.cuda_fabric_vmm_free(dst)


def test_cuda_fabric_ucx_vram_write_roundtrip():
    _require_cuda_fabric()
    size = 4096
    src = utils.cuda_fabric_vmm_alloc(size, device_id=0, require_fabric=True)
    dst = utils.cuda_fabric_vmm_alloc(size, device_id=0, require_fabric=True)

    try:
        try:
            agent1 = nixl_agent("cuda_fabric_src", nixl_agent_config(backends=["UCX"]))
            agent2 = nixl_agent("cuda_fabric_dst", nixl_agent_config(backends=["UCX"]))
        except Exception as exc:
            pytest.skip(f"UCX backend is not available: {exc}")

        if "VRAM" not in agent1.get_backend_mem_types("UCX"):
            pytest.skip("UCX backend does not advertise VRAM support")

        utils.cuda_memset(src, 0xA5, size)
        utils.cuda_memset(dst, 0x00, size)

        reg1 = agent1.register_memory([(src, size, 0, "")], mem_type="VRAM", backends=["UCX"])
        reg2 = agent2.register_memory([(dst, size, 0, "")], mem_type="VRAM", backends=["UCX"])
        try:
            agent1.add_remote_agent(agent2.get_agent_metadata())

            local = agent1.get_xfer_descs([(src, size, 0)], mem_type="VRAM")
            remote = agent1.get_xfer_descs([(dst, size, 0)], mem_type="VRAM")
            handle = agent1.initialize_xfer("WRITE", local, remote, agent2.name, b"fabric_done", backends=["UCX"])
            state = agent1.transfer(handle)
            assert state in ("DONE", "PROC")
            deadline = time.monotonic() + 10.0
            while state == "PROC" and time.monotonic() < deadline:
                state = agent1.check_xfer_state(handle)
                assert state in ("DONE", "PROC")
            assert state == "DONE"

            notif_done = False
            while time.monotonic() < deadline:
                notif_done = agent2.check_remote_xfer_done(agent1.name, b"fabric_done")
                if notif_done:
                    break
                pass
            assert notif_done

            utils.cuda_verify_transfer(src, dst, size)
            telemetry = agent1.get_xfer_telemetry(handle)
            assert telemetry.descCount == 1
            assert telemetry.totalBytes == size
            agent1.release_xfer_handle(handle)
        finally:
            agent1.deregister_memory(reg1, backends=["UCX"])
            agent2.deregister_memory(reg2, backends=["UCX"])
    finally:
        utils.cuda_fabric_vmm_free(src)
        utils.cuda_fabric_vmm_free(dst)
