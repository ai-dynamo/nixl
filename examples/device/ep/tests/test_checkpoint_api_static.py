# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_checkpoint_preserve_va_python_api_is_exposed():
    buffer_py = _repo_root() / "examples/device/ep/nixl_ep/buffer.py"
    source = buffer_py.read_text()

    for api_name in (
        "get_graph_visible_addresses",
        "validate_graph_visible_addresses",
        "checkpoint_pause_preserve_va",
        "checkpoint_resume_preserve_va",
    ):
        assert f"def {api_name}" in source


def test_checkpoint_preserve_va_cpp_bindings_are_exposed():
    binding_cpp = _repo_root() / "examples/device/ep/csrc/nixl_ep.cpp"
    source = binding_cpp.read_text()

    for api_name in (
        "get_graph_visible_addresses",
        "validate_graph_visible_addresses",
        "checkpoint_pause_preserve_va",
        "checkpoint_resume_preserve_va",
    ):
        assert f'"{api_name}"' in source


def test_high_throughput_kernels_use_stable_gpu_context_pointer():
    api_cuh = _repo_root() / "examples/device/ep/csrc/kernels/api.cuh"
    ht_cu = _repo_root() / "examples/device/ep/csrc/kernels/nixl_ep_ht.cu"

    assert "gpu_nixl_ctx nixl_ctx);" not in api_cuh.read_text()
    ht_source = ht_cu.read_text()
    for api_name in (
        "void notify_dispatch",
        "void dispatch",
        "void cached_notify",
        "void combine",
    ):
        assert api_name in ht_source
    assert "gpu_nixl_ctx* nixl_ctx" in ht_source
    assert "const auto nixl_ctx = *nixl_ctx_ptr;" in ht_source


def test_cuda_vmm_is_required_for_graph_stable_buffers():
    vmm_cpp = _repo_root() / "examples/device/ep/csrc/vmm.cpp"
    source = vmm_cpp.read_text()

    assert "cuMemAlloc fallback" not in source
    assert "CUDA VMM fabric memory is required" in source
