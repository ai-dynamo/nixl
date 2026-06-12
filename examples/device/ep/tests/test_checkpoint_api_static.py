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


def test_multinode_checkpoint_reproducer_exercises_preserve_va_flow():
    script = (
        _repo_root()
        / "examples/device/ep/tests/checkpoint_preserve_va_multinode.py"
    )
    source = script.read_text()

    for expected in (
        "Buffer.get_rdma_size_hint",
        "buffer.update_memory_buffers",
        "buffer.connect_ranks",
        "torch.cuda.graph",
        "buffer.get_graph_visible_addresses",
        "buffer.checkpoint_pause_preserve_va",
        "buffer.checkpoint_resume_preserve_va",
        "buffer.validate_graph_visible_addresses",
        "return_recv_hook=use_recv_hooks",
        "captured.graph.replay()",
    ):
        assert expected in source


def test_multinode_checkpoint_reproducer_supports_ibgda_auto_device():
    script = (
        _repo_root()
        / "examples/device/ep/tests/checkpoint_preserve_va_multinode.py"
    )
    source = script.read_text()

    for expected in (
        "--ucx-gda-auto-device",
        "NIXL_EP_UCX_GDA_AUTO_DEVICE",
        "UCX_IB_GDA_RETAIN_INACTIVE_CTX",
        "UCX_GGA_GDA_RETAIN_INACTIVE_CTX",
        "ucx_info",
        "Transport: rc_gda",
        "UCX_NET_DEVICES",
        'GDA_DEFAULT_UCX_TLS = "rc_gda,rc,ud,cuda_copy,cuda_ipc,self"',
        "cuda0-mlx5_5:1,mlx5_5:1",
        "configure_ucx_gda_auto_device(args, env)",
        "import_nixl_ep()",
    ):
        assert expected in source

    assert "full_device_pattern.fullmatch(device)" in source
    assert "cuda_prefix = f\"cuda{cuda_device}-\"" in source
    assert "def ordinary_hca_from_ucx_rc_gda_device" in source
    assert "return f\"{full_gda_device},{hca_device}\"" in source
    assert "selected_devices = format_ucx_gda_net_devices(selected_device)" in source
    assert "os.environ[\"UCX_NET_DEVICES\"] = selected_devices" in source

    main_source = source[source.index("def main() -> None:") :]
    assert main_source.index("initialize_cuda_runtime(env)") < main_source.index(
        "configure_ucx_gda_auto_device(args, env)"
    )
    assert main_source.index("configure_ucx_gda_auto_device(args, env)") < (
        main_source.index("import_nixl_ep()")
    )
