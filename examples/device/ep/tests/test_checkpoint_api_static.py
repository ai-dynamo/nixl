# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_multinode_module():
    script = (
        _repo_root()
        / "examples/device/ep/tests/checkpoint_preserve_va_multinode.py"
    )
    spec = importlib.util.spec_from_file_location(
        "checkpoint_preserve_va_multinode_static",
        script,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.cuda = SimpleNamespace(CUDAGraph=object)
    dist_stub = types.ModuleType("torch.distributed")
    dist_stub.TCPStore = object
    torch_stub.distributed = dist_stub
    with mock.patch.dict(
        sys.modules,
        {
            spec.name: module,
            "torch": torch_stub,
            "torch.distributed": dist_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _spawn_args(**overrides):
    defaults = {
        "spawn_local_ranks": 0,
        "global_world_size": None,
        "rank_base": None,
        "node_rank": None,
        "force_ucx_tcp": False,
        "ucx_gda_auto_device": False,
        "ucx_gda_device": None,
        "ucx_mixed_gda_intranode": False,
        "ucx_intranode": False,
        "external_store": False,
        "store_master_addr": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _transport_args(**overrides):
    defaults = {
        "ucx_gda_auto_device": False,
        "force_ucx_tcp": False,
        "ucx_intranode": False,
        "ucx_mixed_gda_intranode": False,
        "ucx_tls": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


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
    assert "connection waits also use" in source


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
        "--ucx-gda-device",
        "--ucx-gda-device-candidates",
        "NIXL_EP_UCX_GDA_DEVICE",
        "NIXL_EP_UCX_GDA_DEVICE_CANDIDATES",
    ):
        assert expected in source

    assert "full_device_pattern.fullmatch(device)" in source
    assert "cuda_prefix = f\"cuda{cuda_device}-\"" in source
    assert "def ordinary_hca_from_ucx_rc_gda_device" in source
    assert "return f\"{full_gda_device},{hca_device}\"" in source
    assert "explicit_device=args.ucx_gda_device" in source
    assert "candidate_devices=candidate_devices" in source
    assert "selected_devices = format_ucx_gda_net_devices(selected_device)" in source
    assert "os.environ[\"UCX_NET_DEVICES\"] = selected_devices" in source

    main_source = source[source.index("def main() -> None:") :]
    assert main_source.index("initialize_cuda_runtime(env)") < main_source.index(
        "configure_ucx_gda_auto_device(args, env)"
    )
    assert main_source.index("configure_ucx_gda_auto_device(args, env)") < (
        main_source.index("import_nixl_ep()")
    )


def test_multinode_checkpoint_reproducer_supports_intranode_ucx():
    script = (
        _repo_root()
        / "examples/device/ep/tests/checkpoint_preserve_va_multinode.py"
    )
    source = script.read_text()

    for expected in (
        "--ucx-intranode",
        "NIXL_EP_UCX_INTRANODE",
        'INTRANODE_DEFAULT_UCX_TLS = "sm,cuda_ipc,cuda_copy,self"',
        "INTRANODE_SHARED_MEMORY_TLS",
        "validate_intranode_ucx_tls",
        "configure_ucx_intranode_transport(args)",
        'os.environ["UCX_TLS"] = ucx_tls',
        'os.environ["UCX_NET_DEVICES"] = "all"',
        "--ucx-intranode conflicts with --force-ucx-tcp",
        "--ucx-intranode configures pure same-node CUDA IPC/NVL",
        "single-pod intranode",
    ):
        assert expected in source

    assert 'if "all" in tls:' in source
    assert '"cuda_ipc" not in tls' in source
    assert "tls.isdisjoint(INTRANODE_SHARED_MEMORY_TLS)" in source

    transport_source = source[
        source.index("def configure_ucx_transport(args: argparse.Namespace)")
    :]
    assert transport_source.index("configure_ucx_intranode_transport(args)") < (
        transport_source.index("if args.ucx_tls:")
    )


def test_multinode_checkpoint_reproducer_supports_local_intranode_launcher():
    script = (
        _repo_root()
        / "examples/device/ep/tests/checkpoint_preserve_va_multinode.py"
    )
    source = script.read_text()

    for expected in (
        "--spawn-local-ranks",
        "NIXL_EP_LOCAL_SPAWN_CHILD",
        "run_spawn_local_ranks(args)",
        "subprocess.Popen",
        "sys.executable",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_SPAWN_MASTER_ADDR",
        "--spawn-local-ranks conflicts with --force-ucx-tcp",
        "Kubernetes separate pods usually do not",
        "separate pods commonly cannot use UCX shared-memory",
        "PASS local_spawn_ranks=",
    ):
        assert expected in source

    assert 'base_child_args.append("--ucx-intranode")' in source
    assert "pure same-node validation" in source
    assert 'sys.argv[1:], "--spawn-local-ranks"' in source

    main_source = source[source.index("def main() -> None:") :]
    assert main_source.index("run_spawn_local_ranks(args)") < main_source.index(
        "configure_ucx_transport(args)"
    )


def test_multinode_checkpoint_reproducer_supports_multipod_local_launcher():
    script = (
        _repo_root()
        / "examples/device/ep/tests/checkpoint_preserve_va_multinode.py"
    )
    source = script.read_text()

    for expected in (
        "--global-world-size",
        "--rank-base",
        "--global-rank-base",
        "--node-rank",
        "NIXL_EP_GLOBAL_WORLD_SIZE",
        "NIXL_EP_RANK_BASE",
        "NIXL_EP_LOCAL_WORLD_SIZE",
        "resolve_local_spawn_config(args)",
        "global_rank = spawn_config.rank_base + local_rank",
        'child_env["RANK"] = str(global_rank)',
        'child_env["WORLD_SIZE"] = str(spawn_config.global_world_size)',
        'child_env["LOCAL_RANK"] = str(local_rank)',
        'child_env["LOCAL_WORLD_SIZE"]',
        "global ranks ",
        "rank_base=",
        "global_world_size=",
    ):
        assert expected in source

    assert (
        "spawn_config.global_world_size == spawn_config.local_world_size"
        in source
    )
    assert "--spawn-local-ranks with remote ranks requires" in source
    assert "--ucx-gda-device names one rc_gda device" in source


def test_multinode_checkpoint_reproducer_supports_mixed_gda_intranode_mode():
    script = (
        _repo_root()
        / "examples/device/ep/tests/checkpoint_preserve_va_multinode.py"
    )
    source = script.read_text()

    for expected in (
        "--ucx-mixed-gda-intranode",
        "--mixed-local-remote",
        "NIXL_EP_UCX_MIXED_GDA_INTRANODE",
        "MIXED_GDA_INTRANODE_DEFAULT_UCX_TLS",
        'rc_gda,rc,ud,sm,cuda_ipc,cuda_copy,self',
        "validate_mixed_gda_intranode_ucx_tls",
        "configure_ucx_mixed_gda_intranode_transport(args)",
        "--ucx-mixed-gda-intranode requires --ucx-gda-auto-device",
        "one multi-GPU pod per node",
        "LOCAL_WORLD_SIZE",
        "requires both local and remote peers",
    ):
        assert expected in source

    transport_source = source[
        source.index("def configure_ucx_transport(args: argparse.Namespace)")
    :]
    assert transport_source.index(
        "configure_ucx_mixed_gda_intranode_transport(args)"
    ) < transport_source.index("configure_ucx_intranode_transport(args)")


def test_local_spawn_config_assigns_global_ranks_and_requires_external_store():
    module = _load_multinode_module()

    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
        config = module.resolve_local_spawn_config(
            _spawn_args(
                spawn_local_ranks=2,
                global_world_size=4,
                rank_base=2,
                external_store=True,
                store_master_addr="store.example",
            )
        )

    assert config.local_world_size == 2
    assert config.global_world_size == 4
    assert config.rank_base == 2
    assert config.cuda_visible_devices == "0,1"

    try:
        module.resolve_local_spawn_config(
            _spawn_args(spawn_local_ranks=2, global_world_size=4)
        )
    except ValueError as exc:
        assert "--external-store" in str(exc)
    else:
        raise AssertionError("expected remote local-spawn without store to fail")


def test_local_spawn_config_rejects_invalid_intranode_and_gda_combinations():
    module = _load_multinode_module()

    invalid_cases = (
        (
            _spawn_args(
                spawn_local_ranks=2,
                global_world_size=4,
                external_store=True,
                store_master_addr="store.example",
                ucx_intranode=True,
            ),
            "pure intrapod intranode",
        ),
        (
            _spawn_args(
                spawn_local_ranks=8,
                global_world_size=16,
                external_store=True,
                store_master_addr="store.example",
                ucx_gda_auto_device=True,
                ucx_gda_device="cuda0-mlx5_2:1",
            ),
            "--ucx-gda-device names one rc_gda device",
        ),
        (
            _spawn_args(
                spawn_local_ranks=8,
                global_world_size=8,
                ucx_mixed_gda_intranode=True,
                ucx_gda_auto_device=True,
            ),
            "requires remote ranks",
        ),
    )

    for args, expected in invalid_cases:
        try:
            module.resolve_local_spawn_config(args)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"expected {expected!r} validation failure")


def test_runtime_topology_validation_rejects_separate_pod_intranode():
    module = _load_multinode_module()
    env = module.RankEnv(
        rank=0,
        world_size=2,
        local_rank=0,
        local_world_size=1,
    )

    try:
        module.validate_runtime_topology(
            _transport_args(ucx_intranode=True),
            env,
        )
    except ValueError as exc:
        message = str(exc)
        assert "single-pod intranode topology" in message
        assert "separate Kubernetes pods are not supported" in message
    else:
        raise AssertionError("expected invalid separate-pod intranode to fail")


def test_mixed_transport_sets_combined_tls_and_preserves_gda_selection():
    module = _load_multinode_module()

    with mock.patch.dict(os.environ, {}, clear=True):
        module.configure_ucx_transport(
            _transport_args(
                ucx_gda_auto_device=True,
                ucx_mixed_gda_intranode=True,
            )
        )
        assert os.environ["UCX_TLS"] == (
            "rc_gda,rc,ud,sm,cuda_ipc,cuda_copy,self"
        )
        assert os.environ["UCX_IB_GDA_RETAIN_INACTIVE_CTX"] == "y"
        assert os.environ["UCX_GGA_GDA_RETAIN_INACTIVE_CTX"] == "y"

    for ucx_tls, expected in (
        ("rc,sm,cuda_ipc,cuda_copy,self", "include rc_gda"),
        ("rc_gda,rc,cuda_copy,self", "include cuda_ipc"),
        ("rc_gda,cuda_ipc,cuda_copy,self", "include rc or ud"),
    ):
        try:
            module.validate_mixed_gda_intranode_ucx_tls(ucx_tls)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"expected {ucx_tls!r} to fail validation")


def test_ep_connect_failures_have_timeout_diagnostics():
    binding_cpp = _repo_root() / "examples/device/ep/csrc/nixl_ep.cpp"
    source = binding_cpp.read_text()

    for expected in (
        "ep_connect_deadline(timeout_ms)",
        "ep_connect_timed_out(deadline)",
        "Timed out after",
        "waiting for NIXL metadata readiness",
        "waiting for NIXL EP peer info",
        "notification from remote rank",
        "ep_connect_diagnostic_context",
        "UCX_TLS=",
        "UCX_NET_DEVICES=",
        "NIXL_EP_UCX_INTRANODE",
        "NIXL_EP_UCX_GDA_AUTO_DEVICE",
        "genNotif",
        "getNotifs",
        "nixl_status_message(status)",
    ):
        assert expected in source


def test_ep_memory_view_failures_have_actionable_intranode_diagnostics():
    binding_cpp = _repo_root() / "examples/device/ep/csrc/nixl_ep.cpp"
    source = binding_cpp.read_text()

    for expected in (
        "check_ep_mem_view_status",
        "Failed to prepare ",
        "NIXL_EP_UCX_INTRANODE=1",
        "UCX_TLS=sm,cuda_ipc,cuda_copy,self",
        "UCX_NET_DEVICES=all",
        "--ucx-gda-auto-device",
        "Failed to create device memory list(remote): No ",
        "such device",
    ):
        assert expected in source

    assert "prepMemView(remote_descs, gpu_ctx.remote_mvh" in source
    assert '"remote");' in source
    remote_assert = (
        "prepMemView(remote_descs, gpu_ctx.remote_mvh, "
        "&nixl_agent_info->extra_params) == NIXL_SUCCESS"
    )
    assert remote_assert not in source
