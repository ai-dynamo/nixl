# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _function_body(source: str, signature: str) -> str:
    match = re.search(
        rf"def {re.escape(signature)}\n(?P<body>.*?)(?=\ndef |\nif __name__ ==)",
        source,
        re.S,
    )
    assert match is not None, f"missing function body for {signature}"
    return match.group("body")


def _ci_matrix_step(source: str, name: str) -> str:
    match = re.search(
        rf"  - name: {re.escape(name)}\n(?P<body>.*?)(?=\n  - name: |\n\npipeline_stop:)",
        source,
        re.S,
    )
    assert match is not None, f"missing CI matrix step {name!r}"
    return match.group("body")


def test_dl_gpu_flow_builds_nixl_ep() -> None:
    dl_matrix = read_repo_file(".ci/jenkins/lib/test-dl-matrix.yaml")
    gpu_dockerfile = read_repo_file(".ci/dockerfiles/Dockerfile.gpu-test")

    assert "--build-arg BUILD_NIXL_EP=true" in dl_matrix
    assert "ARG BUILD_NIXL_EP=false" in gpu_dockerfile
    assert "-Dbuild_nixl_ep=true" in gpu_dockerfile
    assert "-Dbuild_examples=true" in gpu_dockerfile
    assert ".gitlab/build.sh ${NIXL_INSTALL_DIR}" in gpu_dockerfile
    assert '"${EXTRA_BUILD_ARGS}"' in gpu_dockerfile


def test_dl_gpu_flow_has_dedicated_ep_ht_correctness_step() -> None:
    dl_matrix = read_repo_file(".ci/jenkins/lib/test-dl-matrix.yaml")
    step = _ci_matrix_step(dl_matrix, "Run DL NIXL EP HT correctness test")

    assert "Run DL NIXL EP HT correctness test" in dl_matrix
    assert (
        "testScript: \"bash .ci/scripts/run_ep_ht_correctness_ci.sh ${NIXL_INSTALL_DIR}\""
        in step
    )
    assert "slurmEnv" not in step
    assert "SLURM_NODES=2" not in step


def test_ep_ht_ci_entrypoint_limits_scope_to_four_gpu_correctness() -> None:
    entrypoint = read_repo_file(".ci/scripts/run_ep_ht_correctness_ci.sh")

    assert "NIXL EP HT 4-GPU correctness" in entrypoint
    assert "--correctness-target four_gpu_single_node" in entrypoint
    assert "--ci-correctness-only" not in entrypoint
    assert "--num-processes 4" in entrypoint
    assert "NIXL_EP_HT_CI_NUM_PROCESSES" not in entrypoint
    assert "This check does not cover multi-node RDMA or performance." in entrypoint


def test_ht_test_keeps_default_full_scale_path_and_adds_ci_mode() -> None:
    ht_test = read_repo_file("examples/device/ep/tests/test_ht.py")

    assert "--correctness-target" in ht_test
    assert "four_gpu_single_node" in ht_test
    assert "default=8" in ht_test
    assert "args.ci_correctness_only" in ht_test
    assert "resolve_correctness_target" in ht_test
    assert "num_local_ranks == 8" in ht_test


def test_four_gpu_preflight_runs_before_tcpstore_and_process_spawn() -> None:
    ht_test = read_repo_file("examples/device/ep/tests/test_ht.py")

    preflight_index = ht_test.index("preflight_four_gpu_single_node_target(")
    tcpstore_index = ht_test.index('print("Starting TCPStore and rank server locally"')
    spawn_index = ht_test.index("torch.multiprocessing.spawn(")

    assert preflight_index < tcpstore_index
    assert preflight_index < spawn_index


def test_four_gpu_tcpstore_server_uses_loopback_binding() -> None:
    ht_test = read_repo_file("examples/device/ep/tests/test_ht.py")

    assert 'tcp_store_bind_host = "127.0.0.1"' in ht_test
    assert "run_server, args=(tcp_store_bind_host,)" in ht_test


def test_ht_ci_correctness_allocates_rdma_staging_memory() -> None:
    ht_test = read_repo_file("examples/device/ep/tests/test_ht.py")

    assert "num_rdma_bytes = 0 if args.ci_correctness_only" not in ht_test
    assert "num_rdma_bytes = int(1e9)" in ht_test
    assert "num_rdma_bytes=num_rdma_bytes" in ht_test


def test_ht_ci_correctness_does_not_emit_performance_measurements() -> None:
    ht_test = read_repo_file("examples/device/ep/tests/test_ht.py")
    test_main_body = _function_body(
        ht_test,
        "test_main(\n    args: argparse.Namespace,\n    num_sms: int,\n    local_rank: int,\n    num_local_ranks: int,\n    num_ranks: int,\n    num_nodes: int,\n    rank: int,\n    buffer: nixl_ep.Buffer,\n    group: dist.ProcessGroup,\n):",
    )
    ci_skip_index = test_main_body.find(
        "[ci] Completed reduced-topology correctness checks; skipping performance tuning."
    )
    assert ci_skip_index != -1

    before_ci_skip = test_main_body[:ci_skip_index]
    assert re.search(
        r"if not args\.ci_correctness_only:\s+"
        r"t = bench\(lambda: buffer\.get_dispatch_layout\(topk_idx, num_experts\)\)\[0\]",
        before_ci_skip,
        re.S,
    )
