# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_dl_gpu_flow_builds_nixl_ep() -> None:
    dl_matrix = read_repo_file(".ci/jenkins/lib/test-dl-matrix.yaml")
    gpu_dockerfile = read_repo_file(".ci/dockerfiles/Dockerfile.gpu-test")

    assert "--build-arg BUILD_NIXL_EP=true" in dl_matrix
    assert "ARG BUILD_NIXL_EP=false" in gpu_dockerfile
    assert "-Dbuild_nixl_ep=true" in gpu_dockerfile
    assert ".gitlab/build.sh ${NIXL_INSTALL_DIR}" in gpu_dockerfile


def test_dl_gpu_flow_has_dedicated_ep_ht_correctness_step() -> None:
    dl_matrix = read_repo_file(".ci/jenkins/lib/test-dl-matrix.yaml")

    assert "Run DL NIXL EP HT correctness test" in dl_matrix
    assert "bash .ci/scripts/run_ep_ht_correctness_ci.sh ${NIXL_INSTALL_DIR}" in dl_matrix


def test_ep_ht_ci_entrypoint_limits_scope_to_four_gpu_correctness() -> None:
    entrypoint = read_repo_file(".ci/scripts/run_ep_ht_correctness_ci.sh")

    assert "NIXL EP HT 4-GPU correctness" in entrypoint
    assert "--num-processes \"${NIXL_EP_HT_CI_NUM_PROCESSES:-4}\"" in entrypoint
    assert "--ci-correctness-only" in entrypoint
    assert "This check does not cover multi-node RDMA or performance." in entrypoint


def test_ht_test_keeps_default_full_scale_path_and_adds_ci_mode() -> None:
    ht_test = read_repo_file("examples/device/ep/tests/test_ht.py")

    assert "--ci-correctness-only" in ht_test
    assert "default=8" in ht_test
    assert "args.ci_correctness_only" in ht_test
    assert "num_local_ranks == 8" in ht_test
