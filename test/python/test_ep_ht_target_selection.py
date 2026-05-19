# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from argparse import Namespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EP_TESTS_DIR = REPO_ROOT / "examples" / "device" / "ep" / "tests"
sys.path.insert(0, str(EP_TESTS_DIR))

from ht_target_selection import (  # noqa: E402
    CorrectnessPreflightError,
    EIGHT_GPU_SINGLE_NODE_TARGET,
    FOUR_GPU_SINGLE_NODE_TARGET,
    MULTI_NODE_TARGET,
    TargetSelectionError,
    classify_correctness_failure,
    preflight_four_gpu_single_node_target,
    resolve_correctness_target,
)


LOCAL_ENV = {
    "MASTER_ADDR": "127.0.0.1",
    "WORLD_SIZE": "1",
    "RANK": "0",
}


def _args(**overrides: object) -> Namespace:
    values = {
        "correctness_target": None,
        "ci_correctness_only": False,
        "num_processes": 8,
        "tcp_server": None,
    }
    values.update(overrides)
    return Namespace(**values)


class FakeCuda:
    def __init__(
        self,
        *,
        device_count: int = 4,
        unavailable: bool = False,
        denied_peers: set[tuple[int, int]] | None = None,
        context_failure_device: int | None = None,
    ) -> None:
        self._device_count = device_count
        self._unavailable = unavailable
        self._denied_peers = denied_peers or set()
        self._context_failure_device = context_failure_device
        self.context_devices: list[int] = []
        self.peer_checks: list[tuple[int, int]] = []

    def is_available(self) -> bool:
        return not self._unavailable

    def device_count(self) -> int:
        return self._device_count

    def set_device(self, device: int) -> None:
        if device == self._context_failure_device:
            raise RuntimeError("context unavailable")
        self.context_devices.append(device)

    def get_device_properties(self, device: int) -> object:
        if device == self._context_failure_device:
            raise RuntimeError("properties unavailable")
        return object()

    def can_device_access_peer(self, src: int, dst: int) -> bool:
        self.peer_checks.append((src, dst))
        return (src, dst) not in self._denied_peers


class FakeNixlEp:
    class Buffer:
        pass

    class Config:
        pass

    topk_idx_t = object()


class FakeStoreGroup:
    @staticmethod
    def create_master_store() -> object:
        return object()

    @staticmethod
    def create_client_store() -> object:
        return object()


def _preflight(
    *,
    cuda: FakeCuda | None = None,
    bind_checker=lambda host, port: None,
    nixl_ep_module: object = FakeNixlEp,
    store_group_module: object = FakeStoreGroup,
    args: Namespace | None = None,
):
    return preflight_four_gpu_single_node_target(
        _args(
            correctness_target=FOUR_GPU_SINGLE_NODE_TARGET,
            num_processes=4,
        )
        if args is None
        else args,
        env=LOCAL_ENV,
        cuda=cuda or FakeCuda(),
        nixl_ep_module=nixl_ep_module,
        store_group_module=store_group_module,
        tcp_store_port=9999,
        bind_checker=bind_checker,
    )


def _preflight_error(**kwargs: object) -> CorrectnessPreflightError:
    with pytest.raises(CorrectnessPreflightError) as exc_info:
        _preflight(**kwargs)
    return exc_info.value


def test_named_four_gpu_target_selects_reduced_correctness_mode() -> None:
    target = resolve_correctness_target(
        _args(correctness_target=FOUR_GPU_SINGLE_NODE_TARGET, num_processes=4),
        env=LOCAL_ENV,
    )

    assert target.name == FOUR_GPU_SINGLE_NODE_TARGET
    assert target.label == "four-GPU single-node HT correctness"
    assert target.expected_num_processes == 4
    assert target.expected_num_nodes == 1
    assert target.local_only is True
    assert target.ci_correctness_only is True


def test_four_gpu_preflight_accepts_exact_local_topology_before_spawn() -> None:
    cuda = FakeCuda()

    result = _preflight(cuda=cuda)

    assert result is not None
    assert result.target == FOUR_GPU_SINGLE_NODE_TARGET
    assert result.visible_cuda_devices == 4
    assert result.rank_device_map == ((0, 0), (1, 1), (2, 2), (3, 3))
    assert result.num_nvl_ranks == 4
    assert result.num_rdma_ranks == 1
    assert result.tcp_store_host == "127.0.0.1"
    assert result.tcp_store_port == 9999
    assert cuda.context_devices == [0, 1, 2, 3]
    assert len(cuda.peer_checks) == 12
    assert (0, 1) in cuda.peer_checks
    assert (3, 2) in cuda.peer_checks


@pytest.mark.parametrize("device_count", [0, 3, 5])
def test_four_gpu_preflight_requires_exactly_four_visible_cuda_devices(
    device_count: int,
) -> None:
    exc = _preflight_error(cuda=FakeCuda(device_count=device_count))

    assert exc.failure_category == "gpu_unavailable"
    assert "exactly 4 visible CUDA devices" in str(exc)


def test_four_gpu_preflight_rejects_unavailable_cuda_runtime() -> None:
    exc = _preflight_error(cuda=FakeCuda(unavailable=True))

    assert exc.failure_category == "gpu_unavailable"
    assert "CUDA runtime is unavailable" in str(exc)


def test_four_gpu_preflight_checks_cuda_context_readiness() -> None:
    exc = _preflight_error(cuda=FakeCuda(context_failure_device=2))

    assert exc.failure_category == "runtime_not_ready"
    assert "CUDA context readiness failed for device 2" in str(exc)


def test_four_gpu_preflight_checks_peer_access_for_every_active_pair() -> None:
    exc = _preflight_error(cuda=FakeCuda(denied_peers={(1, 2)}))

    assert exc.failure_category == "peer_wiring_failed"
    assert "CUDA IPC/NVLink peer access unavailable between devices 1 and 2" in str(exc)


def test_four_gpu_preflight_requires_peer_access_probe() -> None:
    class NoPeerProbe(FakeCuda):
        can_device_access_peer = None  # type: ignore[assignment]

    exc = _preflight_error(cuda=NoPeerProbe())

    assert exc.failure_category == "runtime_not_ready"
    assert "peer access probe is unavailable" in str(exc)


def test_four_gpu_preflight_checks_local_tcpstore_binding() -> None:
    def fail_bind(host: str, port: int) -> None:
        raise OSError("port unavailable")

    exc = _preflight_error(bind_checker=fail_bind)

    assert exc.failure_category == "runtime_not_ready"
    assert "local TCPStore bind failed" in str(exc)
    assert "9999" not in str(exc)


def test_four_gpu_preflight_rechecks_topology_agreement() -> None:
    exc = _preflight_error(
        args=_args(correctness_target=FOUR_GPU_SINGLE_NODE_TARGET, num_processes=8)
    )

    assert exc.failure_category == "unsupported_target"
    assert "--num-processes 4" in str(exc)


def test_four_gpu_preflight_requires_runtime_import_contract() -> None:
    class MissingBuffer:
        class Config:
            pass

        topk_idx_t = object()

    exc = _preflight_error(nixl_ep_module=MissingBuffer)

    assert exc.failure_category == "runtime_not_ready"
    assert "NIXL runtime import is missing Buffer" in str(exc)


def test_four_gpu_preflight_is_noop_for_other_targets() -> None:
    assert (
        preflight_four_gpu_single_node_target(
            _args(correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET, num_processes=8),
            env=LOCAL_ENV,
            cuda=None,
            nixl_ep_module=None,
            store_group_module=None,
            bind_checker=lambda host, port: None,
        )
        is None
    )


def test_preflight_failures_preserve_reviewable_failure_categories() -> None:
    exc = CorrectnessPreflightError(
        "peer_wiring_failed",
        "CUDA IPC/NVLink peer access unavailable",
    )

    assert classify_correctness_failure(exc) == "peer_wiring_failed"


def test_default_runner_path_is_not_reclassified_as_ci_correctness() -> None:
    target = resolve_correctness_target(_args(), env=LOCAL_ENV)

    assert target.name is None
    assert target.ci_correctness_only is False


def test_named_eight_gpu_target_selects_established_local_compatibility() -> None:
    target = resolve_correctness_target(
        _args(correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET, num_processes=8),
        env=LOCAL_ENV,
    )

    assert target.name == EIGHT_GPU_SINGLE_NODE_TARGET
    assert target.label == "eight-GPU single-node HT compatibility"
    assert target.expected_num_processes == 8
    assert target.expected_num_nodes == 1
    assert target.local_only is True
    assert target.ci_correctness_only is False


def test_named_multi_node_target_selects_established_distributed_compatibility() -> None:
    target = resolve_correctness_target(
        _args(correctness_target=MULTI_NODE_TARGET, num_processes=8),
        env={**LOCAL_ENV, "WORLD_SIZE": "2", "RANK": "1", "MASTER_ADDR": "10.0.0.1"},
    )

    assert target.name == MULTI_NODE_TARGET
    assert target.label == "multi-node HT compatibility"
    assert target.expected_num_processes == 8
    assert target.expected_num_nodes is None
    assert target.local_only is False
    assert target.ci_correctness_only is False


def test_legacy_ci_flag_without_named_target_is_ambiguous() -> None:
    with pytest.raises(TargetSelectionError, match="ambiguous_target"):
        resolve_correctness_target(
            _args(ci_correctness_only=True, num_processes=4),
            env=LOCAL_ENV,
        )


def test_unsupported_target_name_is_rejected() -> None:
    with pytest.raises(TargetSelectionError, match="unsupported_target"):
        resolve_correctness_target(
            _args(correctness_target="four_gpu", num_processes=4),
            env=LOCAL_ENV,
        )


def test_four_gpu_target_requires_exactly_four_processes() -> None:
    with pytest.raises(TargetSelectionError, match="--num-processes 4"):
        resolve_correctness_target(
            _args(correctness_target=FOUR_GPU_SINGLE_NODE_TARGET, num_processes=8),
            env=LOCAL_ENV,
        )


def test_four_gpu_target_rejects_multi_node_environment() -> None:
    with pytest.raises(TargetSelectionError, match="WORLD_SIZE=1"):
        resolve_correctness_target(
            _args(correctness_target=FOUR_GPU_SINGLE_NODE_TARGET, num_processes=4),
            env={**LOCAL_ENV, "WORLD_SIZE": "2"},
        )


def test_four_gpu_target_rejects_off_node_store_endpoint() -> None:
    with pytest.raises(TargetSelectionError, match="local-only"):
        resolve_correctness_target(
            _args(
                correctness_target=FOUR_GPU_SINGLE_NODE_TARGET,
                num_processes=4,
                tcp_server="192.0.2.42",
            ),
            env=LOCAL_ENV,
        )


@pytest.mark.parametrize("master_addr", ["0.0.0.0", "::"])
def test_four_gpu_target_rejects_unspecified_master_addr(master_addr: str) -> None:
    with pytest.raises(TargetSelectionError, match="local"):
        resolve_correctness_target(
            _args(correctness_target=FOUR_GPU_SINGLE_NODE_TARGET, num_processes=4),
            env={**LOCAL_ENV, "MASTER_ADDR": master_addr},
        )


@pytest.mark.parametrize("tcp_server", ["0.0.0.0", "::"])
def test_four_gpu_target_rejects_unspecified_tcp_store_endpoint(
    tcp_server: str,
) -> None:
    with pytest.raises(TargetSelectionError, match="local-only"):
        resolve_correctness_target(
            _args(
                correctness_target=FOUR_GPU_SINGLE_NODE_TARGET,
                num_processes=4,
                tcp_server=tcp_server,
            ),
            env=LOCAL_ENV,
        )


def test_eight_gpu_target_requires_exactly_eight_processes() -> None:
    with pytest.raises(TargetSelectionError, match="--num-processes 8"):
        resolve_correctness_target(
            _args(correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET, num_processes=4),
            env=LOCAL_ENV,
        )


def test_eight_gpu_target_rejects_multi_node_environment() -> None:
    with pytest.raises(TargetSelectionError, match="WORLD_SIZE=1"):
        resolve_correctness_target(
            _args(correctness_target=EIGHT_GPU_SINGLE_NODE_TARGET, num_processes=8),
            env={**LOCAL_ENV, "WORLD_SIZE": "2"},
        )


def test_multi_node_target_requires_more_than_one_node() -> None:
    with pytest.raises(TargetSelectionError, match="WORLD_SIZE greater than 1"):
        resolve_correctness_target(
            _args(correctness_target=MULTI_NODE_TARGET, num_processes=8),
            env=LOCAL_ENV,
        )
