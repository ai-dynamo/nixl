# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import ipaddress
import os
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping


FOUR_GPU_SINGLE_NODE_TARGET = "four_gpu_single_node"
EIGHT_GPU_SINGLE_NODE_TARGET = "eight_gpu_single_node"
MULTI_NODE_TARGET = "multi_node"
EVIDENCE_CORRECTNESS_TARGETS = frozenset(
    {
        FOUR_GPU_SINGLE_NODE_TARGET,
        EIGHT_GPU_SINGLE_NODE_TARGET,
        MULTI_NODE_TARGET,
    }
)
REVIEWABLE_FAILURE_CATEGORIES = frozenset(
    {
        "configuration_invalid",
        "correctness_failed",
        "evidence_incomplete",
        "gpu_unavailable",
        "layout_metadata_failed",
        "peer_wiring_failed",
        "runtime_not_ready",
        "topology_mismatch",
        "unsupported_target",
    }
)
BLOCKED_COMPATIBILITY_FAILURE_CATEGORIES = frozenset(
    {
        "configuration_invalid",
        "evidence_incomplete",
        "gpu_unavailable",
        "runtime_not_ready",
        "unsupported_target",
    }
)


@dataclass(frozen=True)
class CorrectnessTarget:
    name: str | None
    label: str | None
    expected_num_processes: int | None
    expected_num_nodes: int | None
    min_num_nodes: int | None
    local_only: bool
    ci_correctness_only: bool


SUPPORTED_CORRECTNESS_TARGETS = {
    FOUR_GPU_SINGLE_NODE_TARGET: CorrectnessTarget(
        name=FOUR_GPU_SINGLE_NODE_TARGET,
        label="four-GPU single-node HT correctness",
        expected_num_processes=4,
        expected_num_nodes=1,
        min_num_nodes=None,
        local_only=True,
        ci_correctness_only=True,
    ),
    EIGHT_GPU_SINGLE_NODE_TARGET: CorrectnessTarget(
        name=EIGHT_GPU_SINGLE_NODE_TARGET,
        label="eight-GPU single-node HT compatibility",
        expected_num_processes=8,
        expected_num_nodes=1,
        min_num_nodes=None,
        local_only=True,
        ci_correctness_only=False,
    ),
    MULTI_NODE_TARGET: CorrectnessTarget(
        name=MULTI_NODE_TARGET,
        label="multi-node HT compatibility",
        expected_num_processes=8,
        expected_num_nodes=None,
        min_num_nodes=2,
        local_only=False,
        ci_correctness_only=False,
    ),
}


DEFAULT_CORRECTNESS_TARGET = CorrectnessTarget(
    name=None,
    label=None,
    expected_num_processes=None,
    expected_num_nodes=None,
    min_num_nodes=None,
    local_only=False,
    ci_correctness_only=False,
)


class TargetSelectionError(ValueError):
    pass


class CorrectnessPreflightError(RuntimeError):
    def __init__(self, failure_category: str, message: str):
        super().__init__(message)
        self.failure_category = _reviewable_failure_category(failure_category)


@dataclass(frozen=True)
class FinalReviewEvidenceDecision:
    accepted: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class FourGpuSingleNodePreflight:
    target: str
    visible_cuda_devices: int
    rank_device_map: tuple[tuple[int, int], ...]
    num_nvl_ranks: int
    num_rdma_ranks: int
    tcp_store_host: str
    tcp_store_port: int


FOUR_GPU_EVIDENCE_ID = "ep_ht_four_gpu_single_node"
EIGHT_GPU_COMPATIBILITY_EVIDENCE_ID = "ep_ht_eight_gpu_single_node_compatibility"
MULTI_NODE_COMPATIBILITY_EVIDENCE_ID = "ep_ht_multi_node_compatibility"
ESTABLISHED_COMPATIBILITY_NVL_RANKS = 8
FOUR_GPU_SINGLE_NODE_TCPSTORE_HOST = "127.0.0.1"

_FOUR_GPU_EVIDENCE_FIELDS = {
    "target": FOUR_GPU_SINGLE_NODE_TARGET,
    "topology": "single_node",
    "world_size": "4",
    "num_nvl_ranks": "4",
    "num_rdma_ranks": "1",
    "correctness_only": "true",
    "scope": "single_node_correctness_only_no_multi_node_rdma_no_performance",
}
_EIGHT_GPU_COMPATIBILITY_EVIDENCE_FIELDS = {
    "target": EIGHT_GPU_SINGLE_NODE_TARGET,
    "topology": "single_node",
    "world_size": "8",
    "num_nvl_ranks": "8",
    "num_rdma_ranks": "1",
    "correctness_only": "false",
    "scope": "established_eight_gpu_single_node_ht_compatibility",
}
_MULTI_NODE_COMPATIBILITY_EVIDENCE_FIELDS = {
    "target": MULTI_NODE_TARGET,
    "topology": "multi_node",
    "num_nvl_ranks": "8",
    "correctness_only": "false",
    "scope": "established_multi_node_ht_compatibility",
}


def format_correctness_evidence(
    args: object,
    *,
    result: str,
    failure_category: str | None = None,
) -> str:
    if result not in {"pass", "fail"}:
        raise ValueError(f"unsupported evidence result: {result!r}")

    target = str(getattr(args, "correctness_target", "") or "default")
    target_label = str(getattr(args, "correctness_target_label", "") or target)
    is_four_gpu = target == FOUR_GPU_SINGLE_NODE_TARGET
    is_eight_gpu = target == EIGHT_GPU_SINGLE_NODE_TARGET
    is_multi_node = target == MULTI_NODE_TARGET
    status = result
    failure = (
        "none"
        if result == "pass"
        else _reviewable_failure_category(failure_category)
    )
    num_processes = _optional_int(getattr(args, "num_processes", None))
    num_nodes = _optional_int(getattr(args, "num_nodes", os.getenv("WORLD_SIZE", "1")))
    if is_four_gpu:
        topology = "single_node"
        world_size = 4
        num_nvl_ranks = 4
        num_rdma_ranks = 1
        evidence_id = FOUR_GPU_EVIDENCE_ID
        scope = "single_node_correctness_only_no_multi_node_rdma_no_performance"
    elif is_eight_gpu:
        topology = "single_node"
        world_size = 8
        num_nvl_ranks = 8
        num_rdma_ranks = 1
        evidence_id = EIGHT_GPU_COMPATIBILITY_EVIDENCE_ID
        scope = "established_eight_gpu_single_node_ht_compatibility"
    elif is_multi_node:
        topology = "multi_node"
        world_size = (
            num_processes * num_nodes
            if num_processes is not None and num_nodes is not None
            else "unspecified"
        )
        num_nvl_ranks = 8
        num_rdma_ranks = num_nodes if num_nodes is not None else "unspecified"
        evidence_id = MULTI_NODE_COMPATIBILITY_EVIDENCE_ID
        scope = "established_multi_node_ht_compatibility"
    else:
        topology = "unspecified"
        world_size = getattr(args, "num_processes", "unspecified")
        num_nvl_ranks = "unspecified"
        num_rdma_ranks = "unspecified"
        evidence_id = "ep_ht"
        scope = "unspecified"
    workload_tokens = _evidence_value(getattr(args, "num_tokens", "unspecified"))
    hidden_size = _evidence_value(getattr(args, "hidden", "unspecified"))
    top_k = _evidence_value(getattr(args, "num_topk", "unspecified"))
    expert_count = _evidence_value(getattr(args, "num_experts", "unspecified"))
    workload = (
        f"tokens:{workload_tokens},hidden:{hidden_size},"
        f"top_k:{top_k},experts:{expert_count}"
    )

    fields = (
        ("schema", "ep_ht_correctness_evidence_v1"),
        ("evidence_id", evidence_id),
        ("target", _evidence_value(target)),
        ("target_label", _evidence_value(target_label)),
        ("topology", topology),
        ("world_size", world_size),
        ("num_nvl_ranks", num_nvl_ranks),
        ("num_rdma_ranks", num_rdma_ranks),
        ("workload", workload),
        ("workload_tokens", workload_tokens),
        ("hidden_size", hidden_size),
        ("top_k", top_k),
        ("expert_count", expert_count),
        (
            "correctness_only",
            str(bool(getattr(args, "ci_correctness_only", False))).lower(),
        ),
        ("scope", scope),
        ("result", result),
        ("status", status),
        ("failure_category", _evidence_value(failure)),
    )
    return "[evidence] " + " ".join(f"{key}={value}" for key, value in fields)


def validate_final_review_evidence(
    records: list[str] | tuple[str, ...],
    *,
    expected_compatibility: Mapping[str, tuple[str, str]],
) -> FinalReviewEvidenceDecision:
    evidence_by_id: dict[str, list[dict[str, str]]] = {}
    for record in records:
        fields = _parse_evidence_record(record)
        if fields.get("schema") != "ep_ht_correctness_evidence_v1":
            continue
        evidence_by_id.setdefault(fields.get("evidence_id", "missing"), []).append(
            fields
        )

    blockers: list[str] = []
    _require_exact_evidence(
        evidence_by_id,
        blockers,
        evidence_id=FOUR_GPU_EVIDENCE_ID,
        expected_result="pass",
        expected_failure_category="none",
        expected_fields=_FOUR_GPU_EVIDENCE_FIELDS,
    )
    _require_compatibility_evidence(
        evidence_by_id,
        blockers,
        evidence_id=EIGHT_GPU_COMPATIBILITY_EVIDENCE_ID,
        target=EIGHT_GPU_SINGLE_NODE_TARGET,
        expected_fields=_EIGHT_GPU_COMPATIBILITY_EVIDENCE_FIELDS,
        expected_compatibility=expected_compatibility,
    )
    _require_compatibility_evidence(
        evidence_by_id,
        blockers,
        evidence_id=MULTI_NODE_COMPATIBILITY_EVIDENCE_ID,
        target=MULTI_NODE_TARGET,
        expected_fields=_MULTI_NODE_COMPATIBILITY_EVIDENCE_FIELDS,
        expected_compatibility=expected_compatibility,
    )
    return FinalReviewEvidenceDecision(
        accepted=not blockers,
        blockers=tuple(blockers),
    )


def classify_correctness_failure(exc: BaseException) -> str:
    if isinstance(exc, CorrectnessPreflightError):
        return exc.failure_category

    if isinstance(exc, TargetSelectionError):
        return "unsupported_target"

    message = str(exc).lower()
    if any(term in message for term in ("cuda ipc", "nvlink", "peer", "ipc")):
        return "peer_wiring_failed"
    if any(
        term in message
        for term in ("no cuda", "cuda unavailable", "device count", "gpu")
    ):
        return "gpu_unavailable"
    if any(term in message for term in ("nixl", "nccl", "runtime", "c10")):
        return "runtime_not_ready"
    if any(term in message for term in ("assert", "allclose", "mismatch", "timeout")):
        return "correctness_failed"
    return "correctness_failed"


def _parse_evidence_record(record: str) -> dict[str, str]:
    if not record.startswith("[evidence] "):
        return {}
    return dict(
        field.split("=", 1)
        for field in record.removeprefix("[evidence] ").split()
        if "=" in field
    )


def _require_exact_evidence(
    evidence_by_id: Mapping[str, list[dict[str, str]]],
    blockers: list[str],
    *,
    evidence_id: str,
    expected_result: str,
    expected_failure_category: str,
    expected_fields: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    matches = evidence_by_id.get(evidence_id, [])
    if not matches:
        blockers.append(f"missing_{evidence_id}")
        return None
    if len(matches) != 1:
        blockers.append(f"ambiguous_{evidence_id}")
        return None

    fields = matches[0]
    actual_result = fields.get("result", "missing")
    actual_failure_category = fields.get("failure_category", "missing")
    if (
        actual_result != expected_result
        or actual_failure_category != expected_failure_category
    ):
        blockers.append(
            f"changed_{evidence_id}_expected_"
            f"{expected_result}_{expected_failure_category}_got_"
            f"{actual_result}_{actual_failure_category}"
        )
    if expected_fields is not None:
        _require_evidence_fields(
            fields,
            blockers,
            evidence_id=evidence_id,
            expected_fields=expected_fields,
        )
    return fields


def _require_compatibility_evidence(
    evidence_by_id: Mapping[str, list[dict[str, str]]],
    blockers: list[str],
    *,
    evidence_id: str,
    target: str,
    expected_fields: Mapping[str, str],
    expected_compatibility: Mapping[str, tuple[str, str]],
) -> None:
    expected = expected_compatibility.get(target)
    if expected is None:
        blockers.append(f"missing_expected_compatibility_{target}")
        return

    fields = _require_exact_evidence(
        evidence_by_id,
        blockers,
        evidence_id=evidence_id,
        expected_result=expected[0],
        expected_failure_category=expected[1],
        expected_fields=expected_fields,
    )
    if fields is None:
        return

    failure_category = fields.get("failure_category", "missing")
    if (
        fields.get("status") == "blocked"
        or failure_category in BLOCKED_COMPATIBILITY_FAILURE_CATEGORIES
    ):
        blockers.append(f"blocked_{evidence_id}_{failure_category}")

    if target == MULTI_NODE_TARGET:
        _require_multi_node_evidence_shape(fields, blockers, evidence_id=evidence_id)


def _require_evidence_fields(
    fields: Mapping[str, str],
    blockers: list[str],
    *,
    evidence_id: str,
    expected_fields: Mapping[str, str],
) -> None:
    for field, expected in expected_fields.items():
        actual = fields.get(field, "missing")
        if actual != expected:
            blockers.append(
                f"mismatched_{evidence_id}_{field}_expected_{expected}_got_{actual}"
            )


def _require_multi_node_evidence_shape(
    fields: Mapping[str, str], blockers: list[str], *, evidence_id: str
) -> None:
    num_rdma_ranks = _parse_evidence_int(
        fields,
        blockers,
        evidence_id=evidence_id,
        field="num_rdma_ranks",
        expected="gt_1",
    )
    world_size = _parse_evidence_int(
        fields,
        blockers,
        evidence_id=evidence_id,
        field="world_size",
        expected="gt_8",
    )
    num_nvl_ranks = _parse_evidence_int(
        fields,
        blockers,
        evidence_id=evidence_id,
        field="num_nvl_ranks",
        expected="integer",
    )
    if num_rdma_ranks is None or world_size is None or num_nvl_ranks is None:
        return

    if num_rdma_ranks <= 1:
        blockers.append(
            f"mismatched_{evidence_id}_num_rdma_ranks_expected_gt_1_got_"
            f"{num_rdma_ranks}"
        )
    if world_size <= ESTABLISHED_COMPATIBILITY_NVL_RANKS:
        blockers.append(
            f"mismatched_{evidence_id}_world_size_expected_gt_8_got_{world_size}"
        )
    expected_world_size = num_nvl_ranks * num_rdma_ranks
    if world_size != expected_world_size:
        blockers.append(
            f"mismatched_{evidence_id}_world_size_expected_"
            f"{expected_world_size}_got_{world_size}"
        )


def _parse_evidence_int(
    fields: Mapping[str, str],
    blockers: list[str],
    *,
    evidence_id: str,
    field: str,
    expected: str,
) -> int | None:
    actual = fields.get(field, "missing")
    try:
        return int(actual)
    except ValueError:
        blockers.append(
            f"mismatched_{evidence_id}_{field}_expected_{expected}_got_{actual}"
        )
        return None


def _evidence_value(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return normalized or "unspecified"


def _reviewable_failure_category(failure_category: object | None) -> str:
    if failure_category is None:
        return "correctness_failed"

    normalized = _evidence_value(failure_category)
    if normalized in REVIEWABLE_FAILURE_CATEGORIES:
        return normalized
    return "correctness_failed"


def _optional_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def resolve_correctness_target(
    args: object, env: Mapping[str, str] | None = None
) -> CorrectnessTarget:
    env = os.environ if env is None else env
    selected = getattr(args, "correctness_target", None)
    legacy_ci_flag = bool(getattr(args, "ci_correctness_only", False))

    if legacy_ci_flag and not selected:
        raise TargetSelectionError(
            "ambiguous_target: --ci-correctness-only is not a named target; "
            f"use --correctness-target {FOUR_GPU_SINGLE_NODE_TARGET}"
        )

    if selected is None:
        return DEFAULT_CORRECTNESS_TARGET

    try:
        target = SUPPORTED_CORRECTNESS_TARGETS[selected]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_CORRECTNESS_TARGETS))
        raise TargetSelectionError(
            f"unsupported_target: unsupported correctness target {selected!r}; "
            f"supported targets: {supported}"
        ) from exc

    _validate_num_processes(args, target)
    if target.local_only:
        _validate_local_only_target(args, env, target)
    _validate_min_num_nodes(env, target)
    return target


def preflight_four_gpu_single_node_target(
    args: object,
    *,
    env: Mapping[str, str] | None = None,
    cuda: object | None = None,
    nixl_ep_module: object | None = None,
    store_group_module: object | None = None,
    tcp_store_port: int = 9999,
    tcp_store_host: str = FOUR_GPU_SINGLE_NODE_TCPSTORE_HOST,
    bind_checker: Callable[[str, int], None] | None = None,
) -> FourGpuSingleNodePreflight | None:
    if getattr(args, "correctness_target", None) != FOUR_GPU_SINGLE_NODE_TARGET:
        return None

    env = os.environ if env is None else env
    try:
        target = resolve_correctness_target(args, env=env)
    except TargetSelectionError as exc:
        raise CorrectnessPreflightError("unsupported_target", str(exc)) from exc

    expected_devices = target.expected_num_processes
    if expected_devices != 4:
        raise CorrectnessPreflightError(
            "unsupported_target",
            f"unsupported_target: {FOUR_GPU_SINGLE_NODE_TARGET} topology is not four ranks",
        )

    cuda = _resolve_cuda_probe(cuda)
    nixl_ep_module = _resolve_runtime_module(nixl_ep_module, "nixl_ep", "NIXL runtime")
    store_group_module = _resolve_runtime_module(
        store_group_module, "store_group", "TCPStore runtime"
    )
    _require_runtime_attrs(
        nixl_ep_module, "NIXL runtime", ("Buffer", "Config", "topk_idx_t")
    )
    _require_runtime_attrs(
        store_group_module,
        "TCPStore runtime",
        ("create_master_store", "create_client_store"),
    )

    visible_cuda_devices = _check_visible_cuda_devices(cuda, expected_devices)
    rank_device_map = tuple((rank, rank) for rank in range(expected_devices))
    device_ids = tuple(device for _, device in rank_device_map)
    _check_cuda_contexts(cuda, device_ids)
    _check_cuda_peer_access(cuda, device_ids)

    selected_tcp_store_host = _select_tcp_store_host(args, tcp_store_host)
    if not _is_local_endpoint(selected_tcp_store_host):
        raise CorrectnessPreflightError(
            "unsupported_target",
            f"unsupported_target: {FOUR_GPU_SINGLE_NODE_TARGET} requires a local TCPStore endpoint",
        )
    if not getattr(args, "tcp_server", None):
        _check_local_tcpstore_bind(
            selected_tcp_store_host,
            tcp_store_port,
            bind_checker=bind_checker,
        )

    return FourGpuSingleNodePreflight(
        target=FOUR_GPU_SINGLE_NODE_TARGET,
        visible_cuda_devices=visible_cuda_devices,
        rank_device_map=rank_device_map,
        num_nvl_ranks=4,
        num_rdma_ranks=1,
        tcp_store_host=selected_tcp_store_host,
        tcp_store_port=int(tcp_store_port),
    )


def _resolve_cuda_probe(cuda: object | None) -> object:
    if cuda is not None:
        return cuda

    try:
        torch_module = importlib.import_module("torch")
    except Exception as exc:
        raise CorrectnessPreflightError(
            "runtime_not_ready", "runtime_not_ready: CUDA runtime import failed"
        ) from exc
    return torch_module.cuda


def _resolve_runtime_module(
    module: object | None, module_name: str, display_name: str
) -> object:
    if module is not None:
        return module

    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise CorrectnessPreflightError(
            "runtime_not_ready",
            f"runtime_not_ready: {display_name} import failed",
        ) from exc


def _require_runtime_attrs(
    module: object, display_name: str, required_attrs: tuple[str, ...]
) -> None:
    for attr in required_attrs:
        if not hasattr(module, attr):
            raise CorrectnessPreflightError(
                "runtime_not_ready",
                f"runtime_not_ready: {display_name} import is missing {attr}",
            )


def _check_visible_cuda_devices(cuda: object, expected_devices: int) -> int:
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        raise CorrectnessPreflightError(
            "gpu_unavailable", "gpu_unavailable: CUDA runtime is unavailable"
        )

    device_count = getattr(cuda, "device_count", None)
    if not callable(device_count):
        raise CorrectnessPreflightError(
            "runtime_not_ready",
            "runtime_not_ready: CUDA device-count probe is unavailable",
        )

    visible_cuda_devices = int(device_count())
    if visible_cuda_devices != expected_devices:
        raise CorrectnessPreflightError(
            "gpu_unavailable",
            f"gpu_unavailable: {FOUR_GPU_SINGLE_NODE_TARGET} requires exactly "
            f"{expected_devices} visible CUDA devices; got {visible_cuda_devices}",
        )
    return visible_cuda_devices


def _check_cuda_contexts(cuda: object, device_ids: tuple[int, ...]) -> None:
    set_device = getattr(cuda, "set_device", None)
    get_device_properties = getattr(cuda, "get_device_properties", None)
    if not callable(set_device) or not callable(get_device_properties):
        raise CorrectnessPreflightError(
            "runtime_not_ready",
            "runtime_not_ready: CUDA context readiness probe is unavailable",
        )

    current_device = getattr(cuda, "current_device", None)
    previous_device: int | None = None
    if callable(current_device):
        try:
            previous_device = int(current_device())
        except Exception:
            previous_device = None

    try:
        for device in device_ids:
            try:
                set_device(device)
                get_device_properties(device)
            except Exception as exc:
                raise CorrectnessPreflightError(
                    "runtime_not_ready",
                    f"runtime_not_ready: CUDA context readiness failed for device {device}",
                ) from exc
    finally:
        if previous_device is not None:
            try:
                set_device(previous_device)
            except Exception:
                pass


def _check_cuda_peer_access(cuda: object, device_ids: tuple[int, ...]) -> None:
    can_device_access_peer = getattr(cuda, "can_device_access_peer", None)
    if not callable(can_device_access_peer):
        raise CorrectnessPreflightError(
            "runtime_not_ready",
            "runtime_not_ready: CUDA IPC/NVLink peer access probe is unavailable",
        )

    for src in device_ids:
        for dst in device_ids:
            if src == dst:
                continue
            try:
                can_access = bool(can_device_access_peer(src, dst))
            except Exception as exc:
                raise CorrectnessPreflightError(
                    "peer_wiring_failed",
                    "peer_wiring_failed: CUDA IPC/NVLink peer access probe failed",
                ) from exc
            if not can_access:
                raise CorrectnessPreflightError(
                    "peer_wiring_failed",
                    "peer_wiring_failed: CUDA IPC/NVLink peer access unavailable "
                    f"between devices {src} and {dst}",
                )


def _select_tcp_store_host(args: object, default_host: str) -> str:
    tcp_server = getattr(args, "tcp_server", None)
    if tcp_server:
        return str(tcp_server)
    return default_host


def _check_local_tcpstore_bind(
    host: str,
    port: int,
    *,
    bind_checker: Callable[[str, int], None] | None,
) -> None:
    checker = _socket_bind_check if bind_checker is None else bind_checker
    try:
        checker(host, int(port))
    except Exception as exc:
        raise CorrectnessPreflightError(
            "runtime_not_ready",
            "runtime_not_ready: local TCPStore bind failed for configured loopback endpoint",
        ) from exc


def _socket_bind_check(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))


def _validate_num_processes(args: object, target: CorrectnessTarget) -> None:
    expected = target.expected_num_processes
    if expected is None:
        return

    actual = getattr(args, "num_processes", None)
    try:
        actual_int = int(actual)
    except (TypeError, ValueError) as exc:
        raise TargetSelectionError(
            f"unsupported_target: {target.name} requires --num-processes {expected}"
        ) from exc

    if actual_int != expected:
        raise TargetSelectionError(
            f"unsupported_target: {target.name} requires --num-processes {expected}; "
            f"got {actual_int}"
        )


def _validate_local_only_target(
    args: object, env: Mapping[str, str], target: CorrectnessTarget
) -> None:
    expected_nodes = target.expected_num_nodes
    world_size = _read_int_env(env, "WORLD_SIZE", 1)
    rank = _read_int_env(env, "RANK", 0)

    if expected_nodes is not None and world_size != expected_nodes:
        raise TargetSelectionError(
            f"unsupported_target: {target.name} is local-only and requires "
            f"WORLD_SIZE={expected_nodes}; got {world_size}"
        )

    if rank != 0:
        raise TargetSelectionError(
            f"unsupported_target: {target.name} is local-only and requires RANK=0; "
            f"got {rank}"
        )

    master_addr = env.get("MASTER_ADDR", "127.0.0.1")
    if master_addr and not _is_local_endpoint(master_addr):
        raise TargetSelectionError(
            f"unsupported_target: {target.name} is local-only and requires a local "
            f"MASTER_ADDR; got {master_addr!r}"
        )

    tcp_server = getattr(args, "tcp_server", None)
    if tcp_server and not _is_local_endpoint(str(tcp_server)):
        raise TargetSelectionError(
            f"unsupported_target: {target.name} is local-only and rejects off-node "
            f"TCPStore endpoints; got {tcp_server!r}"
        )


def _validate_min_num_nodes(env: Mapping[str, str], target: CorrectnessTarget) -> None:
    if target.min_num_nodes is None:
        return

    world_size = _read_int_env(env, "WORLD_SIZE", 1)
    if world_size < target.min_num_nodes:
        raise TargetSelectionError(
            f"unsupported_target: {target.name} requires WORLD_SIZE greater than 1; "
            f"got {world_size}"
        )


def _read_int_env(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise TargetSelectionError(
            f"unsupported_target: {name} must be an integer; got {value!r}"
        ) from exc


def _is_local_endpoint(endpoint: str) -> bool:
    normalized = endpoint.strip().strip("[]").lower().rstrip(".")
    local_names = {
        "localhost",
        socket.gethostname().lower().rstrip("."),
        socket.getfqdn().lower().rstrip("."),
    }
    if normalized in local_names:
        return True

    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback
