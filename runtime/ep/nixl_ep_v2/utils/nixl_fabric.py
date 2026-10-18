# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This file incorporates material from the DeepSeek project, licensed under the MIT License.
# The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVLink-domain evidence for a constructing cohort: fail-closed for a declared domain size, inferred from pairwise
evidence otherwise. No CUDA/NVML initialization at import time; GPUs are resolved in NVML by UUID."""

import ctypes
import uuid
from pathlib import Path

# Public NVML constant
_NVML_GPU_FABRIC_STATE_COMPLETED = 3


def _gpu_uuid(value):
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if not isinstance(value, str) or not value.startswith("GPU-"):
        raise ValueError(f"Expected a full GPU UUID; MIG is unsupported: {value!r}")
    return "GPU-" + str(uuid.UUID(value[4:]))


def _fabric_info(pynvml, handle):
    # The NVML fabric info of a GPU, or None when the driver or the binding predates the fabric API
    unavailable = (pynvml.NVMLError_NotSupported, pynvml.NVMLError_FunctionNotFound)
    query = getattr(pynvml, "nvmlDeviceGetGpuFabricInfoV", None)
    info_type = getattr(pynvml, "c_nvmlGpuFabricInfo_v2_t", None)
    if query is None or info_type is None:
        return None
    info = info_type()
    try:
        status = query(handle, ctypes.byref(info))
    except unavailable:
        return None
    if status != 0:
        raise ValueError(f"nvmlDeviceGetGpuFabricInfoV failed with status {status}")
    return info


def _fabric_handle_supported(device_idx):
    # The CUDA driver attribute `HANDLE_TYPE_FABRIC_SUPPORTED`, read by the extension (no driver binding of our own)
    from .. import _C

    return bool(_C.is_fabric_handle_supported(device_idx))


def validate_fabric_record(record):
    """Require one completed fabric enrollment."""
    required = {
        "gpu_uuid",
        "cluster_uuid",
        "clique_id",
        "state",
        "status",
        "handle_supported",
    }
    if (
        not isinstance(record, dict)
        or not required.issubset(record)
        or record.get("error")
    ):
        raise ValueError(f"Missing/failed fabric evidence: {record!r}")
    if _gpu_uuid(record["gpu_uuid"]) != record["gpu_uuid"]:
        raise ValueError("Noncanonical GPU UUID in fabric evidence")
    cluster = record["cluster_uuid"]
    if (
        not isinstance(cluster, str)
        or len(cluster) != 32
        or any(char not in "0123456789abcdef" for char in cluster)
        or int(cluster, 16) == 0
    ):
        raise ValueError("Invalid or unset fabric cluster UUID")
    if type(record["clique_id"]) is not int or not 0 <= record["clique_id"] < 2**32:
        raise ValueError("Invalid fabric clique ID")
    if (
        type(record["state"]) is not int
        or record["state"] != _NVML_GPU_FABRIC_STATE_COMPLETED
        or type(record["status"]) is not int
        or record["status"] != 0
        or record["handle_supported"] is not True
    ):
        raise ValueError(
            f"Fabric registration incomplete, failed or unsupported: {record!r}"
        )


def inspect_topology_device():
    """Inspect the current CUDA device: its UUID, the host's boot id, its NVLink P2P peers (NVML) and, when enrolled,
    its fabric clique. Without the NVML fabric API a domain spanning hosts is seen as per-host domains.
    """
    import pynvml
    import torch

    device_idx = torch.cuda.current_device()
    raw_uuid = bytes(torch.cuda.get_device_properties(device_idx).uuid.bytes)
    device_uuid = "GPU-" + str(uuid.UUID(bytes=raw_uuid))
    host_id = str(
        uuid.UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    )
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
        if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
            raise ValueError("MIG is unsupported for NIXL NVLink topology discovery")
        fabric = None
        info = _fabric_info(pynvml, handle)
        if info is not None:
            cluster = bytes(info.clusterUuid)
            # A zero cluster means this GPU is not enrolled in a fabric; a partial enrollment fails closed below
            if any(cluster):
                fabric = dict(
                    gpu_uuid=device_uuid,
                    cluster_uuid=cluster.hex(),
                    clique_id=int(info.cliqueId),
                    state=int(info.state),
                    status=int(info.status),
                    handle_supported=_fabric_handle_supported(device_idx),
                )
        # A completed fabric clique proves the connectivity; local P2P enumeration is the fallback only
        peers = []
        if fabric is None:
            for ordinal in range(pynvml.nvmlDeviceGetCount()):
                # A device this rank cannot query (another job's GPU) is not a peer and must not fail the probe
                try:
                    other = pynvml.nvmlDeviceGetHandleByIndex(ordinal)
                    other_uuid = _gpu_uuid(pynvml.nvmlDeviceGetUUID(other))
                    if other_uuid == device_uuid:
                        continue
                    status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, other, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                except (pynvml.NVMLError, ValueError):
                    continue
                if status == pynvml.NVML_P2P_STATUS_OK:
                    peers.append(other_uuid)
        return dict(
            gpu_uuid=device_uuid,
            host_id=host_id,
            nvlink_peers=sorted(peers),
            fabric=fabric,
        )
    finally:
        pynvml.nvmlShutdown()


def validate_topology_records(records, requested_scaleup=None):
    """Validate declared logical domains (contiguous blocks of ``requested_scaleup`` ranks, pairwise NVLink-linked;
    a block smaller than its CUDA IPC clique passes here and is refused at ``commit_topology``, since the scale-out
    path needs RDMA to every rail peer) or infer uniform, rank-contiguous physical domains. Returns
    ``(num_scaleout_ranks, num_scaleup_ranks)``.
    """
    if not records:
        raise ValueError("Empty topology cohort")
    if requested_scaleup is not None and (
        type(requested_scaleup) is not int
        or requested_scaleup <= 0
        or len(records) % requested_scaleup
    ):
        raise ValueError(
            "Requested scale-up size must be positive and divide the cohort world"
        )
    identities = set()
    for rank, record in enumerate(records):
        if not isinstance(record, dict) or record.get("error"):
            raise ValueError(
                f"Failed topology evidence at cohort rank {rank}: {record!r}"
            )
        required = {"gpu_uuid", "host_id", "nvlink_peers", "fabric"}
        if not required.issubset(record):
            raise ValueError(f"Incomplete topology evidence at cohort rank {rank}")
        identity = _gpu_uuid(record["gpu_uuid"])
        if identity != record["gpu_uuid"] or identity in identities:
            raise ValueError(
                f"Duplicate/noncanonical GPU identity at cohort rank {rank}"
            )
        identities.add(identity)
        if not isinstance(record["host_id"], str) or not record["host_id"]:
            raise ValueError(f"Missing host identity at cohort rank {rank}")
        if not isinstance(record["nvlink_peers"], (list, tuple)):
            raise ValueError(f"Invalid NVLink peer evidence at cohort rank {rank}")
        for peer in record["nvlink_peers"]:
            if _gpu_uuid(peer) != peer:
                raise ValueError("Noncanonical NVLink peer identity")
        if record["fabric"] is not None:
            validate_fabric_record(record["fabric"])
            if record["fabric"]["gpu_uuid"] != identity:
                raise ValueError("Fabric evidence identifies a different GPU")

    def linked(a, b):
        left, right = records[a], records[b]
        lf, rf = left["fabric"], right["fabric"]
        if lf is not None or rf is not None:
            return (
                lf is not None
                and rf is not None
                and (lf["cluster_uuid"], lf["clique_id"])
                == (rf["cluster_uuid"], rf["clique_id"])
            )
        if left["host_id"] != right["host_id"]:
            return False
        forward = right["gpu_uuid"] in left["nvlink_peers"]
        reverse = left["gpu_uuid"] in right["nvlink_peers"]
        if forward != reverse:
            raise ValueError("Asymmetric NVLink capability evidence")
        return forward and reverse

    if requested_scaleup is not None:
        for first in range(0, len(records), requested_scaleup):
            for i in range(first, first + requested_scaleup):
                for j in range(first, i):
                    if not linked(i, j):
                        raise ValueError(
                            f"Declared scale-up domain [{first}, {first + requested_scaleup}) "
                            f"lacks NVLink connectivity between cohort ranks {j} and {i}"
                        )
        return len(records) // requested_scaleup, requested_scaleup

    adjacency = [{i} for i in range(len(records))]
    for i in range(len(records)):
        for j in range(i):
            if linked(i, j):
                adjacency[i].add(j)
                adjacency[j].add(i)
    unseen, domains = set(range(len(records))), []
    while unseen:
        first = min(unseen)
        domain, frontier = set(), {first}
        while frontier:
            domain.update(frontier)
            frontier = set().union(*(adjacency[i] for i in frontier)) - domain
        if any(adjacency[i] != domain for i in domain):
            raise ValueError(
                "NVLink domain lacks complete pairwise connectivity evidence"
            )
        ordered = sorted(domain)
        if ordered != list(range(first, first + len(domain))):
            raise ValueError(
                "NVLink domains must be contiguous in constructor-cohort rank order"
            )
        unseen.difference_update(domain)
        domains.append(ordered)
    sizes = {len(domain) for domain in domains}
    if len(sizes) != 1:
        raise ValueError("Constructor cohort has nonuniform NVLink domain sizes")
    scaleup = sizes.pop()
    return len(domains), scaleup
