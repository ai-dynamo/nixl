# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright afdEP-comm contributors (formerly afdEP).
# Adapted from afdEP-comm benchmarks/fastafd.py at 575d2f9. See NOTICE.
"""The baseline afdEP-comm is measured against: FastAFD's DeepEP M2N.

FastAFD (https://github.com/hao-ai-lab/FastAFD) runs AFD over DeepEP V2 with its
``DeepEPM2NAdapter`` (python/minisgl/moe/deepep_m2n_adapter.py): DeepEP sees a
symmetric expert-parallel group, every rank sends, receives and waits in both
phases, and Attention ranks hold dummy expert slots that no token routes to. The
AFD plugin's ``deepep_m2n`` connector (https://github.com/ms-llmd/afd-plugin) ships
the same snapshot and layout. This module drives the same contract over the DeepEP
V2 kernels vendored here, unspecialized, behind afdEP-comm's role API, so both arms
share one build and harness. The upstream validation
(`docs/validation-fastafd-equivalence.json` in afdEP-comm) checks it against
the plugin's own buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from afdep import Buffer, Config
from afdep.deepep import ElasticBuffer

from .afdep import Arm as AfdEPArm

if TYPE_CHECKING:
    import torch.distributed as dist


@dataclass(frozen=True)
class FastAFDConfig(Config):
    """Config whose receive bound counts every rank of the symmetric group."""

    def receive_rows(self, bucket: int) -> int:
        return self._receive_rows(self.world_size, bucket)


class FastAFDBuffer(Buffer):
    """afdEP-comm's role API over unspecialized DeepEP V2, as FastAFD runs it."""

    def _open_transport(self, group: dist.ProcessGroup) -> ElasticBuffer:
        config = self.config
        # No ffn_ranks: DeepEP V2 as upstream ships it, over a symmetric group.
        return ElasticBuffer(
            group=group,
            capacity_tokens_per_rank=config.capacity_tokens_per_rank,
            hidden_size=config.hidden_size,
            num_experts=config.logical_experts,
            top_k=config.top_k,
            allow_hybrid_mode=config.allow_hybrid_mode,
            rdma_traffic_class=config.rdma_traffic_class,
            cpu_timeout_s=config.cpu_timeout_s,
            gpu_timeout_s=config.gpu_timeout_s,
            use_fp8_dispatch=config.payload == "fp8",
            num_sms=config.num_sms,
            num_qps=config.num_qps,
        )

    def _source_output_rows(self) -> int:
        # As a member of the symmetric group, a source combines a full receive bound.
        return self.config.receive_rows(self.config.capacity_tokens_per_rank)

    def source_dispatch(
        self, x, ids, weights, x_scale=None, *, logical=False, **kwargs
    ) -> None:
        # Upstream DeepEP routes logical experts. A per-domain FFN prefix puts FFN
        # blocks off [0, num_experts), so the baseline maps real IDs as a caller
        # would, unless the caller passes logical IDs (``Config.logical_ids``).
        if self.config.ffn_ranks_per_domain and not logical:
            ids = self.config.logical_ids(ids)
        super().source_dispatch(x, ids, weights, x_scale, **kwargs)


class Arm(AfdEPArm):
    backend = "fastafd"
    buffer_type = FastAFDBuffer
    config_type = FastAFDConfig

    def prepare_routes(self, routes):
        if self.config.ffn_ranks_per_domain:
            return self.config.logical_ids(routes), {"logical": True}
        return routes, {}
