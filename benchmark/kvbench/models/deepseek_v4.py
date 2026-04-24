# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, List, Optional

import yaml  # type: ignore
from models.model_config import ModelConfig
from models.models import BaseModelArch


_CSA_RATIO = 4
_HCA_RATIO = 128
_INDEX_DTYPE_BYTES = {"fp4": 0.5, "fp8": 1.0, "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0}


class DeepSeekV4(BaseModelArch):
    """DeepSeek-V4 hybrid attention architecture.

    Interleaves Compressed Sparse Attention (CSA, compress_ratio=4) and
    Heavily Compressed Attention (HCA, compress_ratio=128) with a
    sliding-window branch on every layer. CSA layers additionally carry a
    lightning-indexer cache for top-k sparse selection.
    """

    def __init__(
        self,
        model_name: str,
        num_hidden_layers: int,
        num_nextn_predict_layers: int,
        num_query_heads: int,
        num_key_value_heads: int,
        kv_head_dim: int,
        kv_rope_dim: int,
        embedding_dimension: int,
        compress_ratios: List[int],
        sliding_window: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        num_model_params: int,
        model_config: Optional[ModelConfig] = None,
    ):
        self.model_name = model_name
        self.model_config = model_config or ModelConfig()
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_head_dim = kv_head_dim
        self.kv_rope_dim = kv_rope_dim
        self.embedding_dimension = embedding_dimension
        self.compress_ratios = list(compress_ratios)
        self.sliding_window = sliding_window
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.num_model_params = num_model_params

        self.num_layers = num_hidden_layers + num_nextn_predict_layers

        self._validate()

    def _validate(self) -> None:
        expected = self.num_hidden_layers + self.num_nextn_predict_layers
        if len(self.compress_ratios) != expected:
            raise ValueError(
                f"compress_ratios length {len(self.compress_ratios)} does not match "
                f"num_hidden_layers + num_nextn_predict_layers = {expected}"
            )
        for i, r in enumerate(self.compress_ratios):
            if r not in (0, _CSA_RATIO, _HCA_RATIO):
                raise ValueError(
                    f"compress_ratios[{i}] = {r}; expected 0 (SWA), "
                    f"{_CSA_RATIO} (CSA), or {_HCA_RATIO} (HCA)"
                )
        if self.num_key_value_heads < 1:
            raise ValueError(f"num_key_value_heads must be >= 1, got {self.num_key_value_heads}")
        if any(r == 0 for r in self.compress_ratios) and self.sliding_window < 1:
            raise ValueError(
                "sliding_window must be >= 1 when any compress_ratio == 0 (SWA layer present)"
            )

    def _main_slot_bytes(self) -> int:
        # Mixed-precision storage: NoPE dims in fp8, RoPE dims in bf16.
        mode = self.model_config.model.kvcache_quant_mode
        c, rope = self.kv_head_dim, self.kv_rope_dim
        if mode == "fp8_mixed":
            return (c - rope) * 1 + rope * 2
        if mode in ("fp8", "int8"):
            return c * 1
        if mode in ("bf16", "bfloat16", "fp16"):
            return c * 2
        raise ValueError(f"Unsupported kvcache_quant_mode for V4: {mode!r}")

    def _index_slot_bytes(self) -> int:
        mode = self.model_config.model.index_quant_mode
        if mode not in _INDEX_DTYPE_BYTES:
            raise ValueError(
                f"index_quant_mode {mode!r} not in {list(_INDEX_DTYPE_BYTES)}"
            )
        return int(self.index_head_dim * _INDEX_DTYPE_BYTES[mode])

    def _csa_count(self) -> int:
        return sum(1 for r in self.compress_ratios if r == _CSA_RATIO)

    def _hca_count(self) -> int:
        return sum(1 for r in self.compress_ratios if r == _HCA_RATIO)

    def _swa_layer_count(self) -> int:
        # SWA buffer exists on every CSA and HCA layer, and the MTP layer
        # itself runs pure SWA.
        return self._csa_count() + self._hca_count() + self.num_nextn_predict_layers

    def get_kv_size_per_token(self, token_count: int = 1) -> int:
        """Amortized bytes per token (for get_batch_size).

        Excludes the sliding-window fixed cost: SWA buffers don't grow with
        context length past n_win, so their amortized per-token contribution
        vanishes at the large-L regime where batching matters.
        """
        p_main = self._main_slot_bytes()
        p_idx = self._index_slot_bytes()
        n_csa = self._csa_count()
        n_hca = self._hca_count()
        # Integer multiply-before-divide over LCM(4, 128) = 128 to stay exact.
        main = p_main * (n_csa * _HCA_RATIO + n_hca * _CSA_RATIO) // (_CSA_RATIO * _HCA_RATIO)
        idx = p_idx * n_csa // _CSA_RATIO
        return int((main + idx) * token_count)

    def get_total_kv_bytes(self, seq_len: int) -> int:
        """Exact total KV cache bytes at sequence length `seq_len`.

        Includes: compressed main KV (CSA+HCA), lightning-indexer cache (CSA
        only), and sliding-window fixed cost (every layer).
        """
        p_main = self._main_slot_bytes()
        p_idx = self._index_slot_bytes()
        n_csa = self._csa_count()
        n_hca = self._hca_count()
        n_sw_layers = self._swa_layer_count()

        csa_slots = math.ceil(seq_len / _CSA_RATIO)
        hca_slots = math.ceil(seq_len / _HCA_RATIO)

        main_bytes = p_main * (n_csa * csa_slots + n_hca * hca_slots)
        indexer_bytes = p_idx * n_csa * csa_slots
        swa_bytes = n_sw_layers * self.sliding_window * p_main
        return int(main_bytes + indexer_bytes + swa_bytes)

    def get_io_size(self, page_size: int = 1) -> int:
        raise NotImplementedError(
            "DeepSeek V4 has heterogeneous layers; use get_io_sizes() instead"
        )

    def get_io_sizes(self, page_size: int = 1) -> Dict[str, int]:
        """Return a label → per-transfer IO bytes map for V4.

        One key per distinct layer class present in `compress_ratios`, plus a
        per-layer SWA entry when sliding-window layers exist, plus a
        `block__worst_stage` entry when access_pattern == "block".
        """
        p_main = self._main_slot_bytes()
        p_idx = self._index_slot_bytes()
        tp = max(self.model_config.model.tp_size, 1)
        out: Dict[str, int] = {}

        if self._csa_count() > 0:
            out["layer__csa_main"] = max(p_main * page_size // _CSA_RATIO // tp, 1)
            out["layer__csa_indexer"] = max(p_idx * page_size // _CSA_RATIO // tp, 1)
        if self._hca_count() > 0:
            out["layer__hca_main"] = max(p_main * page_size // _HCA_RATIO // tp, 1)
        if self._swa_layer_count() > 0:
            # SWA buffer is fixed at n_win tokens per layer, independent of page_size.
            out["layer__swa_per_layer"] = max(p_main * self.sliding_window // tp, 1)

        if self.model_config.system.access_pattern == "block":
            out["block__worst_stage"] = self._block_worst_stage_bytes(page_size, tp)
        return out

    def _block_worst_stage_bytes(self, page_size: int, tp: int) -> int:
        """Worst-case sum of per-layer bytes across any single pipeline stage.

        For even pp partitioning, returns the max over stages `s` of the
        contiguous slice's per-page bytes (main + indexer contributions,
        amortized per token times page_size). Clamps gracefully when
        pp_size > num_layers.
        """
        pp = max(self.model_config.model.pp_size, 1)
        total_layers = len(self.compress_ratios)
        layers_per_stage = max(math.ceil(total_layers / pp), 1)
        p_main = self._main_slot_bytes()
        p_idx = self._index_slot_bytes()

        worst = 0
        for start in range(0, total_layers, layers_per_stage):
            slice_ratios = self.compress_ratios[start : start + layers_per_stage]
            stage = 0
            for r in slice_ratios:
                if r == _CSA_RATIO:
                    stage += (p_main + p_idx) * page_size // _CSA_RATIO
                elif r == _HCA_RATIO:
                    stage += p_main * page_size // _HCA_RATIO
                # r == 0: SWA-only layer; its per-page contribution is 0
                # (the fixed SWA buffer is transferred via layer__swa_per_layer).
            worst = max(worst, stage)
        return max(worst // tp, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name.lower(),
            "num_hidden_layers": self.num_hidden_layers,
            "num_nextn_predict_layers": self.num_nextn_predict_layers,
            "num_query_heads": self.num_query_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "kv_head_dim": self.kv_head_dim,
            "kv_rope_dim": self.kv_rope_dim,
            "embedding_dimension": self.embedding_dimension,
            "compress_ratios": self.compress_ratios,
            "sliding_window": self.sliding_window,
            "index_head_dim": self.index_head_dim,
            "index_n_heads": self.index_n_heads,
            "index_topk": self.index_topk,
            "num_model_params": self.num_model_params,
        }

    def __str__(self) -> str:
        return yaml.dump(self.to_dict())
