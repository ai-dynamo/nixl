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

import os
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
from . import nixl_ep_cpp

# noinspection PyUnresolvedReferences
from .nixl_ep_cpp import Config, EventHandle
from .utils import check_nvlink_connections
from .utils import EventOverlap

if TYPE_CHECKING:
    import mpi4py  # noqa: F401


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports dispatch and combine operations using NVLink and RDMA.

    Attributes:
        disable_ll_nvlink: whether to disable NVLink communication for low-latency kernels (default: False).
        rank: the local rank number.
        num_rdma_bytes: the buffer size for RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(
        self,
        disable_ll_nvlink: bool = False,
        explicitly_destroy: bool = False,
        rank: int = 0,
        low_latency_mode: bool = False,
        group: Optional[dist.ProcessGroup] = None,
        comm: Optional["mpi4py.MPI.Comm"] = None,
        tcp_store_group: Optional[dist.TCPStore] = None,
    ) -> None:
        """
        Initialize the nixl communication buffer.

        Arguments:
            disable_ll_nvlink: whether to disable NVLink communication for low-latency kernels (default: False).
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            rank: the rank number.
            low_latency_mode: whether to enable low-latency mode.
            group: the communication group (optional).
            comm: the mpi4py.MPI.Comm communicator to use in case the group parameter is absent (optional).
            tcp_store_group: TCPStore for metadata exchange (optional).
        """
        self.rank = rank
        self.group_size = 0  # Will be updated by `update_memory_buffers`
        self.low_latency_mode = low_latency_mode

        self.explicitly_destroy = explicitly_destroy
        self.group = group
        self.comm = comm
        self.tcp_store_group = tcp_store_group
        assert not (group and comm)

        if disable_ll_nvlink:
            os.environ["UCX_TLS"] = "^cuda_ipc"

        if self.group is not None:
            check_nvlink_connections(self.group)

        self.runtime = nixl_ep_cpp.Buffer(self.rank, low_latency_mode, explicitly_destroy)

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.
        """
        assert self.explicitly_destroy, "`explicitly_destroy` flag must be set"

        self.runtime.destroy()
        self.runtime = None

    @staticmethod
    def is_sm90_compiled():
        return nixl_ep_cpp.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, "The SM count must be even"
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return nixl_ep_cpp.get_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(
            stream_id=ts.stream_id,
            device_index=ts.device_index,
            device_type=ts.device_type,
        )

    def get_local_buffer_tensor(
        self, dtype: torch.dtype, size: Optional[torch.Size] = None, offset: int = 0, use_rdma_buffer: bool = False
    ) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)
        if size is None:
            return tensor

        assert tensor.numel() >= size.numel()
        return tensor[: size.numel()].view(size)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 8, 288, 32, 128),
            32: Config(Buffer.num_sms, 32, 288, 32, 128),
            64: Config(Buffer.num_sms, 20, 288, 28, 128),
            128: Config(Buffer.num_sms, 20, 560, 32, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 20, 128),
            128: Config(Buffer.num_sms, 1, 560, 12, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int,
                            previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                            allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = \
            self.runtime.get_dispatch_layout(topk_idx, num_experts, getattr(previous_event, 'event', None),
                                             async_finish, allocate_on_comm_stream)
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap(event)

    def clean_buffer(
        self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int
    ) -> None:
        """
        As the kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        self.runtime.clean_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts)

    # noinspection PyTypeChecker
    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable
    ]:
        """
        A low-latency implementation for dispatching with NIXL device API.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            kernels' result tensors at a single moment.

        Arguments:
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `nixl_ep.topk_idx_t`, shaped as `[num_tokens, num_topk]`, only several top-k shapes
                are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            cumulative_local_expert_recv_stats: a cumulative expert count tensor for statistics, which should have shape
                `[num_local_experts]` and be typed as `torch.int`. This is useful for online service EP load balance
                monitoring.
            dispatch_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and pre-cisely localizing slow anomalies.
            use_fp8: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            round_scale: whether round the scaling factors into power of 2.
            use_ue8m0: whether use UE8M0 as scaling factor format (available only with `round_scale=True`).
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tensor or tuple with received tokens for each expert.
                With `use_fp8=True`: the first element is a `torch.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`,
                if `use_ue8m0=False`. With `use_ue8m0=True`, the second one is packed and shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]` with type `torch.int`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                With `use_fp8=False`, the result would be a tensor shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receives. As mentioned before, not all tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            event,
            hook,
        ) = self.runtime.dispatch(
            x,
            topk_idx,
            cumulative_local_expert_recv_stats,
            dispatch_wait_recv_cost_stats,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8,
            round_scale,
            use_ue8m0,
            async_finish,
            return_recv_hook,
        )
        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
        )
        tensors_to_record = (
            x,
            topk_idx,
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            cumulative_local_expert_recv_stats,
        )
        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    # noinspection PyTypeChecker
    def combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        use_logfmt: bool = False,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
        combine_wait_recv_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        A low-latency implementation for combining tokens (reduce **with weights**) with NIXL device API.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            kernels' result tensors at a single moment.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `nixl_ep.topk_idx_t`, the expert indices selected by the dispatched
                tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals
                to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            use_logfmt: whether to use an internal "LogFMT with dynamic per-64-channel cast" format (10 bits).
            zero_copy: whether the tensor is already copied into the RDMA buffer, should be cooperative
                with `get_next_combine_buffer`.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.
            combine_wait_recv_cost_stats: a cumulative time spent waiting to receive each token tensor for statistics,
                which should have shape `[num_ranks, num_ranks]` and be typed as `torch.int64`.
                This is useful for detecting and pre-cisely localizing slow anomalies.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        combined_x, event, hook = self.runtime.combine(
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combine_wait_recv_cost_stats,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_logfmt,
            zero_copy,
            async_finish,
            return_recv_hook,
            out,
        )
        tensors_to_record = (
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combined_x,
        )
        return (
            combined_x,
            EventOverlap(event, tensors_to_record if async_finish else None),
            hook,
        )

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           handle: Optional[Tuple] = None,
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                           config: Optional[Config] = None,
                           previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        config = self.get_dispatch_config(self.group_size) if config is None else config
        assert config is not None

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            is_token_in_rank, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head = handle
            num_recv_tokens = recv_src_meta.size(0)
            num_rdma_recv_tokens = send_nvl_head.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, _, _, _, _, event = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights,
                None, None, is_token_in_rank, None,
                num_recv_tokens, num_rdma_recv_tokens,
                rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                expert_alignment, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, \
                rdma_channel_prefix_matrix, gbl_channel_prefix_matrix, \
                recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
                recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
                recv_src_meta, send_rdma_head, send_nvl_head, event = self.runtime.internode_dispatch(
                x, x_scales, topk_idx, topk_weights,
                num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert,
                0, 0, None, None, None, None,
                expert_alignment, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (is_token_in_rank,
                      rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
                      recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                      recv_src_meta, send_rdma_head, send_nvl_head)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],
                          topk_weights: Optional[torch.Tensor] = None,
                          bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                          config: Optional[Config] = None,
                          previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        config = self.get_combine_config(self.group_size) if config is None else config
        assert config is not None

        # Unpack handle and bias
        is_combined_token_in_rank, \
            _, _, \
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, gbl_rank_prefix_sum, \
            src_meta, send_rdma_head, send_nvl_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        combined_x, combined_topk_weights, event = self.runtime.internode_combine(
            x, topk_weights, bias_0, bias_1,
            src_meta, is_combined_token_in_rank,
            rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
            send_rdma_head, send_nvl_head, config, getattr(previous_event, 'event', None),
            async_finish, allocate_on_comm_stream)
        return combined_x, combined_topk_weights, EventOverlap(event)


    def update_mask_buffer(self, rank_to_mask: int, mask: bool = False):
        """
        Mask (unmask) a rank during communication (dispatch, combine, and clean)

        Arguments:
            rank: the rank to mask (unmask).
            mask: if True, will mask the rank (do not recvfrom/sendto the rank), otherwise will unmask the rank.
        """
        self.runtime.update_mask_buffer(rank_to_mask, mask)

    def query_mask_buffer(self, mask_status: torch.Tensor):
        """
        Query the mask status of all ranks

        Arguments:
            mask_status: `[num_ranks]` with `torch.int`, the mask status of each rank. `1` means mask and `0` means unmasked.
        """
        self.runtime.query_mask_buffer(mask_status)

    def clean_mask_buffer(self):
        """
        Clean the mask buffer

        """
        self.runtime.clean_mask_buffer()

    def get_next_combine_buffer(
        self, handle: Tuple[torch.Tensor, torch.Tensor, int, int, int]
    ):
        """
        Get the raw registered RDMA buffer tensor for next combine, so that the next combine kernel can skip the copying.

        Arguments:
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA buffer as a BF16 PyTorch tensor with shape
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        return self.runtime.get_next_combine_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )

    def update_memory_buffers(
        self, num_ranks: int, num_experts_per_rank: int, num_nvl_bytes: int, num_rdma_bytes: int
    ):
        """
        Allocate remote memory for the communication buffer.

        Arguments:
            num_ranks: the number of ranks.
            num_experts_per_rank: the number of experts per rank.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for RDMA communication.
        """
        self.group_size = num_ranks
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.runtime.update_memory_buffers(
            num_ranks, num_experts_per_rank, num_nvl_bytes,num_rdma_bytes
        )

    def set_tcp_store_group(self, tcp_store_group: Optional[dist.TCPStore]) -> None:
        """
        Update the TCP Store group for metadata exchange.

        Arguments:
            tcp_store_group: Optional TCPStore for metadata exchange.
        """
        self.tcp_store_group = tcp_store_group

    @contextmanager
    def _fetch_remote_metadata_from_tcp_store(self, remote_ranks: List[int]):
        assert self.tcp_store_group is not None, "TCPStore group is not set"
        md_key = f"NIXL_EP/{self.rank}"
        nixl_metadata_bytes = self.runtime.get_local_metadata()
        self.tcp_store_group.set(md_key, nixl_metadata_bytes)

        remote_md_keys = [f"NIXL_EP/{rank}" for rank in remote_ranks]
        if remote_md_keys:
            self.tcp_store_group.wait(remote_md_keys, timedelta(seconds=300))
            remote_mds = self.tcp_store_group.multi_get(remote_md_keys)
        else:
            remote_mds = []

        try:
            yield remote_mds
        finally:
            self.tcp_store_group.delete_key(md_key)

    def connect_ranks(self, remote_ranks: List[int]) -> None:
        """
        Add connections to remote ranks.

        Arguments:
            remote_ranks: List of remote rank IDs to establish connections with.
                         The current rank will be automatically filtered out.
        """
        if self.low_latency_mode:
            if self.tcp_store_group is not None:
                with self._fetch_remote_metadata_from_tcp_store(remote_ranks) as remote_mds:
                    self.runtime.connect_ranks(remote_ranks, remote_mds)
            else:
                self.runtime.connect_ranks(remote_ranks)
        else:
            # High-throughput internode mode: need group for IPC handles
            if self.group is not None:
                def all_gather_object(obj):
                    object_list = [None] * self.group_size
                    dist.all_gather_object(object_list, obj, self.group)
                    return object_list
            elif self.comm is not None:
                def all_gather_object(obj):
                    return self.comm.allgather(obj)
            else:
                raise ValueError("Either 'group' or 'comm' must be configured.")

            local_ipc_handle = self.runtime.get_local_ipc_handle()
            ipc_handles = all_gather_object(local_ipc_handle)

            # Use TCPStore for NIXL metadata exchange if available, otherwise use ETCD
            if self.tcp_store_group is not None:
                with self._fetch_remote_metadata_from_tcp_store(remote_ranks) as remote_mds:
                    self.runtime.connect_ranks(remote_ranks, remote_mds, ipc_handles)
            else:
                self.runtime.connect_ranks(remote_ranks, None, ipc_handles)


    def disconnect_ranks(self, remote_ranks: List[int]) -> None:
        """
        Remove connections to remote ranks.

        Arguments:
            remote_ranks: List of remote rank IDs to remove connections from.
                         These ranks must be at the end of the current remote_ranks list.
        """
        self.runtime.disconnect_ranks(remote_ranks)

    def barrier(self) -> None:
        """
        barrier for all active ranks.
        notice that this barrier does not flush the network QPs as it is currently doesn't have any use-case that requires it
        """
        self.runtime.barrier()
