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
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
from . import nixl_ep_cpp

# noinspection PyUnresolvedReferences
from .nixl_ep_cpp import EventHandle
from .utils import EventOverlap

if TYPE_CHECKING:
    import mpi4py  # noqa: F401


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports dispatch and combine operations using NVLink and RDMA.

    Attributes:
        nvlink_backend: the backend for NVLink communication, you can choose from 'nixl', 'ipc' and 'none' (which disables NVLink entirely).
        rank: the local rank number.
        num_rdma_bytes: the buffer size for RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(
        self,
        nvlink_backend: Literal["nixl", "ipc", "none"] = "nixl",
        explicitly_destroy: bool = False,
        rank: int = 0,
        enable_shrink: bool = False,
        group: Optional[dist.ProcessGroup] = None,
        comm: Optional["mpi4py.MPI.Comm"] = None,
        tcp_store_group: Optional[dist.TCPStore] = None,
    ) -> None:
        """
        Initialize the nixl communication buffer.

        Arguments:
            nvlink_backend: nvlink implementation to use, you can choose from 'nixl', 'ipc' and 'none' (which disables NVLink entirely).
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            rank: the rank number.
            group: the communication group (optional).
            comm: the mpi4py.MPI.Comm communicator to use in case the group parameter is absent (optional).
            tcp_store_group: TCPStore for metadata exchange (optional).
        """
        self.rank = rank
        self.group_size = 0  # Will be updated by `update_memory_buffers`
        self.explicitly_destroy = explicitly_destroy
        self.group = group
        self.comm = comm
        self.tcp_store_group = tcp_store_group
        assert not (group and comm)

        # Configure NVLINK backend
        os.environ["NIXL_EP_NVLINK_BACKEND_IPC"] = (
            "1" if nvlink_backend == "ipc" else "0"
        )
        if nvlink_backend != "nixl":
            os.environ["UCX_TLS"] = "^cuda_ipc"

        self.runtime = nixl_ep_cpp.Buffer(self.rank, explicitly_destroy, enable_shrink)

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
        self, dtype: torch.dtype, size: Optional[torch.Size] = None, offset: int = 0
    ) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset)
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
        self, num_ranks: int, num_experts_per_rank: int, num_rdma_bytes: int
    ):
        """
        Allocate remote memory for the communication buffer.

        Arguments:
            num_ranks: the number of ranks.
            num_experts_per_rank: the number of experts per rank.
            num_rdma_bytes: the buffer size for RDMA communication.
        """
        self.group_size = num_ranks
        self.num_rdma_bytes = num_rdma_bytes
        self.runtime.update_memory_buffers(
            num_ranks, num_experts_per_rank, num_rdma_bytes
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
        if self.tcp_store_group is not None:
            with self._fetch_remote_metadata_from_tcp_store(remote_ranks) as remote_mds:
                self.runtime.connect_ranks(remote_ranks, remote_mds)
        else:
            self.runtime.connect_ranks(remote_ranks)

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
