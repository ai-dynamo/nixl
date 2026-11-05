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

"""Synthetic GPU checks of `EPHandle.deterministic_sort`, independent of transport setup: the sort is captured into a
CUDA graph and replayed with other receive counts, as a serving loop does after the world changed under fixed slots."""
import torch

from nixl.runtime.ep import EPHandle


def make_handle(
    metadata, count, do_expand=False, ends=None, counts=None, linked_list=None
):
    return EPHandle(
        do_expand=do_expand,
        expert_alignment=4,
        num_max_tokens_per_rank=4,
        num_sms=2,
        topk_idx=torch.zeros((metadata.shape[0], 2), dtype=torch.int64, device="cuda"),
        num_recv_tokens=metadata.shape[0],
        num_expanded_tokens=12 if do_expand else 0,
        num_recv_tokens_per_expert_list=[],
        psum_num_recv_tokens_per_scaleup_rank=count,
        psum_num_recv_tokens_per_expert=(
            ends if ends is not None else torch.zeros(4, dtype=torch.int, device="cuda")
        ),
        num_unaligned_recv_tokens_per_expert=(
            counts
            if counts is not None
            else torch.zeros(4, dtype=torch.int, device="cuda")
        ),
        recv_src_metadata=metadata,
        token_metadata_at_forward=None,
        channel_linked_list=linked_list,
        recv_count_is_exact=False,
    )


def check_deterministic_sort():
    # The public linked-list view is shorter than its allocation, as after capture in a smaller world.
    original_metadata = torch.tensor(
        [[3, 0, -1, -1], [1, 0, -1, -1], [4, 0, -1, -1], [2, 0, -1, -1]],
        dtype=torch.int,
        device="cuda",
    )
    metadata = original_metadata.clone()
    original_x = original_metadata[:, :1].float()
    x = original_x.clone()
    weights = original_x.clone()
    original_links = torch.tensor(
        [-1, 0, 1, 2, 3, -1, 1, 99], dtype=torch.int, device="cuda"
    )
    links = original_links.clone()
    count = torch.tensor([2], dtype=torch.int, device="cuda")
    handle = make_handle(metadata, count, linked_list=links[:2])

    def sort_rows():
        handle.deterministic_sort(
            False, x, None, None, weights, handle.channel_linked_list
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        sort_rows()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sort_rows()
    for num_rows in (0, 2, 4):
        metadata.copy_(original_metadata)
        x.copy_(original_x)
        weights.copy_(original_x)
        links.copy_(original_links)
        count.fill_(num_rows)
        graph.replay()
        order = torch.argsort(original_metadata[:num_rows, 0])
        assert torch.equal(x[:num_rows], original_x[:num_rows][order])
        assert torch.equal(weights[:num_rows], original_x[:num_rows][order])
        reverse = torch.empty_like(order)
        reverse[order] = torch.arange(num_rows, device="cuda")
        expected = original_links.clone()
        valid = (expected >= 0) & (expected < num_rows)
        expected[valid] = reverse[expected[valid].long()].to(expected.dtype)
        assert torch.equal(links, expected)
    del graph

    # Empty internal/final experts, padding, -1 slots, and out-of-range garbage in an unused metadata row.
    original_metadata = torch.tensor(
        [[9, 0, 0, -1], [3, 0, 1, -1], [5, 0, 4, -1], [999, 0, 100, -1]],
        dtype=torch.int,
        device="cuda",
    )
    metadata = original_metadata.clone()
    ends = torch.tensor([2, 4, 5, 8], dtype=torch.int, device="cuda")
    counts = torch.tensor([2, 0, 1, 0], dtype=torch.int, device="cuda")
    count = torch.tensor([3], dtype=torch.int, device="cuda")
    original_x = torch.tensor(
        [90, 30, 0, 0, 50, 0, 0, 0, 7, 8, 9, 10], dtype=torch.float, device="cuda"
    )[:, None]
    x, weights = original_x.clone(), original_x[:, 0].clone()
    handle = make_handle(metadata, count, True, ends, counts)

    def sort_expanded():
        handle.deterministic_sort(False, x, None, None, weights, None)

    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        sort_expanded()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sort_expanded()
    metadata.copy_(original_metadata)
    x.copy_(original_x)
    weights.copy_(original_x[:, 0])
    graph.replay()
    expected_order = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device="cuda")
    assert torch.equal(x, original_x[expected_order])
    assert torch.equal(weights, original_x[expected_order, 0])
    expected_metadata = original_metadata.clone()
    expected_metadata[0, 2], expected_metadata[1, 2] = 1, 0
    assert torch.equal(metadata, expected_metadata)
    del graph

    # An exact zero-row expanded output has no safe index 0 to gather/scatter.
    empty_metadata = torch.empty((0, 4), dtype=torch.int, device="cuda")
    handle = make_handle(
        empty_metadata, torch.zeros(1, dtype=torch.int, device="cuda"), True
    )
    handle.deterministic_sort(
        True,
        torch.empty((0, 1), device="cuda"),
        None,
        None,
        torch.empty(0, device="cuda"),
        None,
    )
    torch.cuda.synchronize()
    print(
        "Deterministic sort: graph replay at 0 / 2 / 4 rows, expanded layout with empty experts, zero rows: passed"
    )


if __name__ == "__main__":
    check_deterministic_sort()
