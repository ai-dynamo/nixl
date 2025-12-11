# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Buffer static methods.

Test IDs: U-STAT-01 through U-STAT-05
"""

import pytest


class TestStaticMethods:
    """Tests for Buffer static methods."""

    def test_is_sm90_compiled(self):
        """
        U-STAT-01: is_sm90_compiled() returns bool.
        """
        from nixl_ep import Buffer

        result = Buffer.is_sm90_compiled()

        assert isinstance(
            result, bool
        ), f"is_sm90_compiled() should return bool, got {type(result)}"

    def test_set_num_sms_even(self):
        """
        U-STAT-02: set_num_sms(20) should succeed.
        Even numbers should be accepted.
        """
        from nixl_ep import Buffer

        # Store original value
        original = Buffer.num_sms

        try:
            Buffer.set_num_sms(20)
            assert Buffer.num_sms == 20, "num_sms should be 20"

            Buffer.set_num_sms(16)
            assert Buffer.num_sms == 16, "num_sms should be 16"

            Buffer.set_num_sms(2)
            assert Buffer.num_sms == 2, "num_sms should be 2"
        finally:
            # Restore original value
            Buffer.num_sms = original

    def test_set_num_sms_odd_raises(self):
        """
        U-STAT-03: set_num_sms(21) should raise AssertionError.
        Odd numbers should be rejected.
        """
        from nixl_ep import Buffer

        with pytest.raises(AssertionError):
            Buffer.set_num_sms(21)

        with pytest.raises(AssertionError):
            Buffer.set_num_sms(1)

        with pytest.raises(AssertionError):
            Buffer.set_num_sms(15)

    def test_capture_returns_event_overlap(self, cuda_available):
        """
        U-STAT-04: capture() returns EventOverlap object.
        """
        from nixl_ep import Buffer
        from nixl_ep.utils import EventOverlap

        event = Buffer.capture()

        assert isinstance(
            event, EventOverlap
        ), f"capture() should return EventOverlap, got {type(event)}"

    def test_get_rdma_size_hint_returns_positive_int(self):
        """
        U-STAT-05: get_rdma_size_hint() returns positive int for valid inputs.
        """
        from nixl_ep import Buffer

        # Typical MoE parameters
        num_max_dispatch_tokens = 512
        hidden = 4096
        num_ranks = 8
        num_experts = 64

        size = Buffer.get_rdma_size_hint(
            num_max_dispatch_tokens, hidden, num_ranks, num_experts
        )

        assert isinstance(
            size, int
        ), f"get_rdma_size_hint should return int, got {type(size)}"
        assert size > 0, f"get_rdma_size_hint should return positive value, got {size}"

    def test_get_rdma_size_hint_scales_with_tokens(self):
        """
        Verify that RDMA size scales linearly with token count.
        4x tokens should give ~4x RDMA size (small signaling buffer overhead).
        """
        from nixl_ep import Buffer

        hidden = 4096
        num_ranks = 8
        num_experts = 64

        size_small = Buffer.get_rdma_size_hint(256, hidden, num_ranks, num_experts)
        size_large = Buffer.get_rdma_size_hint(1024, hidden, num_ranks, num_experts)

        # 1024/256 = 4x tokens, should be ~4x RDMA size
        # Small signaling buffer (num_experts * sizeof(int)) doesn't scale
        ratio = size_large / size_small
        assert (
            3.99 <= ratio <= 4.01
        ), f"RDMA size should scale ~4x with 4x tokens, got ratio={ratio}"

    def test_get_rdma_size_hint_scales_with_hidden(self):
        """
        Verify that RDMA size scales linearly with hidden dimension.
        4x hidden should give ~4x RDMA size (small signaling overhead).
        """
        from nixl_ep import Buffer

        num_tokens = 512
        num_ranks = 8
        num_experts = 64

        size_small = Buffer.get_rdma_size_hint(num_tokens, 2048, num_ranks, num_experts)
        size_large = Buffer.get_rdma_size_hint(num_tokens, 8192, num_ranks, num_experts)

        # 8192/2048 = 4x hidden, should be ~4x RDMA size
        # Small signaling buffer (num_experts * sizeof(int)) doesn't scale
        ratio = size_large / size_small
        assert (
            3.99 <= ratio <= 4.01
        ), f"RDMA size should scale ~4x with 4x hidden, got ratio={ratio}"

    def test_get_rdma_size_hint_consistent_with_ranks(self):
        """
        Verify that RDMA size is consistent regardless of rank count.

        Note: The RDMA size hint is based on per-rank buffer requirements,
        so it doesn't scale with the number of ranks - each rank allocates
        its own buffer of this size.
        """
        from nixl_ep import Buffer

        num_tokens = 512
        hidden = 4096
        num_experts = 64

        size_4_ranks = Buffer.get_rdma_size_hint(num_tokens, hidden, 4, num_experts)
        size_16_ranks = Buffer.get_rdma_size_hint(num_tokens, hidden, 16, num_experts)

        # RDMA size hint is per-rank, so it should be consistent
        assert (
            size_4_ranks == size_16_ranks
        ), "RDMA size hint should be consistent (per-rank allocation)"
        assert size_4_ranks > 0, "RDMA size should be positive"
