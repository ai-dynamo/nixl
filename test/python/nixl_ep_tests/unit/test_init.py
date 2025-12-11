# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Buffer initialization.

Test IDs: U-INIT-01 through U-INIT-06
"""

import os

import pytest
import torch


class TestBufferInitialization:
    """Tests for Buffer.__init__() and initialization options."""

    def test_init_nvlink_backend_nixl(self, cuda_available):
        """
        U-INIT-01: Initialize with nvlink_backend='nixl'
        Verify environment variable is set correctly.
        """
        from nixl_ep import Buffer

        # Clear any existing env var
        os.environ.pop("NIXL_EP_NVLINK_BACKEND_IPC", None)

        buffer = Buffer(nvlink_backend="nixl", rank=0, explicitly_destroy=True)

        # Verify env var was set
        assert (
            os.environ.get("NIXL_EP_NVLINK_BACKEND_IPC") == "0"
        ), "NIXL_EP_NVLINK_BACKEND_IPC should be '0' for nixl backend"

        buffer.destroy()

    def test_init_nvlink_backend_ipc(self, cuda_available):
        """
        U-INIT-02: Initialize with nvlink_backend='ipc'
        Verify environment variable is set correctly.
        """
        from nixl_ep import Buffer

        buffer = Buffer(nvlink_backend="ipc", rank=0, explicitly_destroy=True)

        assert (
            os.environ.get("NIXL_EP_NVLINK_BACKEND_IPC") == "1"
        ), "NIXL_EP_NVLINK_BACKEND_IPC should be '1' for ipc backend"

        buffer.destroy()

    def test_init_nvlink_backend_none(self, cuda_available):
        """
        U-INIT-03: Initialize with nvlink_backend='none'
        Verify UCX_TLS is configured to disable cuda_ipc.
        """
        from nixl_ep import Buffer

        buffer = Buffer(nvlink_backend="none", rank=0, explicitly_destroy=True)

        # Verify UCX_TLS excludes cuda_ipc
        ucx_tls = os.environ.get("UCX_TLS", "")
        assert (
            "^cuda_ipc" in ucx_tls or "cuda_ipc" not in ucx_tls
        ), "UCX_TLS should exclude cuda_ipc for 'none' backend"

        buffer.destroy()

    def test_init_group_comm_mutual_exclusion(self, cuda_available):
        """
        U-INIT-04: Mutually exclusive group/comm parameters.
        Providing both group and comm should raise an assertion.
        """
        from nixl_ep import Buffer

        # Create a mock comm object (we just need something non-None)
        class MockComm:
            pass

        mock_comm = MockComm()

        # Create a mock group (we just need something non-None)
        class MockGroup:
            pass

        mock_group = MockGroup()

        # Should raise assertion when both are provided
        with pytest.raises(AssertionError):
            Buffer(rank=0, group=mock_group, comm=mock_comm)

    def test_init_explicitly_destroy_true(self, cuda_available):
        """
        U-INIT-05: explicitly_destroy=True flag.
        destroy() should succeed without error.
        """
        from nixl_ep import Buffer

        buffer = Buffer(rank=0, explicitly_destroy=True)

        # Should not raise
        buffer.destroy()

        # Runtime should be None after destroy
        assert buffer.runtime is None, "Runtime should be None after destroy()"

    def test_init_explicitly_destroy_false_destroy_raises(self, cuda_available):
        """
        U-INIT-06: explicitly_destroy=False flag.
        Calling destroy() should raise AssertionError.
        """
        from nixl_ep import Buffer

        buffer = Buffer(rank=0, explicitly_destroy=False)

        with pytest.raises(AssertionError):
            buffer.destroy()

        # Clean up - buffer will be destroyed by destructor
        # (we can't explicitly destroy it, so just let it go out of scope)

    def test_init_default_values(self, cuda_available):
        """
        Test that default initialization works with minimal parameters.
        """
        from nixl_ep import Buffer

        buffer = Buffer(rank=0, explicitly_destroy=True)

        assert buffer.rank == 0
        assert buffer.runtime is not None

        buffer.destroy()

    def test_init_with_rank(self, cuda_available):
        """
        Test initialization with different rank values.
        """
        from nixl_ep import Buffer

        for rank in [0, 1, 7]:
            buffer = Buffer(rank=rank, explicitly_destroy=True)
            assert buffer.rank == rank, f"Buffer rank should be {rank}"
            buffer.destroy()
