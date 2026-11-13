# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""API shell; production docstrings below are API review contracts.

Calls are inert: no CUDA work, synchronization or callback execution occurs.
Hooks are retained until a wait discards them.
"""
from typing import Any, Callable, Optional, Tuple

import torch


class EventHandle:
    """Placeholder for a recorded event."""

    def __init__(self):
        pass

    def current_stream_wait(self):
        pass


class EventOverlap:
    """CUDA event with optional tensor retention and context-manager waits."""

    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[torch.Tensor]] = None,
    ) -> None:
        self.event = event
        self.extra_tensors = extra_tensors

        # A wrapper for `with event_overlap(release_handle=True)`
        self._release_handle_by_call = False

        self.hook_after_wait: Optional[Callable] = None

    def current_stream_wait(self, release_handle: bool = False) -> None:
        """Queue a wait on the current CUDA stream, then invoke any registered hook.

        With ``NIXL_EP_AVOID_RECORD_STREAM=1``, keep the event or tensors alive
        until their last GPU use completes, unless otherwise protected.
        ``release_handle=True`` drops retained references without waiting for consumers.
        """
        if self.event is None:
            raise ValueError("No event to wait on")
        self.hook_after_wait = None
        if release_handle:
            self.event = None

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        """Register a one-shot hook after the stream wait."""
        if self.hook_after_wait is not None:
            raise ValueError("A hook is already registered")
        self.hook_after_wait = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        """Configure release on context exit; see ``current_stream_wait`` for lifetime requirements."""
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> Any:
        """Run work inside ``with event_overlap:``; wait for the event on exit."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.event is not None:
            self.current_stream_wait(release_handle=self._release_handle_by_call)
        self._release_handle_by_call = False
