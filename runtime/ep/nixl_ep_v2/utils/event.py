# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

from typing import Any, Callable, Optional, Tuple

import torch

# noinspection PyUnresolvedReferences
from .._C import EventHandle


class EventOverlap:
    """CUDA event with optional tensor retention and context-manager waits."""

    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[torch.Tensor]] = None,
    ) -> None:
        self.event = event

        # Native events retain EPv2 tensors; this retains caller-supplied tensors.
        self.extra_tensors = extra_tensors

        # A wrapper for `with event_overlap(release_handle=True)`
        self._release_handle_by_call = False

        # Deterministic dispatch sorts on the current stream after the wait.
        self.hook_after_wait: Optional[Callable] = None
        self._hook_event: Optional[EventHandle] = None

    def current_stream_wait(self, release_handle: bool = False) -> None:
        """Queue a wait on the current CUDA stream, then invoke any registered hook.

        With ``NIXL_EP_AVOID_RECORD_STREAM=1``, keep the event or tensors alive
        until their last GPU use completes, unless otherwise protected.
        ``release_handle=True`` drops retained references without waiting for consumers.
        """
        assert self.event is not None
        event = self._hook_event if self._hook_event is not None else self.event
        event.current_stream_wait()

        if self.hook_after_wait is not None:
            self.hook_after_wait()
            self._hook_event = EventHandle()
            self.hook_after_wait = None

        # Keep the original tensor-owning event and the hook completion for later streams.
        if release_handle:
            self.event = self._hook_event = None

    def register_hook_after_wait(self, hook_after_wait: Callable) -> None:
        """Register a one-shot hook after the stream wait."""
        assert (
            self.hook_after_wait is None
        ), "A hook is already registered on this `EventOverlap`"
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
