# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check event ordering and ownership with only the native/CUDA boundary replaced."""

import importlib.util
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class EventOverlapTests(unittest.TestCase):
    def setUp(self):
        self.stream = "communication"
        self.timeline = []
        self.events = []
        state = self

        class EventHandle:
            def __init__(self):
                self.index = len(state.events)
                state.events.append(weakref.ref(self))
                state.timeline.append(("record", self.index, state.stream))

            def current_stream_wait(self):
                state.timeline.append(("wait", self.index, state.stream))

        self.EventHandle = EventHandle
        path = Path(__file__).resolve().parents[2] / "nixl_ep_v2/utils/event.py"
        spec = importlib.util.spec_from_file_location("event_host.utils.event", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(Tensor=object),
                "event_host._C": SimpleNamespace(EventHandle=EventHandle),
            },
        ):
            spec.loader.exec_module(module)
        self.EventOverlap = module.EventOverlap

    def test_later_stream_waits_for_hook_and_retains_original_event(self):
        overlap = self.EventOverlap(self.EventHandle())
        overlap.register_hook_after_wait(
            lambda: self.timeline.append(("sort", self.stream))
        )
        self.stream = "A"
        overlap.current_stream_wait()
        self.stream = "B"
        overlap.current_stream_wait()
        self.assertEqual(
            self.timeline,
            [
                ("record", 0, "communication"),
                ("wait", 0, "A"),
                ("sort", "A"),
                ("record", 1, "A"),
                ("wait", 1, "B"),
            ],
        )
        # The original native event owns tensors when stream recording is disabled.
        self.assertIs(overlap.event, self.events[0]())
        self.assertIsNotNone(self.events[0]())
        overlap.current_stream_wait(release_handle=True)
        self.assertIsNone(overlap.event)
        self.assertTrue(all(event() is None for event in self.events))

    def test_no_hook_does_not_record_another_event(self):
        overlap = self.EventOverlap(self.EventHandle())
        self.stream = "A"
        overlap.current_stream_wait()
        self.stream = "B"
        overlap.current_stream_wait(release_handle=True)
        self.assertEqual(
            self.timeline,
            [
                ("record", 0, "communication"),
                ("wait", 0, "A"),
                ("wait", 0, "B"),
            ],
        )
        self.assertIsNone(self.events[0]())

    def test_context_exit_releases_both_events_after_first_hook(self):
        overlap = self.EventOverlap(self.EventHandle())
        overlap.register_hook_after_wait(
            lambda: self.timeline.append(("sort", self.stream))
        )
        self.stream = "A"
        with overlap(release_handle=True):
            pass
        self.assertEqual(self.timeline[-2:], [("sort", "A"), ("record", 1, "A")])
        self.assertIsNone(overlap.event)
        self.assertTrue(all(event() is None for event in self.events))


if __name__ == "__main__":
    unittest.main()
