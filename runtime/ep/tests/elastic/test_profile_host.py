# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""CPU-only checks of the diagnostic timestamp summary; no Torch import or GPU is needed."""
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "nixl_ep_v2/utils/profiling.py"
spec = importlib.util.spec_from_file_location("profile_summary", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Samples:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class ProfileSummaryTests(unittest.TestCase):
    def test_sparse_writers_and_independent_operation_origins(self):
        result = module.summarize_profile(
            {
                "combine.reduce_submitted": Samples([[12500, 0], [13000, 0]]),
                "dispatch.entry": Samples([[1000, 2000], [0, 0]]),
                "dispatch.count_prefix": Samples([[4000, 5000], [0, 0]]),
                "dispatch.forward_submitted": Samples([[0, 0], [0, 0]]),
                "combine.entry": Samples([[10000, 11000], [0, 0]]),
            }
        )
        self.assertEqual(
            result["dispatch.entry"], dict(first_us=0, last_us=1, writers=2)
        )
        self.assertEqual(
            result["dispatch.count_prefix"], dict(first_us=3, last_us=4, writers=2)
        )
        self.assertEqual(
            result["combine.reduce_submitted"], dict(first_us=2.5, last_us=3, writers=2)
        )
        self.assertNotIn("dispatch.forward_submitted", result)

    def test_cleared_profile(self):
        self.assertEqual(
            module.summarize_profile({"dispatch.entry": Samples([[0, 0]])}), {}
        )

    def test_missing_entry_is_not_reported_as_a_latency(self):
        with self.assertRaisesRegex(
            ValueError, r"combine\.main_end.*combine\.entry.*record one pair"
        ):
            module.summarize_profile({"combine.main_end": Samples([[1000]])})


if __name__ == "__main__":
    unittest.main()
