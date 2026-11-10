# SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0

"""CPU-only checks of the Python NVFP4 boundary, with the native dispatch mocked."""
import unittest
from types import SimpleNamespace

from test_auto_sm_host import load_buffer


class Tensor:
    sizes = {"uint8": 1, "float8": 1, "int32": 4, "float16": 2}

    def __init__(self, shape, dtype, storage=None, contiguous=True):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.storage = bytearray(range(16)) if storage is None else storage
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def view(self, dtype):
        byte_width = self.shape[-1] * self.sizes[self.dtype]
        assert byte_width % self.sizes[dtype] == 0
        return Tensor(
            (*self.shape[:-1], byte_width // self.sizes[dtype]), dtype, self.storage
        )


class NVFP4HostTests(unittest.TestCase):
    def setUp(self):
        Buffer, _, _ = load_buffer()
        torch = Buffer.dispatch.__globals__["torch"]
        torch.uint8, torch.float8_e4m3fn, torch.int32 = "uint8", "float8", "int32"
        self.buffer = Buffer(
            num_bytes=2097152,
            num_max_tokens_per_rank=16,
            hidden=128,
            num_topk=2,
            max_num_scaleout_ranks=1,
            max_num_scaleup_ranks=4,
            store=object(),
            bootstrap_id="nvfp4-host",
            rank_idx=0,
            num_ranks=4,
            explicitly_destroy=True,
        )
        self.calls = []

        def dispatch(*args):
            self.calls.append(args)
            x, sf, topk = args[:3]
            return (
                x,
                sf,
                None,
                None,
                topk,
                2,
                [],
                object(),
                object(),
                object(),
                object(),
                object(),
                object(),
                None,
            )

        self.buffer.runtime.dispatch = dispatch
        self.codes = Tensor((2, 64), "uint8")
        self.topk = SimpleNamespace(shape=(2, 2))

    def send(self, scales):
        return self.buffer.dispatch((self.codes, scales), self.topk, 16, num_sms=2)[0]

    def test_scale_bytes_are_packed_and_returned_without_copy_or_cast(self):
        for dtype in ("uint8", "float8"):
            with self.subTest(dtype=dtype):
                scales = Tensor((2, 8), dtype)
                codes, returned = self.send(scales)
                packed = self.calls[-1][1]
                self.assertEqual((packed.dtype, packed.shape), ("int32", (2, 2)))
                self.assertEqual((returned.dtype, returned.shape), ("float8", (2, 8)))
                self.assertIs(packed.storage, scales.storage)
                self.assertIs(returned.storage, scales.storage)
                self.assertIs(codes, self.codes)

    def test_malformed_scales_are_refused_before_native_dispatch(self):
        for scales in (
            Tensor((2, 4), "uint8"),
            Tensor((2, 8), "int32"),
            Tensor((2, 8), "float16"),
            Tensor((2, 8), "uint8", contiguous=False),
        ):
            with self.subTest(
                shape=scales.shape, dtype=scales.dtype, contiguous=scales.contiguous
            ):
                with self.assertRaisesRegex(AssertionError, "NVFP4 requires"):
                    self.send(scales)
        self.assertEqual(self.calls, [])

    def test_packed_width_must_represent_a_multiple_of_64_elements(self):
        self.codes = Tensor((2, 48), "uint8")
        with self.assertRaisesRegex(AssertionError, "hidden divisible by 64"):
            self.send(Tensor((2, 6), "uint8"))
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
