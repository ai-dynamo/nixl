# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NIXL Memory Utilities

Helper functions for reading and writing data to memory addresses using NumPy.
These utilities provide a safe and efficient way to work with raw memory pointers.
"""

import ctypes

import numpy as np


def write_uint64(addr, value):
    """
    Write uint64 to local memory using NumPy.

    Args:
        addr: Memory address (integer)
        value: 64-bit unsigned integer value to write
    """
    # Create a NumPy view of the memory location (8 bytes for uint64)
    char_buffer = (ctypes.c_char * 8).from_address(addr)
    arr = np.ndarray((1,), dtype=np.uint64, buffer=char_buffer)
    arr[0] = value


def read_uint64(addr):
    """
    Read uint64 from local memory using NumPy.

    Args:
        addr: Memory address (integer)

    Returns:
        64-bit unsigned integer value
    """
    # Create a NumPy view of the memory location (8 bytes for uint64)
    char_buffer = (ctypes.c_char * 8).from_address(addr)
    arr = np.frombuffer(char_buffer, dtype=np.uint64, count=1)
    return int(arr[0])
