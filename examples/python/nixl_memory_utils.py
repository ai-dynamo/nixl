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


def write_uint8(addr, value):
    """
    Write uint8 (1 byte) to local memory using ctypes.

    Args:
        addr: Memory address (integer)
        value: 8-bit unsigned integer value to write (0-255)
    """
    byte_value = ctypes.c_uint8(value)
    ctypes.memmove(addr, ctypes.addressof(byte_value), 1)


def read_uint8(addr):
    """
    Read uint8 (1 byte) from local memory using ctypes.

    Args:
        addr: Memory address (integer)

    Returns:
        8-bit unsigned integer value (0-255)
    """
    byte_value = ctypes.c_uint8()
    ctypes.memmove(ctypes.addressof(byte_value), addr, 1)
    return byte_value.value


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


def write_data(addr, data):
    """
    Write data to local memory using NumPy.

    Args:
        addr: Memory address (integer)
        data: Data to write (NumPy array, bytes, or bytearray)

    Raises:
        TypeError: If data type is not supported
    """
    # Convert data to NumPy array if needed
    if isinstance(data, np.ndarray):
        src_arr = data
    elif isinstance(data, (bytes, bytearray)):
        src_arr = np.frombuffer(data, dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Create a NumPy view of the destination and copy
    char_buffer = (ctypes.c_char * len(src_arr)).from_address(addr)
    dst_arr = np.ndarray(src_arr.shape, dtype=np.uint8, buffer=char_buffer)
    np.copyto(dst_arr, src_arr)


def read_data(addr, size):
    """
    Read data from local memory using NumPy.

    Args:
        addr: Memory address (integer)
        size: Number of bytes to read

    Returns:
        NumPy array containing the read data
    """
    # Create a NumPy view and return a copy
    char_buffer = (ctypes.c_char * size).from_address(addr)
    arr = np.ndarray((size,), dtype=np.uint8, buffer=char_buffer)
    return arr.copy()
