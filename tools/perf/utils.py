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
import logging

import numpy as np

log = logging.getLogger(__name__)


def format_size(nbytes: int, precision=2) -> str:
    units = ["B", "K", "M", "G"]
    units_ix = 0
    while nbytes / 1024 >= 1 and units_ix < len(units) - 1:
        nbytes /= 1024
        units_ix += 1

    nbytes = round(nbytes, precision)
    return f"{nbytes:g}{units[units_ix]}"


def parse_size(nbytes: str) -> int:
    """Convert formatted string with unit to bytes"""
    options = {"g": 1024 * 1024 * 1024, "m": 1024 * 1024, "k": 1024}
    unit = 1
    key = nbytes[-1].lower()
    if key in options:
        unit = options[key]
        value = int(nbytes[:-1])
    else:
        value = int(nbytes)
    count = unit * value
    return count


def load_matrix(matrix_file: str) -> list[list[int]]:
    # Cell i,j of the matrix is the size of the message to send from process i to process j
    matrix = []
    with open(matrix_file, "r") as f:
        for line in f:
            row = line.strip().split()
            row = [parse_size(x) for x in row]
            matrix.append(row)
    mat = np.array(matrix)

    return mat
