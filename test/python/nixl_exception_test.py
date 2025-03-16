#!/usr/bin/env python3

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

import nixl._utils as nixl_utils
from nixl._api import nixl_agent
import nixl._bindings as nixl_bindings

if __name__ == "__main__":

    nixl_agent1 = nixl_agent("bad agent", None)

    buf_size = 256
    addr1 = nixl_utils.malloc_passthru(buf_size * 2)
    addr2 = addr1 + buf_size

    agent1_addrs = [(addr1, buf_size, 0), (addr2, buf_size, 0)]
    agent1_strings = [(addr1, buf_size, 0, "a"), (addr2, buf_size, 0, "b")]

    agent1_reg_descs = nixl_agent1.get_reg_descs(agent1_strings, "DRAM", True)

    agent1_xfer_descs = nixl_agent1.get_xfer_descs(agent1_addrs, "DRAM", True)

    try:
        #null backend not supported
        nixl_agent1.backends["UVX"] = 0
        nixl_agent1.register_memory(agent1_reg_descs, backend="UVX")
    except nixl_bindings.nixlNotSupportedError as e:
        print("Caught you!")
        print(e)
