# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys

p = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, p + "/../")


def nixl_bench_option_names():
    from commands.args import nixl_bench_args

    def placeholder(**kwargs):
        pass

    return [param.name for param in nixl_bench_args(placeholder).__click_params__]


def test_every_nixlbench_option_is_carried_by_the_wrapper():
    """An option the wrapper does not name is dropped on the way to nixlbench.

    main.py keeps only the arguments NIXLBench.defaults() names:

        filtered_args = {
            k: v for k, v in kwargs.items() if k in NIXLBench.defaults() and v is not None
        }

    so an option added to the CLI alone is accepted, listed in --help, and then
    discarded without a word. --obj_ca_bundle sat in that state.
    """
    from commands.nixlbench import NIXLBench

    defaults = NIXLBench.defaults()
    missing = sorted(name for name in nixl_bench_option_names() if name not in defaults)

    assert missing == []
