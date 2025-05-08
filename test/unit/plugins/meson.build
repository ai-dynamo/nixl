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

subdir('ucx')
subdir('ucx_mo')
subdir('posix')

disable_gds_backend = get_option('disable_gds_backend')
if not disable_gds_backend
    subdir('cuda_gds')
endif

cc = meson.get_compiler('cpp')
libtransfer_engine = cc.find_library('transfer_engine', required: false)
disable_mooncake_backend = get_option('disable_mooncake_backend')
if libtransfer_engine.found() and not disable_mooncake_backend
    subdir('mooncake')
endif