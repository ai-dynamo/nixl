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

import os

from .etcd_rt import _EtcdDistUtils

# Initialize with default parameters, can be overridden by environment variables
etcd_endpoints = os.environ.get("ETCD_ENDPOINTS", "http://localhost:2379")
world_size = int(os.environ.get("WORLD_SIZE", "1"))

dist_rt = _EtcdDistUtils(etcd_endpoints=etcd_endpoints, size=world_size)
dist_rt.init_dist()
