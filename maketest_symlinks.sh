#!/bin/bash

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

ln -sf ./nixl/agent_example ./test/agent_example
ln -sf ./nixl/desc_example  ./test/desc_example
ln -sf ./unit/plugins/ucx/ucx_backend_test ./test/ucx_backend_test
ln -sf ./unit/plugins/ucx/ucx_backend_multi ./test/ucx_backend_multi
ln -sf ../../../test/unit/utils/serdes/serdes_test ./src/utils/serdes/serdes_test
