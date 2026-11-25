# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

fs = import('fs')
nixl_init_copy = fs.copyfile('__init__.py')
if cuda_wheel_dir.startswith('nixl_rocm') and cuda_wheel_dir != 'nixl_rocm'
  # The type stub names the unversioned nixl_rocm package: point it at the
  # nixl_rocm<X>_<Y> package this build installs.
  nixl_api_copy = configure_file(
    input: '_api.py',
    output: '_api.py',
    capture: true,
    command: [
      'sed', 's/from nixl_rocm\\._api import/from ' + cuda_wheel_dir + '._api import/', '@INPUT@',
    ],
  )
else
  nixl_api_copy = fs.copyfile('_api.py')
endif
nixl_logging_copy = fs.copyfile('logging.py')
nixl_cute_copy = []
if get_option('build_cute')
  nixl_cute_copy = fs.copyfile('cute.py')
endif

subdir('runtime')
