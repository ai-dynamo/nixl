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

install_script=git-pre-commit-format
formatter=apply-format

bash_source="${BASH_SOURCE[0]:-$0}"

top_dir=$(git -C "$git_test_dir" rev-parse --show-toplevel) || \
    error_exit "You need to be in the git repository to run this script."

hook_path="$top_dir/.git/hooks"

my_path=$(realpath "$bash_source") || exit 1
my_dir=$(dirname "$my_path") || exit 1

echo "Installing pre-commit hook"
$my_dir/$install_script install || exit 1

existing_formatter=$hook_path/$formatter
if [ ! -e "$existing_formatter" ]; then
    cp $my_dir/apply-format $hook_path/
    echo "Copied formatter script"
fi
