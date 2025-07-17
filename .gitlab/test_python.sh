#!/bin/sh
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

source $(dirname $0)/../.ci/scripts/common.sh

set -e
set -x

INSTALL_DIR=$1

check_install_dir $INSTALL_DIR

# For running as user - check if running as root, if not set sudo variable
if [ "$(id -u)" -ne 0 ]; then
    SUDO=sudo
else
    SUDO=""
fi

$SUDO apt-get -qq install liburing-dev

set_env $INSTALL_DIR

pip3 install --break-system-packages .
pip3 install --break-system-packages pytest
pip3 install --break-system-packages pytest-timeout
pip3 install --break-system-packages zmq

echo "==== Running python tests ===="
python3 examples/python/nixl_api_example.py
pytest test/python
python3 test/python/prep_xfer_perf.py list
python3 test/python/prep_xfer_perf.py array

echo "==== Running python example ===="
blocking_send_recv_port=$(get_next_server_port)

cd examples/python
python3 blocking_send_recv_example.py --mode="target" --ip=127.0.0.1 --port=$blocking_send_recv_port&
sleep 5
python3 blocking_send_recv_example.py --mode="initiator" --ip=127.0.0.1 --port=$blocking_send_recv_port
python3 partial_md_example.py
