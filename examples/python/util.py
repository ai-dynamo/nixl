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


def wait_for_transfer_completion(init_agent, target_agent, xfer_handle, uuid):
    """Wait for both initiator and target to complete transfer."""
    target_done = False
    init_done = False

    while (not init_done) or (not target_done):
        if not init_done:
            state = init_agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                raise RuntimeError("Transfer got to Error state.")
            if state == "DONE":
                init_done = True
                print("Initiator done")

        if (not target_done) and target_agent.check_remote_xfer_done("initiator", uuid):
            target_done = True
            print("Target done")
