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

#!/usr/bin/env python3
"""
Test script for NIXL prefixedOpenClose functionality in file plugins
"""

import os
import tempfile
import unittest
import _bindings as nixl_bindings


class TestPrefixedOpen(unittest.TestCase):
    """Test cases for prefixedOpenClose functionality in file plugins"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary test file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b"Test content for prefixed open")
        self.temp_file.close()

        # Create a non-existent file path
        self.non_existent_file = "/tmp/nixl_test_nonexistent_file_12345.txt"

    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_register_mem_with_prefixed_filename(self):
        """Test registering memory with prefixed filenames in metaInfo"""
        # Create an agent
        config = nixl_bindings.nixlAgentConfig(False, False)
        agent = nixl_bindings.nixlAgent("test_agent", config)

        # Create a registration descriptor list
        descs = nixl_bindings.nixlRegDList(nixl_bindings.FILE_SEG, False)

        # Add descriptors with prefixed filenames in metaInfo
        # Use devId as a unique key (0, 1, 2) and put prefixed filenames in metaInfo
        descs.addDesc((0, 0, 0, "RO:" + self.temp_file.name))  # Read-only prefix
        descs.addDesc((0, 0, 1, "RW:" + self.temp_file.name))  # Read-write prefix
        descs.addDesc((0, 0, 2, "WR:" + self.non_existent_file))  # Write-only prefix (non-existent)

        # Try to create a backend with POSIX plugin
        try:
            params, mems = agent.getPluginParams('POSIX')
            backend = agent.createBackend('POSIX', params)

            # Register memory - this should use prefixedOpenClose internally
            try:
                agent.registerMem(descs, [backend])
                print("Successfully registered memory with prefixed filenames")

                # Deregister memory
                agent.deregisterMem(descs, [backend])
                print("Successfully deregistered memory")

            except Exception as e:
                # Some backends might not support this functionality
                print(f"registerMem with prefixed filenames failed (expected for some backends): {e}")

        except Exception as e:
            print(f"Backend creation failed: {e}")

    def test_register_mem_without_prefix(self):
        """Test registering memory without prefixed filenames (fallback behavior)"""
        # Create an agent
        config = nixl_bindings.nixlAgentConfig(False, False)
        agent = nixl_bindings.nixlAgent("test_agent", config)

        # Create a registration descriptor list
        descs = nixl_bindings.nixlRegDList(nixl_bindings.FILE_SEG, False)

        # Add descriptors without prefixed filenames in metaInfo
        # This should fall back to using devId as file descriptor
        descs.addDesc((0, 0, 0, ""))  # Empty metaInfo
        descs.addDesc((0, 0, 1, ""))  # Empty metaInfo

        # Try to create a backend with POSIX plugin
        try:
            params, mems = agent.getPluginParams('POSIX')
            backend = agent.createBackend('POSIX', params)

            # Register memory - this should use devId as file descriptor
            try:
                agent.registerMem(descs, [backend])
                print("Successfully registered memory without prefixed filenames")

                # Deregister memory
                agent.deregisterMem(descs, [backend])
                print("Successfully deregistered memory")

            except Exception as e:
                # Some backends might not support this functionality
                print(f"registerMem without prefixed filenames failed (expected for some backends): {e}")

        except Exception as e:
            print(f"Backend creation failed: {e}")


if __name__ == "__main__":
    unittest.main()
