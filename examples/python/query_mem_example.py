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

import os
import tempfile
import sys

# Try to import the NIXL bindings
try:
    import _bindings as nixl_bindings
    NIXL_AVAILABLE = True
except ImportError:
    print("NIXL bindings not available. This example requires NIXL to be installed.")
    NIXL_AVAILABLE = False

if __name__ == "__main__":
    print("NIXL queryMem Python Bindings Example")
    print("=" * 40)

    if not NIXL_AVAILABLE:
        print("Skipping example - NIXL bindings not available")
        sys.exit(0)

    # Create temporary test files
    temp_files = []
    for i in range(3):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.txt")
        temp_file.write(f"Test content for file {i}".encode())
        temp_file.close()
        temp_files.append(temp_file.name)

    # Create a non-existent file path
    non_existent_file = "/tmp/nixl_example_nonexistent_file.txt"

    try:
        print("Using NIXL Plugins from:")
        print(os.environ["NIXL_PLUGIN_DIR"])

        # Create an NIXL agent
        print("Creating NIXL agent...")
        config = nixl_bindings.nixlAgentConfig(False, False)  # enable_prog_thread=False, enable_listen_thread=False
        agent = nixl_bindings.nixlAgent("example_agent", config)

        # Create a registration descriptor list
        print("Creating registration descriptor list...")
        descs = nixl_bindings.nixlRegDList(nixl_bindings.FILE_SEG, False)

        # Add descriptors with file paths in metaInfo
        # Note: For file backends, the metaInfo field contains the file path
        descs.addDesc((0, 0, 0, temp_files[0]))  # Existing file 1
        descs.addDesc((0, 0, 0, temp_files[1]))  # Existing file 2
        descs.addDesc((0, 0, 0, temp_files[2]))  # Existing file 3
        descs.addDesc((0, 0, 0, non_existent_file))  # Non-existent file

        print(f"Added {descs.descCount()} descriptors to query")

        # Query memory using queryMem
        print("Querying memory/storage information...")

        # Try to create a backend with POSIX plugin
        try:
            params, mems = agent.getPluginParams('POSIX')
            backend = agent.createBackend('POSIX', params)
            print(f"Created backend: POSIX")

            # Query with specific backend
            resp = agent.queryMem(descs, [backend])
        except Exception as e:
            print(f"POSIX backend creation failed: {e}")
            # Try MOCK_DRAM as fallback
            try:
                params, mems = agent.getPluginParams('MOCK_DRAM')
                backend = agent.createBackend('MOCK_DRAM', params)
                print(f"Created backend: MOCK_DRAM")

                # Query with specific backend
                resp = agent.queryMem(descs, [backend])
            except Exception as e2:
                print(f"MOCK_DRAM also failed: {e2}")
                print("No working backends available")
                sys.exit(0)

        # Display results
        print(f"\nQuery results ({len(resp)} responses):")
        print("-" * 50)

        for i, result in enumerate(resp):
            print(f"Descriptor {i}:")
            print(f"  Accessible: {result.accessible}")
            if result.accessible:
                print(f"  File size: {result.info.get('size', 'N/A')} bytes")
                print(f"  File mode: {result.info.get('mode', 'N/A')}")
                print(f"  Modified time: {result.info.get('mtime', 'N/A')}")
            else:
                print("  File does not exist or is not accessible")
            print()

        print("Example completed successfully!")

    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                print(f"Removed: {temp_file}")
