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
import sys
import tempfile

try:
    from nixl._api import nixl_agent, nixl_agent_config

    NIXL_AVAILABLE = True
except ImportError:
    print("NIXL API missing install NIXL.")
    NIXL_AVAILABLE = False

if __name__ == "__main__":
    print("NIXL queryMem Python API Example")
    print("=" * 40)

    if not NIXL_AVAILABLE:
        print("Skipping example - NIXL bindings not available")
        sys.exit(0)

    # Create temporary test files
    os.makedirs("files_for_query", exist_ok=True)
    temp_files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(
            dir="files_for_query", delete=False, suffix=f"_{i}.txt", mode="wb"
        ) as temp_file:
            temp_file.write(f"Test content for file {i}".encode())
            temp_files.append(str(temp_file.name))

    # Create a non-existent file path
    non_existent_file = "/tmp/nixl_example_nonexistent.txt"

    try:
        print("Using NIXL Plugins from:")
        print(os.environ["NIXL_PLUGIN_DIR"])

        # Create an NIXL agent
        print("Creating NIXL agent...")
        config = nixl_agent_config(False, False, 0, [])
        agent = nixl_agent("example_agent", config)

        # Prepare a list of tuples as file paths in metaInfo field for querying.
        # Addr and length and devID fields are set to 0 for file queries.
        print("Preparing file paths for querying...")
        file_paths = [
            (0, 0, 0, temp_files[0]),  # Existing file 1
            (0, 0, 0, temp_files[1]),  # Existing file 2
            (0, 0, 0, non_existent_file),  # Non-existent file
            (0, 0, 0, temp_files[2]),  # Existing file 3
        ]

        # Query memory using queryMem
        print("Querying memory/storage information...")

        # Try to create a backend with POSIX plugin
        try:
            params = agent.get_plugin_params("POSIX")
            agent.create_backend("POSIX", params)
            print("Created backend: POSIX")

            # Query with specific backend
            resp = agent.query_memory(file_paths, "POSIX", mem_type="FILE")
        except Exception as e:
            print(f"POSIX backend creation failed: {e}")
            # Try MOCK_DRAM as fallback
            try:
                params = agent.get_plugin_params("MOCK_DRAM")
                agent.create_backend("MOCK_DRAM", params)
                print("Created backend: MOCK_DRAM")

                # Query with specific backend
                resp = agent.query_memory(file_paths, "MOCK_DRAM", mem_type="FILE")
            except Exception as e2:
                print(f"MOCK_DRAM also failed: {e2}")
                print("No working backends available")
                sys.exit(0)

        # Display results
        print(f"\nQuery results ({len(resp)} responses):")
        print("-" * 50)

        for i, result in enumerate(resp):
            print(f"Descriptor {i}:")
            if result is not None:
                print(f"  File size: {result.get('size', 'N/A')} bytes")
                print(f"  File mode: {result.get('mode', 'N/A')}")
                print(f"  Modified time: {result.get('mtime', 'N/A')}")
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
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                print(f"Removed: {temp_file_path}")
