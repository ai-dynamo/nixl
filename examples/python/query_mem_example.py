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
    from nixl.logging import get_logger

    logger = get_logger(__name__)

    NIXL_AVAILABLE = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logger.error("NIXL API missing install NIXL.")
    NIXL_AVAILABLE = False

if __name__ == "__main__":
    logger.info("NIXL queryMem Python API Example")
    logger.info("=" * 40)

    if not NIXL_AVAILABLE:
        logger.warning("Skipping example - NIXL bindings not available")
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
        logger.info(
            "Using NIXL Plugins from: %s",
            os.environ.get("NIXL_PLUGIN_DIR", "default location"),
        )

        # Create an NIXL agent
        logger.info("Creating NIXL agent...")
        config = nixl_agent_config(
            enable_prog_thread=False, enable_listen_thread=False, backends=[]
        )
        agent = nixl_agent("example_agent", config)

        # Prepare a list of tuples as file paths in metaInfo field for querying.
        # Addr and length and devID fields are set to 0 for file queries.
        logger.info("Preparing file paths for querying...")
        file_paths = [
            (0, 0, 0, temp_files[0]),  # Existing file 1
            (0, 0, 0, temp_files[1]),  # Existing file 2
            (0, 0, 0, non_existent_file),  # Non-existent file
            (0, 0, 0, temp_files[2]),  # Existing file 3
        ]

        # Query memory using queryMem
        logger.info("Querying memory/storage information...")

        # Try to create a backend with POSIX plugin
        try:
            params = agent.get_plugin_params("POSIX")
            agent.create_backend("POSIX", params)
            logger.info("Created backend: POSIX")

            # Query with specific backend
            resp = agent.query_memory(file_paths, "POSIX", mem_type="FILE")
        except Exception as e:
            logger.exception("POSIX backend creation failed: %s", e)
            # Try MOCK_DRAM as fallback
            try:
                params = agent.get_plugin_params("MOCK_DRAM")
                agent.create_backend("MOCK_DRAM", params)
                logger.info("Created backend: MOCK_DRAM")

                # Query with specific backend
                resp = agent.query_memory(file_paths, "MOCK_DRAM", mem_type="FILE")
            except Exception as e2:
                logger.exception("MOCK_DRAM also failed: %s", e2)
                logger.exception("No working backends available")
                sys.exit(0)

        # Display results
        logger.info("\nQuery results (%d responses):", len(resp))
        logger.info("-" * 50)

        for i, result in enumerate(resp):
            logger.info("Descriptor %d:", i)
            if result is not None:
                logger.info("  File size: %s bytes", result.get("size", "N/A"))
                logger.info("  File mode: %s", result.get("mode", "N/A"))
                logger.info("  Modified time: %s", result.get("mtime", "N/A"))
            else:
                logger.info("  File does not exist or is not accessible")
            logger.info("")

        logger.info("Example completed successfully!")

    except Exception as e:
        logger.exception("Error in example: %s", e)
        import traceback

        traceback.print_exc()

    finally:
        # Clean up temporary files
        logger.info("Cleaning up temporary files...")
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.info("Removed: %s", temp_file_path)
