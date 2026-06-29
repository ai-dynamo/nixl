# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
NIXL Metadata Exchange Utilities

Helper functions for publishing and retrieving NIXL agent metadata and descriptors.

Supports two exchange modes:
1. TCP server (default): Uses a simple TCP key-value server for local testing
2. NIXL built-in: Uses NIXL's native metadata APIs (etcd or socket-based)

To use NIXL's built-in etcd support:
- Set NIXL_ETCD_ENDPOINTS environment variable (e.g., "http://localhost:2379")
- Pass use_nixl_builtin=True to the functions
"""

import base64
import os
import time

from nixl.logging import get_logger

from . import tcp_server

logger = get_logger(__name__)

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 9998
DEFAULT_TIMEOUT = 10.0


def _is_etcd_available():
    """Check if NIXL etcd support is available (via environment variable)."""
    return bool(os.environ.get("NIXL_ETCD_ENDPOINTS"))


def publish_agent_metadata(
    agent,
    key,
    host=DEFAULT_SERVER_HOST,
    port=DEFAULT_SERVER_PORT,
    use_nixl_builtin=False,
    reg_descs=None,
):
    """
    Publish agent metadata for remote agents to discover.

    Args:
        agent: NIXL agent instance
        key: Metadata key/label name
        host: TCP server host (ignored if use_nixl_builtin=True)
        port: TCP server port (ignored if use_nixl_builtin=True)
        use_nixl_builtin: If True, use NIXL's built-in metadata exchange (etcd/socket)
        reg_descs: Registration descriptors (required for use_nixl_builtin=True)
    """
    if use_nixl_builtin:
        # Use NIXL's built-in metadata APIs
        # This will use etcd if NIXL_ETCD_ENDPOINTS is set
        if reg_descs is None:
            logger.error("reg_descs required for NIXL built-in metadata exchange")
            return
        try:
            agent.send_partial_agent_metadata(
                reg_descs, inc_conn_info=True, backends=["UCX"], label=key
            )
            logger.debug(f"Published metadata via NIXL built-in API with label '{key}'")
        except Exception as e:
            if "NOT_SUPPORTED" in str(e):
                logger.error(
                    "NIXL was built without etcd support. Use TCP mode instead (remove --use-etcd)"
                )
            raise
    else:
        # Use TCP server
        metadata = agent.get_agent_metadata()
        metadata_b64 = base64.b64encode(metadata).decode("utf-8")
        tcp_server.set_metadata(key, metadata_b64, host, port)
        logger.debug(f"Published metadata to TCP server with key '{key}'")


def retrieve_agent_metadata(
    agent,
    key,
    host=DEFAULT_SERVER_HOST,
    port=DEFAULT_SERVER_PORT,
    timeout=DEFAULT_TIMEOUT,
    role_name="process",
    use_nixl_builtin=False,
    remote_agent_name=None,
):
    """
    Retrieve remote agent metadata and add to local agent.

    Args:
        agent: NIXL agent instance
        key: Metadata key/label name
        host: TCP server host (ignored if use_nixl_builtin=True)
        port: TCP server port (ignored if use_nixl_builtin=True)
        timeout: Timeout in seconds
        role_name: Name for logging (e.g., "initiator", "sender")
        use_nixl_builtin: If True, use NIXL's built-in metadata exchange (etcd/socket)
        remote_agent_name: Required when use_nixl_builtin=True - name of remote agent to fetch

    Returns:
        Remote agent name (str) or None on failure
    """
    logger.info(f"[{role_name}] Waiting for {key}...")

    if use_nixl_builtin:
        if not remote_agent_name:
            logger.error(
                f"[{role_name}] remote_agent_name required for NIXL built-in mode"
            )
            return None

        # Use NIXL's built-in fetch_remote_metadata
        # This will fetch from etcd if NIXL_ETCD_ENDPOINTS is set
        try:
            agent.fetch_remote_metadata(remote_agent_name, label=key)
        except Exception as e:
            if "NOT_SUPPORTED" in str(e):
                logger.error(
                    "NIXL was built without etcd support. Use TCP mode instead (remove --use-etcd)"
                )
            raise

        # Wait for metadata to be available
        start_wait = time.time()
        while (time.time() - start_wait) < timeout:
            if agent.check_remote_metadata(remote_agent_name):
                logger.info(f"[{role_name}] Loaded remote agent: {remote_agent_name}")
                return remote_agent_name
            time.sleep(0.1)

        logger.error(f"[{role_name}] Timeout waiting for {key}")
        return None
    else:
        # Use TCP server
        start_wait = time.time()
        metadata_b64 = None

        while not metadata_b64 and (time.time() - start_wait) < timeout:
            metadata_b64 = tcp_server.get_metadata(key, host, port)
            if not metadata_b64:
                time.sleep(0.1)

        if not metadata_b64:
            logger.error(f"[{role_name}] Timeout waiting for {key}")
            return None

        metadata = base64.b64decode(metadata_b64.encode("utf-8"))
        remote_name = agent.add_remote_agent(metadata)

        # Convert bytes to string if needed
        if isinstance(remote_name, bytes):
            remote_name = remote_name.decode("utf-8")

        logger.info(f"[{role_name}] Loaded remote agent: {remote_name}")
        return remote_name


def publish_descriptors(
    agent,
    xfer_descs,
    key,
    host=DEFAULT_SERVER_HOST,
    port=DEFAULT_SERVER_PORT,
):
    """
    Serialize and publish descriptors to TCP server.

    Note: This function only supports TCP server mode. For NIXL built-in
    metadata exchange, use send_partial_agent_metadata() which includes
    descriptors in the metadata.

    Args:
        agent: NIXL agent instance
        xfer_descs: Transfer descriptors to publish
        key: Metadata key name
        host: TCP server host
        port: TCP server port
    """
    serialized = agent.get_serialized_descs(xfer_descs)
    serialized_b64 = base64.b64encode(serialized).decode("utf-8")
    tcp_server.set_metadata(key, serialized_b64, host, port)


def retrieve_descriptors(
    agent,
    key,
    host=DEFAULT_SERVER_HOST,
    port=DEFAULT_SERVER_PORT,
):
    """
    Retrieve and deserialize descriptors from TCP server.

    Note: This function only supports TCP server mode. For NIXL built-in
    metadata exchange, descriptors are included in the metadata fetched
    via fetch_remote_metadata().

    Args:
        agent: NIXL agent instance
        key: Metadata key name
        host: TCP server host
        port: TCP server port

    Returns:
        Deserialized descriptors or None on failure
    """
    serialized_b64 = tcp_server.get_metadata(key, host, port)
    if not serialized_b64:
        return None

    serialized = base64.b64decode(serialized_b64.encode("utf-8"))
    return agent.deserialize_descs(serialized)
