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

"""
NIXL Metadata Exchange Utilities

Helper functions for publishing and retrieving NIXL agent metadata and descriptors
via TCP server for metadata exchange.
"""

import time
import base64
import tcp_server
from nixl.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 9998
DEFAULT_TIMEOUT = 10.0


def publish_agent_metadata(agent, key, host=DEFAULT_SERVER_HOST, port=DEFAULT_SERVER_PORT):
    """
    Publish agent metadata to TCP server.
    
    Args:
        agent: NIXL agent instance
        key: Metadata key name
        host: TCP server host
        port: TCP server port
    """
    metadata = agent.get_agent_metadata()
    metadata_b64 = base64.b64encode(metadata).decode('utf-8')
    tcp_server.set_metadata(key, metadata_b64, host, port)


def retrieve_agent_metadata(agent, key, host=DEFAULT_SERVER_HOST, port=DEFAULT_SERVER_PORT, 
                           timeout=DEFAULT_TIMEOUT, role_name="process"):
    """
    Retrieve remote agent metadata and add to local agent.
    
    Args:
        agent: NIXL agent instance
        key: Metadata key name
        host: TCP server host
        port: TCP server port
        timeout: Timeout in seconds
        role_name: Name for logging (e.g., "initiator", "sender")
    
    Returns:
        Remote agent name (str) or None on failure
    """
    logger.info(f"[{role_name}] Waiting for {key}...")
    start_wait = time.time()
    metadata_b64 = None
    
    while not metadata_b64 and (time.time() - start_wait) < timeout:
        metadata_b64 = tcp_server.get_metadata(key, host, port)
        if not metadata_b64:
            time.sleep(0.1)
    
    if not metadata_b64:
        logger.error(f"[{role_name}] Timeout waiting for {key}")
        return None
    
    metadata = base64.b64decode(metadata_b64.encode('utf-8'))
    remote_name = agent.add_remote_agent(metadata)
    
    # Convert bytes to string if needed
    if isinstance(remote_name, bytes):
        remote_name = remote_name.decode('utf-8')
    
    logger.info(f"[{role_name}] Loaded remote agent: {remote_name}")
    return remote_name


def publish_descriptors(agent, xfer_descs, key, host=DEFAULT_SERVER_HOST, port=DEFAULT_SERVER_PORT):
    """
    Serialize and publish descriptors to TCP server.
    
    Args:
        agent: NIXL agent instance
        xfer_descs: Transfer descriptors to publish
        key: Metadata key name
        host: TCP server host
        port: TCP server port
    """
    serialized = agent.get_serialized_descs(xfer_descs)
    serialized_b64 = base64.b64encode(serialized).decode('utf-8')
    tcp_server.set_metadata(key, serialized_b64, host, port)


def retrieve_descriptors(agent, key, host=DEFAULT_SERVER_HOST, port=DEFAULT_SERVER_PORT):
    """
    Retrieve and deserialize descriptors from TCP server.
    
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
    
    serialized = base64.b64decode(serialized_b64.encode('utf-8'))
    return agent.deserialize_descs(serialized)

