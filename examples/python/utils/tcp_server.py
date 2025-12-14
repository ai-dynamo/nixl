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

"""
Simple TCP server for NIXL metadata exchange.

Provides a lightweight key-value store for exchanging metadata between processes.
This is used when NIXL's built-in etcd support is not available.
"""

import json
import socket
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Lock, Thread

from nixl.logging import get_logger

logger = get_logger(__name__)

# --- Server state ---
_metadata = {}
_lock = Lock()


# --- Request handler ---
class MetadataHandler(StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline().strip().decode()
            request = json.loads(line)

            cmd = request.get("cmd")

            if cmd == "SET":
                # Store metadata: {"cmd": "SET", "key": "...", "value": ...}
                key = request.get("key")
                value = request.get("value")
                with _lock:
                    _metadata[key] = value
                response = {"status": "OK"}

            elif cmd == "GET":
                # Retrieve metadata: {"cmd": "GET", "key": "..."}
                key = request.get("key")
                with _lock:
                    value = _metadata.get(key)
                if value is not None:
                    response = {"status": "OK", "value": value}
                else:
                    response = {"status": "ERROR", "message": f"Key '{key}' not found"}

            elif cmd == "CLEAR":
                # Clear all metadata: {"cmd": "CLEAR"}
                with _lock:
                    _metadata.clear()
                response = {"status": "OK"}

            else:
                response = {"status": "ERROR", "message": f"Unknown command: {cmd}"}

            self.wfile.write((json.dumps(response) + "\n").encode())

        except json.JSONDecodeError:
            error = {"status": "ERROR", "message": "Invalid JSON"}
            self.wfile.write((json.dumps(error) + "\n").encode())
        except Exception as e:
            error = {"status": "ERROR", "message": str(e)}
            self.wfile.write((json.dumps(error) + "\n").encode())


# --- TCPServer subclass to reuse port immediately ---
class ReusableTCPServer(ThreadingTCPServer):
    allow_reuse_address = True


# --- Singleton server instance ---
_server_instance = None
_server_lock = Lock()


def start_server(port=9998):
    """Start the metadata server (singleton pattern)"""
    global _server_instance
    with _server_lock:
        if _server_instance is None:
            try:
                _server_instance = ReusableTCPServer(("0.0.0.0", port), MetadataHandler)
                Thread(target=_server_instance.serve_forever, daemon=True).start()
                return True
            except OSError:
                # Server already running (possibly from another process)
                return False
        return False


# --- Client API ---


def set_metadata(key, value, server="127.0.0.1", port=9998):
    """Set a metadata key-value pair"""
    s = socket.create_connection((server, port))
    request = {"cmd": "SET", "key": key, "value": value}
    s.sendall((json.dumps(request) + "\n").encode())
    response = json.loads(s.recv(4096).decode().strip())
    s.close()
    return response.get("status") == "OK"


def get_metadata(key, server="127.0.0.1", port=9998):
    """Get a metadata value by key"""
    s = socket.create_connection((server, port))
    request = {"cmd": "GET", "key": key}
    s.sendall((json.dumps(request) + "\n").encode())
    response = json.loads(s.recv(4096).decode().strip())
    s.close()
    if response.get("status") == "OK":
        return response.get("value")
    return None


def clear_metadata(server="127.0.0.1", port=9998):
    """Clear all metadata"""
    s = socket.create_connection((server, port))
    request = {"cmd": "CLEAR"}
    s.sendall((json.dumps(request) + "\n").encode())
    response = json.loads(s.recv(4096).decode().strip())
    s.close()
    return response.get("status") == "OK"

