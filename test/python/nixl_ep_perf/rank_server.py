# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Rank server for multi-process test coordination.

Provides rank assignment and distributed barriers.
"""

import multiprocessing as mp
import os
import socket
import time
from collections import defaultdict
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Lock
from typing import Any, Dict, Optional, Set, Tuple


class RankServerHandler(StreamRequestHandler):
    """Handles GET_RANK, RELEASE_RANK, BARRIER, CLEAR_BARRIERS, RESET."""

    _lock: Lock = Lock()
    _counts: Dict[str, list] = defaultdict(list)
    _rank_to_host: Dict[int, Tuple[str, int]] = {}
    _all_global_ranks: Set[int] = set()
    _removed_global_ranks: Set[int] = set()
    _barriers: Dict[str, Dict[str, Any]] = {}
    _completed_barriers: Set[str] = set()

    def handle(self):
        try:
            line = self.rfile.readline().strip().decode()

            with self._lock:
                if line.startswith("BARRIER"):
                    self._handle_barrier(line)
                elif line.startswith("RELEASE_RANK"):
                    self._handle_release(line)
                elif line.startswith("CLEAR_BARRIERS"):
                    self._handle_clear_barriers()
                elif line.startswith("RESET"):
                    self._handle_reset()
                elif line.startswith("GET_RANK") or line:
                    self._handle_get_rank(line)

        except Exception as e:
            try:
                self.wfile.write(f"ERROR: {e}\n".encode())
            except Exception:
                pass

    def _handle_barrier(self, line: str):
        """BARRIER <barrier_id> <rank> <world_size>"""
        parts = line.split()
        if len(parts) < 4:
            self.wfile.write(b"ERROR: BARRIER requires barrier_id rank world_size\n")
            return

        barrier_id, rank, world_size = parts[1], int(parts[2]), int(parts[3])

        if barrier_id in self._completed_barriers:
            self.wfile.write(b"BARRIER_DONE\n")
            return

        if barrier_id not in self._barriers:
            self._barriers[barrier_id] = {"expected": world_size, "arrived": set()}

        self._barriers[barrier_id]["arrived"].add(rank)

        if len(self._barriers[barrier_id]["arrived"]) >= world_size:
            self._completed_barriers.add(barrier_id)
            del self._barriers[barrier_id]
            self.wfile.write(b"BARRIER_DONE\n")
        else:
            arrived = len(self._barriers[barrier_id]["arrived"])
            self.wfile.write(f"BARRIER_WAIT {arrived}/{world_size}\n".encode())

    def _handle_release(self, line: str):
        """RELEASE_RANK <rank>"""
        parts = line.split()
        if len(parts) < 2:
            self.wfile.write(b"ERROR: RELEASE_RANK requires rank\n")
            return

        rank = int(parts[1])
        if rank in self._all_global_ranks:
            self._all_global_ranks.discard(rank)
            self._removed_global_ranks.add(rank)
            if rank in self._rank_to_host:
                host, local_rank = self._rank_to_host[rank]
                if local_rank in self._counts[host]:
                    self._counts[host].remove(local_rank)
                del self._rank_to_host[rank]
        self.wfile.write(b"OK\n")

    def _handle_clear_barriers(self):
        """CLEAR_BARRIERS"""
        count = len(self._barriers) + len(self._completed_barriers)
        self._barriers.clear()
        self._completed_barriers.clear()
        self.wfile.write(f"OK {count}\n".encode())

    def _handle_reset(self):
        """RESET"""
        self._counts.clear()
        self._rank_to_host.clear()
        self._all_global_ranks.clear()
        self._removed_global_ranks.clear()
        self._barriers.clear()
        self._completed_barriers.clear()
        self.wfile.write(b"OK\n")

    def _handle_get_rank(self, line: str):
        """GET_RANK [hostname]"""
        if line.startswith("GET_RANK"):
            parts = line.split(maxsplit=1)
            host = parts[1] if len(parts) > 1 else os.uname().nodename
        else:
            host = line if line else os.uname().nodename

        used = set(self._counts[host])
        local = 0
        while local in used:
            local += 1
        self._counts[host].append(local)

        if self._removed_global_ranks:
            global_rank = min(self._removed_global_ranks)
            self._removed_global_ranks.remove(global_rank)
        else:
            global_rank = len(self._all_global_ranks)

        self._all_global_ranks.add(global_rank)
        self._rank_to_host[global_rank] = (host, local)
        self.wfile.write(f"{local} {global_rank}\n".encode())


class ReusableTCPServer(ThreadingTCPServer):
    """TCP server with port reuse."""

    allow_reuse_address = True
    daemon_threads = True


class RankClient:
    """Client for rank server communication."""

    def __init__(self, server: str = "127.0.0.1", port: int = 9998):
        self.server = server
        self.port = port
        self.global_rank: Optional[int] = None
        self.local_rank: Optional[int] = None

    def wait_for_server(
        self, timeout: float = 60.0, poll_interval: float = 0.5
    ) -> bool:
        """Wait for the rank server to be ready.

        Polls the server until it responds, with exponential backoff.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: Initial interval between connection attempts

        Returns:
            True if server is ready, raises TimeoutError otherwise
        """
        import logging

        logger = logging.getLogger(__name__)

        start_time = time.time()
        attempt = 0
        current_interval = poll_interval

        while time.time() - start_time < timeout:
            attempt += 1
            try:
                # Try to connect and send a simple command
                s = socket.create_connection((self.server, self.port), timeout=2.0)
                s.close()
                logger.info(
                    f"Rank server at {self.server}:{self.port} is ready "
                    f"(attempt {attempt}, waited {time.time() - start_time:.1f}s)"
                )
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                if attempt == 1:
                    logger.info(
                        f"Waiting for rank server at {self.server}:{self.port}..."
                    )
                elif attempt % 10 == 0:
                    logger.info(
                        f"Still waiting for rank server... "
                        f"(attempt {attempt}, {time.time() - start_time:.1f}s)"
                    )
                time.sleep(current_interval)
                # Exponential backoff up to 2 seconds
                current_interval = min(current_interval * 1.2, 2.0)

        raise TimeoutError(
            f"Rank server at {self.server}:{self.port} not ready after {timeout}s"
        )

    def _send(self, command: str, timeout: float = 10.0) -> str:
        """Send command and return response."""
        s = socket.create_connection((self.server, self.port), timeout=timeout)
        try:
            s.sendall(f"{command}\n".encode())
            return s.recv(4096).decode().strip()
        finally:
            s.close()

    def get_rank(self) -> Tuple[int, int]:
        """Get (local_rank, global_rank) from server."""
        if self.global_rank is not None and self.local_rank is not None:
            return (self.local_rank, self.global_rank)

        response = self._send(f"GET_RANK {os.uname().nodename}")
        parts = response.split()
        if len(parts) >= 2:
            self.local_rank = int(parts[0])
            self.global_rank = int(parts[1])
            return (self.local_rank, self.global_rank)
        raise RuntimeError(f"Unexpected response: {response}")

    def release_rank(self) -> bool:
        """Release assigned rank."""
        if self.global_rank is None:
            return True
        response = self._send(f"RELEASE_RANK {self.global_rank}")
        self.global_rank = None
        self.local_rank = None
        return response == "OK"

    def barrier_wait(
        self, barrier_id: str, rank: int, world_size: int, timeout: float = 60.0
    ) -> bool:
        """Wait at barrier until all ranks arrive."""
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                response = self._send(
                    f"BARRIER {barrier_id} {rank} {world_size}",
                    timeout=min(5.0, deadline - time.time()),
                )
                if response == "BARRIER_DONE":
                    return True
                elif response.startswith("BARRIER_WAIT"):
                    time.sleep(0.05)
                elif response.startswith("ERROR"):
                    raise RuntimeError(f"Barrier error: {response}")
            except socket.timeout:
                continue
            except ConnectionRefusedError:
                time.sleep(0.1)
                continue

        raise TimeoutError(f"Barrier {barrier_id} timeout after {timeout}s")

    def reset(self) -> bool:
        """Reset server state."""
        return self._send("RESET") == "OK"

    def clear_barriers(self) -> int:
        """Clear pending barriers."""
        response = self._send("CLEAR_BARRIERS")
        if response.startswith("OK"):
            parts = response.split()
            return int(parts[1]) if len(parts) > 1 else 0
        return 0


def start_server(port: int = 9998) -> mp.Process:
    """Start rank server in background process."""

    def run():
        try:
            server = ReusableTCPServer(("0.0.0.0", port), RankServerHandler)
            server.serve_forever()
        except OSError:
            pass  # Already running

    process = mp.Process(target=run, daemon=True)
    process.start()
    time.sleep(0.5)
    return process
