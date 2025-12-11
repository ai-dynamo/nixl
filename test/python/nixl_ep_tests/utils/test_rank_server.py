# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Rank Server - Coordination server for multi-node tests.

Provides rank assignment, distributed barriers, and state reset.
"""

import argparse
import multiprocessing as mp
import os
import socket
import sys
import time
from collections import defaultdict
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Lock
from typing import Optional, Tuple

# ============================================================================
# Server Implementation
# ============================================================================


class RankServerHandler(StreamRequestHandler):
    """Handles GET_RANK, RELEASE_RANK, BARRIER, CLEAR_BARRIERS, RESET commands."""

    # Shared state
    _lock = Lock()
    _counts = defaultdict(list)  # host -> [local_ranks]
    _rank_to_host = {}  # global_rank -> (host, local_rank)
    _all_global_ranks = set()
    _removed_global_ranks = set()
    _barriers = {}  # barrier_id -> {expected: int, arrived: set()}
    _completed_barriers = set()  # barrier_ids that completed (so all ranks get DONE)

    @classmethod
    def reset_state(cls):
        """Reset all server state (for test isolation)."""
        with cls._lock:
            cls._counts.clear()
            cls._rank_to_host.clear()
            cls._all_global_ranks.clear()
            cls._removed_global_ranks.clear()
            cls._barriers.clear()
            cls._completed_barriers.clear()

    def handle(self):
        try:
            # Read line OUTSIDE the lock (can block on network I/O)
            line = self.rfile.readline().strip().decode()

            # Acquire lock only for state manipulation
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
                self.wfile.write(f"ERROR: {str(e)}\n".encode())
            except:
                pass  # Client may have disconnected

    def _handle_barrier(self, line: str):
        """Handle barrier synchronization: BARRIER <barrier_id> <rank> <world_size>"""
        parts = line.split()
        if len(parts) < 4:
            self.wfile.write(
                "ERROR: BARRIER requires barrier_id rank world_size\n".encode()
            )
            return

        barrier_id = parts[1]
        rank = int(parts[2])
        world_size = int(parts[3])

        # Check if this barrier was already completed (all ranks need to get DONE)
        if barrier_id in self._completed_barriers:
            self.wfile.write("BARRIER_DONE\n".encode())
            return

        # Initialize barrier if needed
        if barrier_id not in self._barriers:
            self._barriers[barrier_id] = {"expected": world_size, "arrived": set()}

        # Record this rank's arrival
        self._barriers[barrier_id]["arrived"].add(rank)

        # Check if all ranks have arrived
        if len(self._barriers[barrier_id]["arrived"]) >= world_size:
            # All arrived - mark as complete (don't delete yet, other ranks need DONE too)
            self._completed_barriers.add(barrier_id)
            del self._barriers[barrier_id]
            self.wfile.write("BARRIER_DONE\n".encode())
        else:
            # Not all arrived yet - respond with wait count
            arrived = len(self._barriers[barrier_id]["arrived"])
            self.wfile.write(f"BARRIER_WAIT {arrived}/{world_size}\n".encode())

    def _handle_release(self, line: str):
        """Handle rank release: RELEASE_RANK <rank>"""
        parts = line.split()
        if len(parts) < 2:
            self.wfile.write("ERROR: RELEASE_RANK requires rank\n".encode())
            return

        rank_to_release = int(parts[1])

        if rank_to_release in self._all_global_ranks:
            self._all_global_ranks.discard(rank_to_release)
            self._removed_global_ranks.add(rank_to_release)

            if rank_to_release in self._rank_to_host:
                host, local_rank = self._rank_to_host[rank_to_release]
                if local_rank in self._counts[host]:
                    self._counts[host].remove(local_rank)
                del self._rank_to_host[rank_to_release]

            self.wfile.write("OK\n".encode())
        else:
            self.wfile.write("OK\n".encode())  # Idempotent

    def _handle_clear_barriers(self):
        """Handle barrier cleanup: CLEAR_BARRIERS"""
        count = len(self._barriers) + len(self._completed_barriers)
        self._barriers.clear()
        self._completed_barriers.clear()
        self.wfile.write(f"OK {count}\n".encode())

    def _handle_reset(self):
        """Handle state reset: RESET"""
        self._counts.clear()
        self._rank_to_host.clear()
        self._all_global_ranks.clear()
        self._removed_global_ranks.clear()
        self._barriers.clear()
        self._completed_barriers.clear()
        self.wfile.write("OK\n".encode())

    def _handle_get_rank(self, line: str):
        """Handle rank assignment: GET_RANK or just hostname"""
        # Extract hostname (either "GET_RANK hostname" or just "hostname")
        if line.startswith("GET_RANK"):
            parts = line.split(maxsplit=1)
            host = parts[1] if len(parts) > 1 else os.uname().nodename
        else:
            host = line if line else os.uname().nodename

        # Find the lowest unused local rank for this host
        used_ranks = set(self._counts[host])
        local = 0
        while local in used_ranks:
            local += 1

        # Add this local rank to the used list
        self._counts[host].append(local)

        # Assign global rank (reuse removed ranks if available)
        if self._removed_global_ranks:
            global_rank = min(self._removed_global_ranks)
            self._removed_global_ranks.remove(global_rank)
        else:
            global_rank = len(self._all_global_ranks)

        self._all_global_ranks.add(global_rank)
        self._rank_to_host[global_rank] = (host, local)

        self.wfile.write(f"{local} {global_rank}\n".encode())


class ReusableTCPServer(ThreadingTCPServer):
    """TCP server that allows port reuse."""

    allow_reuse_address = True
    daemon_threads = True


class RankServer:
    """TCP server for rank assignment and barrier coordination."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9998):
        self.host = host
        self.port = port
        self.server = None
        self.process = None

    def start(self, background: bool = True):
        """Start the server."""
        if background:
            self.process = mp.Process(target=self._run_server, daemon=True)
            self.process.start()
            time.sleep(0.5)  # Give server time to start
        else:
            self._run_server()

    def _run_server(self):
        """Run the server (blocking)."""
        try:
            self.server = ReusableTCPServer((self.host, self.port), RankServerHandler)
            self.server.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e):
                pass  # Another process already started the server
            else:
                raise

    def stop(self):
        """Stop the server."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.server:
            self.server.shutdown()

    def reset(self):
        """Reset server state."""
        client = RankClient(
            server=self.host if self.host != "0.0.0.0" else "127.0.0.1", port=self.port
        )
        client.reset()


# ============================================================================
# Client Implementation
# ============================================================================


class RankClient:
    """Client for test rank server communication."""

    def __init__(self, server: str = "127.0.0.1", port: int = 9998):
        self.server = server
        self.port = port
        self.global_rank: Optional[int] = None
        self.local_rank: Optional[int] = None

    def _send_command(self, command: str, timeout: float = 10.0) -> str:
        """Send a command to the server and return the response."""
        s = socket.create_connection((self.server, self.port), timeout=timeout)
        try:
            s.sendall(f"{command}\n".encode())
            response = s.recv(4096).decode().strip()
            return response
        finally:
            s.close()

    def get_rank(self) -> Tuple[int, int]:
        """
        Get rank assignment from server.

        Returns:
            (local_rank, global_rank)
        """
        if self.global_rank is not None:
            return self.local_rank, self.global_rank

        hostname = os.uname().nodename
        response = self._send_command(f"GET_RANK {hostname}")

        parts = response.split()
        if len(parts) >= 2:
            self.local_rank = int(parts[0])
            self.global_rank = int(parts[1])
            return self.local_rank, self.global_rank
        else:
            raise RuntimeError(f"Unexpected response from server: {response}")

    def release_rank(self) -> bool:
        """Release the assigned rank."""
        if self.global_rank is None:
            return True

        response = self._send_command(f"RELEASE_RANK {self.global_rank}")
        self.global_rank = None
        self.local_rank = None
        return response == "OK"

    def barrier_wait(
        self, barrier_id: str, rank: int, world_size: int, timeout: float = 60.0
    ) -> bool:
        """
        Wait at a barrier until all ranks arrive.

        Args:
            barrier_id: Unique identifier for this barrier
            rank: This process's rank
            world_size: Total number of ranks expected
            timeout: Maximum time to wait in seconds

        Returns:
            True if barrier completed

        Raises:
            TimeoutError if timeout exceeded
        """
        deadline = time.time() + timeout
        poll_interval = 0.05

        while time.time() < deadline:
            try:
                response = self._send_command(
                    f"BARRIER {barrier_id} {rank} {world_size}",
                    timeout=min(5.0, deadline - time.time()),
                )

                if response == "BARRIER_DONE":
                    return True
                elif response.startswith("BARRIER_WAIT"):
                    time.sleep(poll_interval)
                elif response.startswith("ERROR"):
                    raise RuntimeError(f"Barrier error: {response}")
                else:
                    raise RuntimeError(f"Unexpected barrier response: {response}")

            except socket.timeout:
                continue
            except ConnectionRefusedError:
                time.sleep(0.1)
                continue

        raise TimeoutError(f"Barrier {barrier_id} timeout after {timeout}s")

    def reset(self) -> bool:
        """Reset server state (for test isolation)."""
        response = self._send_command("RESET")
        return response == "OK"

    def clear_barriers(self) -> int:
        """Clear all pending barriers (for test cleanup between runs)."""
        response = self._send_command("CLEAR_BARRIERS")
        if response.startswith("OK"):
            parts = response.split()
            return int(parts[1]) if len(parts) > 1 else 0
        return 0


# ============================================================================
# Convenience Functions
# ============================================================================

_test_server_process = None


def start_test_server(port: int = 9998) -> mp.Process:
    """Start the test rank server in a background process."""
    global _test_server_process

    def run_server():
        server = RankServer(port=port)
        server.start(background=False)

    _test_server_process = mp.Process(target=run_server, daemon=True)
    _test_server_process.start()
    time.sleep(0.5)
    return _test_server_process


def stop_test_server():
    """Stop the test rank server."""
    global _test_server_process
    if _test_server_process and _test_server_process.is_alive():
        _test_server_process.terminate()
        _test_server_process.join(timeout=2)
    _test_server_process = None


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test Rank Server for multi-node tests"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9998, help="Port to listen on")
    args = parser.parse_args()

    sys.stderr.write(f"Starting Test Rank Server on {args.host}:{args.port}\n")
    sys.stderr.write("Commands: GET_RANK, RELEASE_RANK, BARRIER, RESET\n")
    sys.stderr.write("Press Ctrl+C to stop\n")

    server = RankServer(host=args.host, port=args.port)
    try:
        server.start(background=False)
    except KeyboardInterrupt:
        sys.stderr.write("\nShutting down...\n")
        server.stop()


if __name__ == "__main__":
    main()
