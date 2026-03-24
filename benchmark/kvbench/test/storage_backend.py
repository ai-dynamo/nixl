# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Storage backend abstraction for KV cache I/O benchmarking.

Provides an abstract interface for storage operations, allowing different
backend implementations (filesystem, Redis, block devices, etc.).

Current implementations:
- FilesystemBackend: Uses NIXL POSIX/GDS for file I/O
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from nixl._api import nixl_agent
from nixl.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StorageHandle:
    """Handle returned by storage backend for a prepared region.

    Attributes:
        tp_idx: Traffic pattern index
        rank: Rank this handle belongs to
        read_size: Size of read region in bytes
        write_size: Size of write region in bytes
        backend_data: Backend-specific data (fd, connection, etc.)
    """

    tp_idx: int
    rank: int
    read_size: int
    write_size: int
    backend_data: (
        Any  # Backend-specific (fd for filesystem, connection for redis, etc.)
    )


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    Implementations must provide methods to:
    - Prepare storage regions (create files, connect to Redis, etc.)
    - Get NIXL-compatible transfer handles for read/write operations
    - Clean up resources

    Example usage:
        backend = FilesystemBackend(nixl_agent, base_path="/mnt/storage")

        # Prepare storage for a rank
        handle = backend.prepare(tp_idx=0, rank=0, read_size=1000, write_size=500)

        # Get NIXL transfer handles
        read_handle = backend.get_read_handle(handle, gpu_buffer)
        write_handle = backend.get_write_handle(handle, gpu_buffer)

        # Execute transfers via NIXL
        nixl_agent.transfer(read_handle)
        nixl_agent.transfer(write_handle)

        # Cleanup
        backend.close()
    """

    @abstractmethod
    def prepare(
        self,
        tp_idx: int,
        rank: int,
        read_size: int,
        write_size: int,
    ) -> StorageHandle:
        """Prepare storage region for a rank.

        Creates/opens the storage region and returns a handle for later use.
        For filesystem: creates file, prefills read region
        For Redis: creates key, populates with initial data

        Args:
            tp_idx: Traffic pattern index
            rank: Rank number
            read_size: Size of read region in bytes
            write_size: Size of write region in bytes

        Returns:
            StorageHandle for use with get_read_handle/get_write_handle
        """
        pass

    @abstractmethod
    def get_read_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get NIXL transfer handle for reading from storage.

        Args:
            handle: StorageHandle from prepare()
            buffer: GPU/CPU buffer to read into

        Returns:
            NIXL transfer handle (ready for nixl_agent.transfer())
        """
        pass

    @abstractmethod
    def get_write_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get NIXL transfer handle for writing to storage.

        Args:
            handle: StorageHandle from prepare()
            buffer: GPU/CPU buffer to write from

        Returns:
            NIXL transfer handle (ready for nixl_agent.transfer())
        """
        pass

    @abstractmethod
    def close(self):
        """Close all storage handles and release resources."""
        pass


class FilesystemBackend(StorageBackend):
    """Filesystem-based storage backend using NIXL POSIX/GDS.

    File layout per rank:
        <base_path>/tp_<idx>/rank_<rank>.bin

    File structure:
        [read_region (prefilled)][write_region (empty)]
        offset=0                  offset=read_size
    """

    def __init__(
        self,
        agent: nixl_agent,
        base_path: Path,
        nixl_backend: str = "POSIX",
        use_direct_io: bool = False,
        block_size: int = 0,
        backend_params: dict = None,
    ):
        """Initialize filesystem backend.

        Args:
            agent: NIXL agent for transfers
            base_path: Base directory for storage files
            nixl_backend: NIXL backend to use ("POSIX", "GDS", or "GDS_MT")
            use_direct_io: Use O_DIRECT for file I/O (recommended for GDS)
            block_size: Split I/O into blocks of this size for higher queue depth.
                        0 = no splitting (legacy single-descriptor mode).
                        Recommended: 1048576 (1MB) to match NFS rsize and maximize
                        I/O concurrency across nconnect sessions.
            backend_params: Optional dict of backend-specific params (e.g.,
                           {"use_uring": "true"} for io_uring).
        """
        self._agent = agent
        self._base_path = Path(base_path)
        self._nixl_backend = nixl_backend
        self._use_direct_io = use_direct_io
        self._block_size = block_size
        self._num_handles = 1  # Set by caller; controls file sharding
        self._handles: Dict[str, StorageHandle] = {}  # key -> handle
        self._file_descriptors: Dict[str, int] = {}  # file_path -> fd
        self._file_reg_descs: Dict[str, Any] = {}  # file_path -> nixl reg_descs

        # Ensure backend is created
        try:
            self._agent.create_backend(nixl_backend, backend_params or {})
        except Exception as e:
            logger.debug(
                "create_backend(%s) returned: %s (may already exist)", nixl_backend, e
            )

        logger.info(
            "FilesystemBackend: path=%s backend=%s direct_io=%s block_size=%d params=%s",
            base_path, nixl_backend, use_direct_io, block_size, backend_params,
        )

    def _get_file_path(self, tp_idx: int, rank: int, shard: int = -1) -> Path:
        """Get file path for a rank, optionally sharded."""
        if shard >= 0:
            return self._base_path / f"tp_{tp_idx}" / f"rank_{rank}_shard_{shard}.bin"
        return self._base_path / f"tp_{tp_idx}" / f"rank_{rank}.bin"

    def _create_file(self, file_path: Path, file_size: int, read_size: int, rank: int):
        """Create and prefill storage file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(
            "Creating storage file: %s (size=%d, read_region=%d)",
            file_path,
            file_size,
            read_size,
        )

        with open(file_path, "wb") as f:
            if read_size > 0:
                chunk_size = min(8 * 1024 * 1024, read_size)
                chunk = bytes([rank % 256]) * chunk_size
                written = 0
                while written < read_size:
                    to_write = min(chunk_size, read_size - written)
                    f.write(chunk[:to_write])
                    written += to_write

            if file_size > read_size:
                try:
                    os.posix_fallocate(f.fileno(), read_size, file_size - read_size)
                except OSError:
                    f.seek(file_size - 1)
                    f.write(b"\0")
            f.flush()
            os.fsync(f.fileno())

    def _open_and_register_file(self, file_path: Path, file_size: int) -> int:
        """Open a file and register it with NIXL. Returns fd."""
        flags = os.O_RDWR
        if self._use_direct_io:
            flags |= os.O_DIRECT
        fd = os.open(str(file_path), flags)
        self._file_descriptors[str(file_path)] = fd

        reg_list = [(0, file_size, fd, str(file_path))]
        reg_descs = self._agent.register_memory(
            reg_list, "FILE", backends=[self._nixl_backend]
        )
        self._file_reg_descs[str(file_path)] = reg_descs
        return fd

    @property
    def _num_shards(self):
        """Number of file shards per rank. Matches num_handles for parallel I/O."""
        return self._num_handles if self._num_handles > 1 else 1

    def prepare(
        self,
        tp_idx: int,
        rank: int,
        read_size: int,
        write_size: int,
    ) -> StorageHandle:
        """Prepare storage file(s) for a rank.

        When num_handles > 1, creates N shard files (one per handle) so each
        gets its own fd. The NFS client can then parallelize I/O across files.
        This is the key to matching nixlbench's 45 GB/s per node.
        """
        file_size = read_size + write_size
        key = f"{tp_idx}:{rank}"
        n_shards = self._num_shards

        if n_shards <= 1:
            # Legacy: single file
            file_path = self._get_file_path(tp_idx, rank)
            if not file_path.exists():
                self._create_file(file_path, file_size, read_size, rank)
            fd = self._open_and_register_file(file_path, file_size)

            handle = StorageHandle(
                tp_idx=tp_idx, rank=rank,
                read_size=read_size, write_size=write_size,
                backend_data={"file_path": str(file_path), "fd": fd},
            )
        else:
            # Multi-file: N shards, each with size/N
            align = max(self._block_size, 4096) if self._block_size > 0 else 4096
            shard_read = ((read_size // n_shards + align - 1) // align) * align
            shard_write = ((write_size // n_shards + align - 1) // align) * align if write_size > 0 else 0
            shard_size = shard_read + shard_write

            shard_fds = []
            shard_paths = []
            for s in range(n_shards):
                fpath = self._get_file_path(tp_idx, rank, shard=s)
                actual_read = min(shard_read, read_size - s * shard_read)
                actual_read = max(actual_read, 0)
                actual_size = actual_read + shard_write
                if actual_size <= 0:
                    break
                if not fpath.exists():
                    self._create_file(fpath, actual_size, actual_read, rank)
                fd = self._open_and_register_file(fpath, actual_size)
                shard_fds.append(fd)
                shard_paths.append(str(fpath))

            handle = StorageHandle(
                tp_idx=tp_idx, rank=rank,
                read_size=read_size, write_size=write_size,
                backend_data={
                    "file_path": shard_paths[0],
                    "fd": shard_fds[0],
                    "shard_fds": shard_fds,
                    "shard_paths": shard_paths,
                    "shard_read_size": shard_read,
                    "shard_write_size": shard_write,
                },
            )

            logger.debug(
                "Prepared sharded storage: tp=%d, rank=%d, %d shards x %d bytes",
                tp_idx, rank, len(shard_fds), shard_size,
            )

        self._handles[key] = handle
        return handle

    def _create_chunked_descs(self, fd, file_offset, total_size, buffer):
        """Create file and local memory descriptors, optionally chunked for high queue depth.

        When block_size > 0, splits the transfer into multiple small descriptors.
        This increases the async I/O queue depth in the POSIX/GDS backend, allowing
        the NFS client to pipeline requests across nconnect sessions.

        Both local and file descriptor lists must have matching entry counts
        (required by the POSIX backend: posix_backend.cpp line 57).

        Args:
            fd: File descriptor
            file_offset: Starting offset in file
            total_size: Total bytes to transfer
            buffer: Memory buffer (torch tensor)

        Returns:
            (local_descs, file_descs) tuple of NIXL descriptor lists
        """
        bs = self._block_size
        if bs <= 0 or total_size <= bs:
            # Legacy: single descriptor (queue_depth = 1)
            file_descs = self._agent.get_xfer_descs(
                [(file_offset, total_size, fd)], "FILE"
            )
            local_descs = self._agent.get_xfer_descs(buffer)
            return local_descs, file_descs

        # Chunked: N descriptors of block_size each (queue_depth = N)
        # This mimics nixlbench's approach: 64 x 1MB = 64 outstanding I/Os
        buf_addr = buffer.data_ptr()
        dev_id = buffer.device.index if buffer.is_cuda else 0

        file_tuples = []
        local_tuples = []
        for off in range(0, total_size, bs):
            chunk = min(bs, total_size - off)
            file_tuples.append((file_offset + off, chunk, fd))
            local_tuples.append((buf_addr + off, chunk, dev_id or 0))

        file_descs = self._agent.get_xfer_descs(file_tuples, "FILE")
        local_mem_type = "VRAM" if buffer.is_cuda else "DRAM"
        local_descs = self._agent.get_xfer_descs(local_tuples, local_mem_type)

        logger.debug(
            "Chunked I/O: %d descs x %d bytes (total %d, queue_depth=%d)",
            len(file_tuples),
            bs,
            total_size,
            len(file_tuples),
        )

        return local_descs, file_descs

    def _create_handle(self, op, fd, file_offset, size, buffer):
        """Create a single NIXL transfer handle for a file region."""
        local_descs, file_descs = self._create_chunked_descs(
            fd, file_offset, size, buffer
        )
        return self._agent.initialize_xfer(
            op,
            local_descs,
            file_descs,
            self._agent.name,
            backends=[self._nixl_backend],
        )

    def _create_chunked_descs(self, fd, file_offset, total_size, buffer):
        """Create file and local memory descriptors, optionally chunked for high queue depth.

        When block_size > 0, splits the transfer into multiple small descriptors.
        This increases the async I/O queue depth in the POSIX/GDS backend, allowing
        the NFS client to pipeline requests across nconnect sessions.

        Both local and file descriptor lists must have matching entry counts
        (required by the POSIX backend: posix_backend.cpp line 57).

        Args:
            fd: File descriptor
            file_offset: Starting offset in file
            total_size: Total bytes to transfer
            buffer: Memory buffer (torch tensor)

        Returns:
            (local_descs, file_descs) tuple of NIXL descriptor lists
        """
        bs = self._block_size
        if bs <= 0 or total_size <= bs:
            # Legacy: single descriptor (queue_depth = 1)
            file_descs = self._agent.get_xfer_descs(
                [(file_offset, total_size, fd)], "FILE"
            )
            local_descs = self._agent.get_xfer_descs(buffer)
            return local_descs, file_descs

        # Chunked: N descriptors of block_size each (queue_depth = N)
        # This mimics nixlbench's approach: 64 x 1MB = 64 outstanding I/Os
        buf_addr = buffer.data_ptr()
        dev_id = buffer.device.index if buffer.is_cuda else 0

        file_tuples = []
        local_tuples = []
        for off in range(0, total_size, bs):
            chunk = min(bs, total_size - off)
            file_tuples.append((file_offset + off, chunk, fd))
            local_tuples.append((buf_addr + off, chunk, dev_id or 0))

        file_descs = self._agent.get_xfer_descs(file_tuples, "FILE")
        local_mem_type = "VRAM" if buffer.is_cuda else "DRAM"
        local_descs = self._agent.get_xfer_descs(local_tuples, local_mem_type)

        logger.debug(
            "Chunked I/O: %d descs x %d bytes (total %d, queue_depth=%d)",
            len(file_tuples),
            bs,
            total_size,
            len(file_tuples),
        )

        return local_descs, file_descs

    def get_read_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get NIXL transfer handle for reading from file."""
        if handle.read_size == 0:
            return None
        return self._create_handle(
            "READ", handle.backend_data["fd"], 0, handle.read_size, buffer
        )

    def get_write_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get NIXL transfer handle for writing to file."""
        if handle.write_size == 0:
            return None
        fd = handle.backend_data["fd"]
        write_offset = handle.read_size
        return self._create_handle(
            "WRITE", fd, write_offset, handle.write_size, buffer
        )

    def get_read_handles(
        self,
        handle: StorageHandle,
        buffer: Any,
        num_handles: int = 8,
    ) -> list:
        """Create multiple concurrent transfer handles for high-throughput reads.

        When sharded files exist (prepare() created N files), each handle uses
        a separate fd. The NFS client parallelizes I/O across different fds,
        enabling ~10 GB/s per handle = N*10 GB/s total.

        Args:
            handle: StorageHandle from prepare()
            buffer: Memory buffer (full size)
            num_handles: Number of concurrent handles (default: 8, matching nixlbench)

        Returns:
            List of NIXL transfer handles
        """
        if handle.read_size == 0:
            return []

        if num_handles <= 1:
            h = self.get_read_handle(handle, buffer)
            return [h] if h else []

        total = handle.read_size
        shard_fds = handle.backend_data.get("shard_fds")

        if shard_fds and len(shard_fds) > 1:
            # Sharded: each handle reads from its own file/fd
            shard_read = handle.backend_data["shard_read_size"]
            handles = []
            for i, fd in enumerate(shard_fds):
                offset_in_buf = i * shard_read
                size = min(shard_read, total - offset_in_buf)
                if size <= 0:
                    break
                buf_slice = buffer[offset_in_buf:offset_in_buf + size]
                # Each shard file starts at offset 0
                xfer = self._create_handle("READ", fd, 0, size, buf_slice)
                handles.append(xfer)
            return handles
        else:
            # Single file: split into regions (same fd, limited by NFS)
            fd = handle.backend_data["fd"]
            align = max(self._block_size, 4096) if self._block_size > 0 else 4096
            chunk_per_handle = ((total // num_handles + align - 1) // align) * align

            handles = []
            for i in range(num_handles):
                offset = i * chunk_per_handle
                size = min(chunk_per_handle, total - offset)
                if size <= 0:
                    break
                buf_slice = buffer[offset:offset + size]
                xfer = self._create_handle("READ", fd, offset, size, buf_slice)
                handles.append(xfer)
            return handles

    def get_write_handles(
        self,
        handle: StorageHandle,
        buffer: Any,
        num_handles: int = 8,
    ) -> list:
        """Create multiple concurrent transfer handles for high-throughput writes."""
        if handle.write_size == 0:
            return []

        if num_handles <= 1:
            h = self.get_write_handle(handle, buffer)
            return [h] if h else []

        total = handle.write_size
        shard_fds = handle.backend_data.get("shard_fds")

        if shard_fds and len(shard_fds) > 1:
            shard_read = handle.backend_data["shard_read_size"]
            shard_write = handle.backend_data["shard_write_size"]
            handles = []
            for i, fd in enumerate(shard_fds):
                offset_in_buf = i * shard_write
                size = min(shard_write, total - offset_in_buf)
                if size <= 0:
                    break
                buf_slice = buffer[offset_in_buf:offset_in_buf + size]
                xfer = self._create_handle("WRITE", fd, shard_read, size, buf_slice)
                handles.append(xfer)
            return handles
        else:
            fd = handle.backend_data["fd"]
            write_offset = handle.read_size
            align = max(self._block_size, 4096) if self._block_size > 0 else 4096
            chunk_per_handle = ((total // num_handles + align - 1) // align) * align

            handles = []
            for i in range(num_handles):
                offset = i * chunk_per_handle
                size = min(chunk_per_handle, total - offset)
                if size <= 0:
                    break
                buf_slice = buffer[offset:offset + size]
                xfer = self._create_handle("WRITE", fd, write_offset + offset, size, buf_slice)
                handles.append(xfer)
            return handles

    def close(self):
        """Close all files and deregister from NIXL."""
        # Deregister from NIXL
        for file_path, reg_descs in self._file_reg_descs.items():
            try:
                self._agent.deregister_memory(reg_descs, backends=[self._nixl_backend])
            except Exception:
                pass

        # Close file descriptors
        for file_path, fd in self._file_descriptors.items():
            try:
                os.close(fd)
            except Exception:
                pass

        self._handles.clear()
        self._file_descriptors.clear()
        self._file_reg_descs.clear()

        logger.debug("FilesystemBackend closed")
