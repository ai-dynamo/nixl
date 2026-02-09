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
    ):
        """Initialize filesystem backend.

        Args:
            agent: NIXL agent for transfers
            base_path: Base directory for storage files
            nixl_backend: NIXL backend to use ("POSIX", "GDS", or "GDS_MT")
            use_direct_io: Use O_DIRECT for file I/O (recommended for GDS)
        """
        self._agent = agent
        self._base_path = Path(base_path)
        self._nixl_backend = nixl_backend
        self._use_direct_io = use_direct_io
        self._handles: Dict[str, StorageHandle] = {}  # key -> handle
        self._file_descriptors: Dict[str, int] = {}  # file_path -> fd
        self._file_reg_descs: Dict[str, Any] = {}  # file_path -> nixl reg_descs

        # Ensure backend is created
        try:
            self._agent.create_backend(nixl_backend)
        except Exception as e:
            logger.debug(
                "create_backend(%s) returned: %s (may already exist)", nixl_backend, e
            )

        logger.debug(
            "FilesystemBackend initialized: base_path=%s, backend=%s, direct_io=%s",
            base_path,
            nixl_backend,
            use_direct_io,
        )

    def _get_file_path(self, tp_idx: int, rank: int) -> Path:
        """Get file path for a rank."""
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
            # Prefill READ region (simulates cached data)
            if read_size > 0:
                chunk_size = min(8 * 1024 * 1024, read_size)
                chunk = bytes([rank % 256]) * chunk_size
                written = 0
                while written < read_size:
                    to_write = min(chunk_size, read_size - written)
                    f.write(chunk[:to_write])
                    written += to_write

            # Preallocate WRITE region blocks (avoids first-write block allocation
            # latency, which is especially important with O_DIRECT)
            if file_size > read_size:
                try:
                    os.posix_fallocate(f.fileno(), read_size, file_size - read_size)
                except OSError:
                    # Fallback for filesystems that don't support fallocate
                    f.seek(file_size - 1)
                    f.write(b"\0")
            f.flush()
            os.fsync(f.fileno())

    def prepare(
        self,
        tp_idx: int,
        rank: int,
        read_size: int,
        write_size: int,
    ) -> StorageHandle:
        """Prepare storage file for a rank."""
        file_path = self._get_file_path(tp_idx, rank)
        file_size = read_size + write_size
        key = f"{tp_idx}:{rank}"

        # Create file if doesn't exist
        if not file_path.exists():
            self._create_file(file_path, file_size, read_size, rank)

        # Open file with optional O_DIRECT for GDS
        flags = os.O_RDWR
        if self._use_direct_io:
            flags |= os.O_DIRECT
        fd = os.open(str(file_path), flags)
        self._file_descriptors[str(file_path)] = fd

        # Register with NIXL
        reg_list = [(0, file_size, fd, str(file_path))]
        reg_descs = self._agent.register_memory(
            reg_list, "FILE", backends=[self._nixl_backend]
        )
        self._file_reg_descs[str(file_path)] = reg_descs

        handle = StorageHandle(
            tp_idx=tp_idx,
            rank=rank,
            read_size=read_size,
            write_size=write_size,
            backend_data={
                "file_path": str(file_path),
                "fd": fd,
            },
        )
        self._handles[key] = handle

        logger.debug(
            "Prepared storage: tp=%d, rank=%d, file=%s, read=%d, write=%d",
            tp_idx,
            rank,
            file_path,
            read_size,
            write_size,
        )

        return handle

    def get_read_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get NIXL transfer handle for reading from file."""
        if handle.read_size == 0:
            return None

        fd = handle.backend_data["fd"]

        # File descriptors: (offset, size, fd)
        file_descs = self._agent.get_xfer_descs([(0, handle.read_size, fd)], "FILE")
        local_descs = self._agent.get_xfer_descs(buffer)

        xfer_handle = self._agent.initialize_xfer(
            "READ",
            local_descs,
            file_descs,
            self._agent.name,
            backends=[self._nixl_backend],
        )

        return xfer_handle

    def get_write_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get NIXL transfer handle for writing to file."""
        if handle.write_size == 0:
            return None

        fd = handle.backend_data["fd"]
        write_offset = handle.read_size  # Write starts after read region

        # File descriptors: (offset, size, fd)
        file_descs = self._agent.get_xfer_descs(
            [(write_offset, handle.write_size, fd)], "FILE"
        )
        local_descs = self._agent.get_xfer_descs(buffer)

        xfer_handle = self._agent.initialize_xfer(
            "WRITE",
            local_descs,
            file_descs,
            self._agent.name,
            backends=[self._nixl_backend],
        )

        return xfer_handle

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
