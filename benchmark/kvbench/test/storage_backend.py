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
- FilesystemBackend: Uses NIXL POSIX/GDS for file I/O (FILE-type memory)
- DocaMemosBackend: Uses NIXL DOCA_MEMOS for an object/key KV store
  (OBJ-type memory; WRITE=STORE, READ=RETRIEVE)
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
        backend_data: Backend-specific data. For a FILE backend this carries the
            fd / file_path (and shard info); for an OBJECT backend
            (e.g. DocaMemosBackend) it carries the registered OBJ reg-dlists,
            the resolved object key(s), and a synthetic file_path label used only
            for log strings. Backends may be FILE- or OBJECT-based.
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
        backend_params: Optional[Dict[str, str]] = None,
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
            base_path,
            nixl_backend,
            use_direct_io,
            block_size,
            backend_params,
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
        """Open a file and register it with NIXL. Returns fd.

        If register_memory() raises, closes the fd and drops the bookkeeping
        entry so the failure doesn't leak descriptors or leave stale state.
        """
        flags = os.O_RDWR
        if self._use_direct_io:
            flags |= os.O_DIRECT
        fd = os.open(str(file_path), flags)
        self._file_descriptors[str(file_path)] = fd

        try:
            reg_list = [(0, file_size, fd, str(file_path))]
            reg_descs = self._agent.register_memory(
                reg_list, "FILE", backends=[self._nixl_backend]
            )
        except Exception:
            self._file_descriptors.pop(str(file_path), None)
            try:
                os.close(fd)
            except OSError as exc:
                logger.debug(
                    "os.close(fd=%d) failed during register rollback: %s", fd, exc
                )
            raise
        self._file_reg_descs[str(file_path)] = reg_descs
        return fd

    def _release_file(self, file_path: str) -> None:
        """Deregister and close one file. Safe to call on partial state."""
        reg_descs = self._file_reg_descs.pop(file_path, None)
        if reg_descs is not None:
            try:
                self._agent.deregister_memory(reg_descs, backends=[self._nixl_backend])
            except Exception as exc:
                logger.debug("deregister_memory(%s) failed: %s", file_path, exc)
        fd = self._file_descriptors.pop(file_path, None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError as exc:
                logger.debug("os.close(fd=%d) failed: %s", fd, exc)

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
                tp_idx=tp_idx,
                rank=rank,
                read_size=read_size,
                write_size=write_size,
                backend_data={"file_path": str(file_path), "fd": fd},
            )
        else:
            # Multi-file: N shards, each with ceil(size/N) bytes (then
            # aligned up). Floor division here would yield 0 whenever
            # read_size < n_shards and silently produce no shards.
            align = max(self._block_size, 4096) if self._block_size > 0 else 4096

            def _ceil_align(total: int, n: int) -> int:
                if total <= 0:
                    return 0
                per_shard = (total + n - 1) // n
                return ((per_shard + align - 1) // align) * align

            shard_read = _ceil_align(read_size, n_shards)
            shard_write = _ceil_align(write_size, n_shards)
            shard_size = shard_read + shard_write

            shard_fds = []
            shard_paths = []
            shard_actual_reads = []
            try:
                for shard in range(n_shards):
                    fpath = self._get_file_path(tp_idx, rank, shard=shard)
                    actual_read = min(shard_read, read_size - shard * shard_read)
                    actual_read = max(actual_read, 0)
                    actual_size = actual_read + shard_write
                    if actual_size <= 0:
                        break
                    if not fpath.exists():
                        self._create_file(fpath, actual_size, actual_read, rank)
                    fd = self._open_and_register_file(fpath, actual_size)
                    shard_fds.append(fd)
                    shard_paths.append(str(fpath))
                    shard_actual_reads.append(actual_read)
            except Exception:
                # Roll back any shards opened so far; otherwise the caller's
                # except handler would inherit half-registered files.
                for path in shard_paths:
                    self._release_file(path)
                raise

            handle = StorageHandle(
                tp_idx=tp_idx,
                rank=rank,
                read_size=read_size,
                write_size=write_size,
                backend_data={
                    "file_path": shard_paths[0],
                    "fd": shard_fds[0],
                    "shard_fds": shard_fds,
                    "shard_paths": shard_paths,
                    "shard_read_size": shard_read,
                    "shard_write_size": shard_write,
                    # Per-shard actual read region size. The trailing shard's
                    # read region may be shorter than shard_read when read_size
                    # is not divisible by n_shards; writes for that shard must
                    # start at this actual offset, not at shard_read, or they
                    # would land past EOF of the (smaller) shard file.
                    "shard_actual_reads": shard_actual_reads,
                },
            )

            logger.debug(
                "Prepared sharded storage: tp=%d, rank=%d, %d shards x %d bytes",
                tp_idx,
                rank,
                len(shard_fds),
                shard_size,
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

    @staticmethod
    def _is_sharded(handle: StorageHandle) -> bool:
        shard_fds = handle.backend_data.get("shard_fds")
        return bool(shard_fds) and len(shard_fds) > 1

    def get_read_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get a single NIXL transfer handle that covers the whole read region.

        Not supported for sharded backends (multiple shard files per rank) —
        a single handle would only touch the first shard. Use
        get_read_handles() instead, which returns one handle per shard.
        """
        if handle.read_size == 0:
            return None
        if self._is_sharded(handle):
            raise ValueError(
                "get_read_handle() does not support sharded storage "
                "(rank has multiple shard files). Use get_read_handles() instead."
            )
        return self._create_handle(
            "READ", handle.backend_data["fd"], 0, handle.read_size, buffer
        )

    def get_write_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get a single NIXL transfer handle that covers the whole write region.

        Not supported for sharded backends — see get_read_handle().
        """
        if handle.write_size == 0:
            return None
        if self._is_sharded(handle):
            raise ValueError(
                "get_write_handle() does not support sharded storage "
                "(rank has multiple shard files). Use get_write_handles() instead."
            )
        fd = handle.backend_data["fd"]
        write_offset = handle.read_size
        return self._create_handle("WRITE", fd, write_offset, handle.write_size, buffer)

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

        total = handle.read_size

        # Sharded mode always uses one handle per shard; num_handles is
        # ignored because mixing fds within a shard buys nothing on NFS.
        # Check this BEFORE the num_handles<=1 singular fallback, since
        # get_read_handle() can't represent a sharded read.
        if self._is_sharded(handle):
            shard_read = handle.backend_data["shard_read_size"]
            handles = []
            for i, fd in enumerate(handle.backend_data["shard_fds"]):
                offset_in_buf = i * shard_read
                size = min(shard_read, total - offset_in_buf)
                if size <= 0:
                    break
                buf_slice = buffer[offset_in_buf : offset_in_buf + size]
                # Each shard file starts at offset 0
                xfer = self._create_handle("READ", fd, 0, size, buf_slice)
                handles.append(xfer)
            return handles

        if num_handles <= 1:
            h = self.get_read_handle(handle, buffer)
            return [h] if h else []

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
            buf_slice = buffer[offset : offset + size]
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

        total = handle.write_size

        # Sharded mode: one handle per shard regardless of num_handles.
        # See the symmetric note in get_read_handles().
        if self._is_sharded(handle):
            shard_write = handle.backend_data["shard_write_size"]
            shard_actual_reads = handle.backend_data["shard_actual_reads"]
            handles = []
            for i, fd in enumerate(handle.backend_data["shard_fds"]):
                offset_in_buf = i * shard_write
                size = min(shard_write, total - offset_in_buf)
                if size <= 0:
                    break
                buf_slice = buffer[offset_in_buf : offset_in_buf + size]
                # Write goes immediately after this shard's read region. For
                # the trailing shard that region may be shorter than
                # shard_read_size, so use the per-shard value.
                write_offset = shard_actual_reads[i]
                xfer = self._create_handle("WRITE", fd, write_offset, size, buf_slice)
                handles.append(xfer)
            return handles

        if num_handles <= 1:
            h = self.get_write_handle(handle, buffer)
            return [h] if h else []

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
            buf_slice = buffer[offset : offset + size]
            xfer = self._create_handle(
                "WRITE", fd, write_offset + offset, size, buf_slice
            )
            handles.append(xfer)
        return handles

    def close(self):
        """Close all files and deregister from NIXL."""
        # Deregister from NIXL. Failures here are not fatal — we still need
        # to release the file descriptors below — but we log them at debug
        # level so they're recoverable when investigating a cleanup issue.
        for reg_descs in self._file_reg_descs.values():
            try:
                self._agent.deregister_memory(reg_descs, backends=[self._nixl_backend])
            except Exception as exc:
                logger.debug("deregister_memory failed during close: %s", exc)

        # Close file descriptors
        for fd in self._file_descriptors.values():
            try:
                os.close(fd)
            except OSError as exc:
                logger.debug("os.close(fd=%d) failed during close: %s", fd, exc)

        self._handles.clear()
        self._file_descriptors.clear()
        self._file_reg_descs.clear()

        logger.debug("FilesystemBackend closed")


# Hard limit on the DOCA_MEMOS object key length, mirroring
# DOCA_MEMOS_MAX_OBJECT_KEY_LEN in the plugin header
# (src/plugins/doca_memos/doca_memos_backend.h). A non-hex metaInfo longer than
# this fails registration with NIXL_ERR_INVALID_PARAM, so we assert on it.
DOCA_MEMOS_MAX_OBJECT_KEY_LEN = 16


class DocaMemosBackend(StorageBackend):
    """Object/key storage backend using the NIXL DOCA_MEMOS plugin.

    DOCA_MEMOS is an OBJECT/KEY store (NVIDIA DOCA hardware KV on BlueField),
    not a file store: local DRAM/VRAM buffers are the local side and keyed
    objects (OBJ-type memory) are the remote side. WRITE=STORE, READ=RETRIEVE.

    Unlike FilesystemBackend there is no fd/offset and no file sharding. The
    object key is bound at register_memory(OBJ) time from the descriptor's
    metaInfo, so prepare() registers one OBJ descriptor per op up front and the
    read/write handle builders reuse a matching xfer descriptor (matched by
    devId). This mirrors the C++ nixlbench worker lifecycle.

    Object layout per (tp, rank):
        key = f"{key_prefix}{tp_idx}_{rank}"           (single shared key)
        key = f"{key_prefix}{tp_idx}_{rank}_r" / "_w"  (when both sizes != 0)

    Only the singular ABC methods (prepare / get_read_handle / get_write_handle
    / close) are implemented; the plural get_read_handles()/get_write_handles()
    file-sharding helpers are deliberately NOT provided, so the runner must
    drive this backend in single-handle mode.
    """

    NIXL_BACKEND = "DOCA_MEMOS"
    MEM_TYPE = "OBJ"

    def __init__(
        self,
        agent: nixl_agent,
        device_name: str,
        num_tasks: Optional[int] = None,
        nguid: Optional[str] = None,
        query_mem_mode: str = "assume_success",
        ignore_read_not_found: bool = False,
        key_prefix: str = "kvbench",
    ):
        """Initialize the DOCA_MEMOS object/key backend.

        Args:
            agent: NIXL agent for transfers.
            device_name: Path to the NVMe KV device (e.g. "/dev/nvme0n1").
                         Required by the plugin.
            num_tasks: Optional DOCA task-pool size (plugin default 8192).
            nguid: Optional 32-char hex NVMe namespace NGUID.
            query_mem_mode: queryMem() behavior, "assume_success" or "actual".
            ignore_read_not_found: When True, RETRIEVE of a missing key
                                   succeeds (undefined buffer) instead of
                                   erroring. Escape hatch for code/mock runs;
                                   prepare() pre-stores read objects regardless.
            key_prefix: Prefix used to build per-(tp,rank,op) object keys.
        """
        if not device_name:
            raise ValueError("DocaMemosBackend requires a non-empty device_name")

        self._agent = agent
        self._device_name = device_name
        self._key_prefix = key_prefix
        self._ignore_read_not_found = ignore_read_not_found

        # Bookkeeping: object key -> registered OBJ nixlRegDList, so close()
        # can deregister every OBJ descriptor (analogous to
        # FilesystemBackend._file_reg_descs).
        self._obj_reg_descs: Dict[str, Any] = {}
        # tp:rank -> StorageHandle, analogous to FilesystemBackend._handles.
        self._handles: Dict[str, StorageHandle] = {}
        # Monotonic devId allocator: each registered object gets a unique small
        # integer so the core can bind the xfer descriptor back to the
        # registered OBJ metadata by devId.
        self._next_dev_id = 0

        # Build the create_backend params (string->string map). Mirrors the C++
        # nixlbench worker (nixl_worker.cpp:315-331); optional values are only
        # emitted when provided so the plugin defaults apply otherwise.
        params: Dict[str, str] = {
            "device_name": device_name,
            "query_mem_mode": query_mem_mode,
        }
        if num_tasks is not None:
            params["num_tasks"] = str(num_tasks)
        if nguid:
            params["nguid"] = nguid
        if ignore_read_not_found:
            params["ignore_read_not_found"] = "true"
        self._backend_params = params

        try:
            self._agent.create_backend(self.NIXL_BACKEND, params)
        except Exception as e:
            logger.debug(
                "create_backend(%s) returned: %s (may already exist)",
                self.NIXL_BACKEND,
                e,
            )

        logger.info(
            "DocaMemosBackend: device=%s query_mem_mode=%s num_tasks=%s "
            "nguid=%s ignore_read_not_found=%s key_prefix=%s",
            device_name,
            query_mem_mode,
            num_tasks,
            nguid,
            ignore_read_not_found,
            key_prefix,
        )

    def _alloc_dev_id(self) -> int:
        """Allocate a unique devId for one registered object."""
        dev_id = self._next_dev_id
        self._next_dev_id += 1
        return dev_id

    def _obj_key(self, tp_idx: int, rank: int, suffix: str = "") -> str:
        """Build the deterministic <=16-byte ASCII object key for an op.

        Raises ValueError if the encoded key exceeds DOCA_MEMOS_MAX_OBJECT_KEY_LEN;
        a non-hex key longer than 16 bytes would fail registration in the plugin
        (resolveMemosKey -> NIXL_ERR_INVALID_PARAM).
        """
        key = f"{self._key_prefix}{tp_idx}_{rank}{suffix}"
        encoded_len = len(key.encode("utf-8"))
        if encoded_len > DOCA_MEMOS_MAX_OBJECT_KEY_LEN:
            raise ValueError(
                f"DOCA_MEMOS object key {key!r} encodes to {encoded_len} bytes, "
                f"exceeding the {DOCA_MEMOS_MAX_OBJECT_KEY_LEN}-byte limit. "
                "Use a shorter key_prefix or fewer ranks."
            )
        return key

    def _register_obj(self, key: str, size: int) -> tuple:
        """Register one OBJ_SEG descriptor for `key` with value length `size`.

        Returns (reg_descs, dev_id). The 4-tuple registration descriptor is
        (addr, len, devId, metaInfo); addr is irrelevant for OBJ (0), len is the
        object value size, devId is unique, metaInfo is the key string.
        """
        dev_id = self._alloc_dev_id()
        reg_list = [(0, size, dev_id, key)]
        reg_descs = self._agent.register_memory(
            reg_list, self.MEM_TYPE, backends=[self.NIXL_BACKEND]
        )
        self._obj_reg_descs[key] = reg_descs
        return reg_descs, dev_id

    def _prestore_read_object(self, key: str, dev_id: int, size: int) -> None:
        """STORE a `size`-byte value at `key` so a later RETRIEVE finds it.

        DOCA RETRIEVE fails unless the key was previously STORED with a value of
        matching length (mirrors the C++ pre-population at
        nixl_worker.cpp:1994-2001). Uses a transient scratch DRAM buffer and a
        synchronous WRITE (STORE) transfer.
        """
        import torch

        scratch = torch.zeros(size, dtype=torch.uint8)
        local_descs = self._agent.get_xfer_descs(scratch)
        obj_descs = self._agent.get_xfer_descs([(0, size, dev_id)], self.MEM_TYPE)
        xfer = self._agent.initialize_xfer(
            "WRITE",
            local_descs,
            obj_descs,
            self._agent.name,
            backends=[self.NIXL_BACKEND],
        )
        try:
            state = self._agent.transfer(xfer)
            while state == "PROC":
                state = self._agent.check_xfer_state(xfer)
            if state == "ERR":
                raise RuntimeError(f"DOCA_MEMOS pre-store STORE failed for key {key!r}")
        finally:
            try:
                self._agent.release_xfer_handle(xfer)
            except Exception as exc:
                logger.debug("release_xfer_handle (pre-store) failed: %s", exc)
        # Keep `scratch` referenced until the transfer is released so its memory
        # stays valid for the duration of the STORE.
        del scratch

    def prepare(
        self,
        tp_idx: int,
        rank: int,
        read_size: int,
        write_size: int,
    ) -> StorageHandle:
        """Register OBJ descriptor(s) for a rank and pre-store read objects.

        Registers one OBJ_SEG descriptor per op. When both read_size and
        write_size are non-zero, distinct `_r`/`_w` keys are used so a write of
        write_size bytes does not change the length of the read object
        (avoiding a RETRIEVE length mismatch). When only one of them is
        non-zero, a single shared key is used.

        For a read op (read_size > 0) the object is pre-stored with a read_size
        value so the first RETRIEVE finds it.
        """
        backend_data: Dict[str, Any] = {}
        both = read_size > 0 and write_size > 0
        read_suffix = "_r" if both else ""
        write_suffix = "_w" if both else ""

        registered_keys = []
        try:
            if read_size > 0:
                read_key = self._obj_key(tp_idx, rank, read_suffix)
                _, read_dev_id = self._register_obj(read_key, read_size)
                registered_keys.append(read_key)
                backend_data["read_key"] = read_key
                backend_data["read_dev_id"] = read_dev_id
                # Pre-store so RETRIEVE won't miss (Sec. 4d).
                self._prestore_read_object(read_key, read_dev_id, read_size)

            if write_size > 0:
                write_key = self._obj_key(tp_idx, rank, write_suffix)
                _, write_dev_id = self._register_obj(write_key, write_size)
                registered_keys.append(write_key)
                backend_data["write_key"] = write_key
                backend_data["write_dev_id"] = write_dev_id
        except Exception:
            # Roll back any objects registered so far so a failure does not
            # leak registrations or leave stale bookkeeping.
            for k in registered_keys:
                reg_descs = self._obj_reg_descs.pop(k, None)
                if reg_descs is not None:
                    try:
                        self._agent.deregister_memory(
                            reg_descs, backends=[self.NIXL_BACKEND]
                        )
                    except Exception as exc:
                        logger.debug(
                            "deregister_memory(%s) failed during rollback: %s",
                            k,
                            exc,
                        )
            raise

        # Synthetic file_path label: used only for log strings by the runner's
        # StorageXferHandle (the runner reads backend_data["file_path"]).
        backend_data["file_path"] = backend_data.get(
            "read_key", backend_data.get("write_key", f"tp_{tp_idx}_rank_{rank}")
        )

        handle = StorageHandle(
            tp_idx=tp_idx,
            rank=rank,
            read_size=read_size,
            write_size=write_size,
            backend_data=backend_data,
        )
        self._handles[f"{tp_idx}:{rank}"] = handle

        logger.debug(
            "Prepared DOCA_MEMOS objects: tp=%d rank=%d read_key=%s write_key=%s",
            tp_idx,
            rank,
            backend_data.get("read_key"),
            backend_data.get("write_key"),
        )
        return handle

    def _build_obj_xfer(self, dev_id: int, size: int):
        """Build the OBJ xfer dlist for a registered object (3-tuple)."""
        return self._agent.get_xfer_descs([(0, size, dev_id)], self.MEM_TYPE)

    def get_read_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get a NIXL RETRIEVE (READ) handle for the rank's read object."""
        if handle.read_size == 0:
            return None
        dev_id = handle.backend_data["read_dev_id"]
        local_descs = self._agent.get_xfer_descs(buffer)
        obj_descs = self._build_obj_xfer(dev_id, handle.read_size)
        return self._agent.initialize_xfer(
            "READ",
            local_descs,
            obj_descs,
            self._agent.name,
            backends=[self.NIXL_BACKEND],
        )

    def get_write_handle(
        self,
        handle: StorageHandle,
        buffer: Any,
    ) -> Any:
        """Get a NIXL STORE (WRITE) handle for the rank's write object."""
        if handle.write_size == 0:
            return None
        dev_id = handle.backend_data["write_dev_id"]
        local_descs = self._agent.get_xfer_descs(buffer)
        obj_descs = self._build_obj_xfer(dev_id, handle.write_size)
        return self._agent.initialize_xfer(
            "WRITE",
            local_descs,
            obj_descs,
            self._agent.name,
            backends=[self.NIXL_BACKEND],
        )

    def exists(self, handle: StorageHandle) -> bool:
        """Optional EXIST helper: query whether the rank's object(s) exist.

        Uses agent.query_memory (the plugin's queryMem=EXIST). Not used by the
        runner; provided for mock/unit verification. In `assume_success` mode the
        plugin always reports success.
        """
        keys = []
        if handle.read_size > 0:
            keys.append(("read_dev_id", handle.read_size))
        if handle.write_size > 0:
            keys.append(("write_dev_id", handle.write_size))
        if not keys:
            return False
        for dev_id_field, size in keys:
            dev_id = handle.backend_data[dev_id_field]
            reg_list = [(0, size, dev_id, "")]
            results = self._agent.query_memory(
                reg_list, self.NIXL_BACKEND, self.MEM_TYPE
            )
            if not results or results[0] is None:
                return False
        return True

    def close(self):
        """Deregister every OBJ descriptor and clear bookkeeping.

        There is no object-delete API in the plugin (see C++ note at
        nixl_worker.cpp), so close() only deregisters; the stored objects
        persist on the device.
        """
        for key, reg_descs in self._obj_reg_descs.items():
            try:
                self._agent.deregister_memory(reg_descs, backends=[self.NIXL_BACKEND])
            except Exception as exc:
                logger.debug("deregister_memory(%s) failed during close: %s", key, exc)

        self._handles.clear()
        self._obj_reg_descs.clear()

        logger.debug("DocaMemosBackend closed")
