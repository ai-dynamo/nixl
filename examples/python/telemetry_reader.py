#!/usr/bin/env python3

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

import argparse
import ctypes
import logging
import mmap
import os
import signal
import sys
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants from telemetry_event.h
TELEMETRY_VERSION = 1
MAX_EVENT_NAME_LEN = 32

# NIXL telemetry categories
NIXL_TELEMETRY_MEMORY = 0
NIXL_TELEMETRY_TRANSFER = 1
NIXL_TELEMETRY_CONNECTION = 2
NIXL_TELEMETRY_BACKEND = 3
NIXL_TELEMETRY_ERROR = 4
NIXL_TELEMETRY_PERFORMANCE = 5
NIXL_TELEMETRY_SYSTEM = 6
NIXL_TELEMETRY_CUSTOM = 7
NIXL_TELEMETRY_MAX = 8

# Global flag for graceful shutdown
running = True


def signal_handler(signum, _):
    """Signal handler for Ctrl+C"""
    global running
    if signum == signal.SIGINT:
        logger.info("\nReceived Ctrl+C, shutting down...")
        running = False


class NixlTelemetryEvent(ctypes.Structure):
    """Python equivalent of nixlTelemetryEvent struct"""

    _pack_ = 1
    _fields_ = [
        ("timestamp_us", ctypes.c_uint64),
        ("category", ctypes.c_int),
        ("event_name", ctypes.c_char * MAX_EVENT_NAME_LEN),
        ("_padding", ctypes.c_uint32),
        ("value", ctypes.c_uint64),
    ]


class BufferHeader(ctypes.Structure):
    """Python equivalent of BufferHeader struct from cyclic_buffer.h"""

    _pack_ = 1
    _fields_ = [
        ("write_pos", ctypes.c_size_t),
        ("read_pos", ctypes.c_size_t),
        ("version", ctypes.c_uint32),
        ("expected_version", ctypes.c_uint32),
        ("capacity", ctypes.c_size_t),
        ("mask", ctypes.c_size_t),
    ]


class SharedRingBuffer:
    """Python wrapper for the C++ SharedRingBuffer class"""

    def __init__(self, file_path, version=1):
        self.file_path = file_path
        self.version = version
        self.file_fd = -1
        self.mmap_obj = None
        self.header = None
        self.data = None
        self.buffer_size = None

        self._open_file()
        self._map_memory()

    def _open_file(self):
        """Open existing file"""
        self.file_fd = os.open(self.file_path, os.O_RDWR)

    def _map_memory(self):
        """Map the file into memory"""
        self._map_header_only()

    def _map_header_only(self):
        """Map only the header to read buffer size"""
        # Map just the header first
        header_mmap = mmap.mmap(
            self.file_fd,
            ctypes.sizeof(BufferHeader),
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )

        temp_header = BufferHeader.from_buffer(header_mmap)

        if temp_header.version != self.version:
            del temp_header
            header_mmap.close()
            raise RuntimeError(
                f"Version mismatch: expected {self.version}, got {temp_header.version}"
            )

        self.buffer_size = temp_header.capacity
        logger.info("Auto-detected buffer size: %d", self.buffer_size)

        del temp_header
        header_mmap.close()

        # Now map the entire buffer
        total_size = (
            ctypes.sizeof(BufferHeader)
            + ctypes.sizeof(NixlTelemetryEvent) * self.buffer_size
        )
        self.mmap_obj = mmap.mmap(
            self.file_fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
        )

        # Create ctypes pointers to the mapped memory
        self.header = BufferHeader.from_buffer(self.mmap_obj)
        data_offset = ctypes.sizeof(BufferHeader)
        self.data = (NixlTelemetryEvent * self.buffer_size).from_buffer(
            self.mmap_obj, data_offset
        )

    def get_version(self):
        """Get the buffer version"""
        return self.header.version

    def size(self):
        """Get the number of events in the buffer"""
        write_pos = self.header.write_pos
        read_pos = self.header.read_pos
        return (write_pos - read_pos) & self.header.mask

    def get_capacity(self):
        """Get the buffer capacity"""
        return self.buffer_size

    def empty(self):
        """Check if buffer is empty"""
        return self.header.read_pos == self.header.write_pos

    def full(self):
        """Check if buffer is full"""
        write_pos = self.header.write_pos
        next_write = (write_pos + 1) & self.header.mask
        return next_write == self.header.read_pos

    def pop(self):
        """Pop an event from the buffer"""
        read_pos = self.header.read_pos

        if read_pos == self.header.write_pos:
            return None

        event = self.data[read_pos]

        next_read = (read_pos + 1) & self.header.mask
        self.header.read_pos = next_read

        return event

    def __del__(self):
        """Cleanup resources"""
        # if self.mmap_obj:
        #     self.mmap_obj.close()
        if self.file_fd != -1:
            os.close(self.file_fd)


def format_timestamp(timestamp_us):
    """Format timestamp in microseconds to readable format"""
    dt = datetime.fromtimestamp(timestamp_us / 1_000_000)
    microseconds = timestamp_us % 1_000_000
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{microseconds:06d}"


def format_bytes(bytes_val):
    """Format bytes to human readable format"""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    value = float(bytes_val)

    while value >= 1024.0 and unit_index < 4:
        value /= 1024.0
        unit_index += 1

    return f"{value:.2f} {units[unit_index]}"


def get_telemetry_category_string(category):
    """Get string representation of telemetry category"""
    category_strings = {
        NIXL_TELEMETRY_MEMORY: "MEMORY",
        NIXL_TELEMETRY_TRANSFER: "TRANSFER",
        NIXL_TELEMETRY_CONNECTION: "CONNECTION",
        NIXL_TELEMETRY_BACKEND: "BACKEND",
        NIXL_TELEMETRY_ERROR: "ERROR",
        NIXL_TELEMETRY_PERFORMANCE: "PERFORMANCE",
        NIXL_TELEMETRY_SYSTEM: "SYSTEM",
        NIXL_TELEMETRY_CUSTOM: "CUSTOM",
    }
    return category_strings.get(category, f"UNKNOWN_CATEGORY_{category}")


def print_telemetry_event(event):
    """Print telemetry event in a formatted way"""
    logger.info("\n=== NIXL Telemetry Event ===")
    logger.info("Timestamp: %s", format_timestamp(event.timestamp_us))

    # Decode event name
    event_name = event.event_name.decode("utf-8").rstrip("\x00")
    category_str = get_telemetry_category_string(event.category)

    logger.info("Category: %s", category_str)
    logger.info("Event: %s", event_name)
    logger.info("Value: %s", event.value)
    logger.info("===========================")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="NIXL Telemetry Reader")
    parser.add_argument(
        "--telemetry_path", help="Path to the telemetry file", required=True
    )

    args = parser.parse_args()

    logger.info("Telemetry path: %s", args.telemetry_path)
    telemetry_file_name = args.telemetry_path
    if not os.path.exists(telemetry_file_name):
        logger.error("Telemetry file %s does not exist", telemetry_file_name)
        return 1

    signal.signal(signal.SIGINT, signal_handler)

    try:
        logger.info("Opening telemetry buffer: %s", telemetry_file_name)
        logger.info("Press Ctrl+C to stop reading telemetry...")

        buffer = SharedRingBuffer(telemetry_file_name, version=TELEMETRY_VERSION)

        logger.info(
            "Successfully opened telemetry buffer (version: %d)", buffer.get_version()
        )
        logger.info("Buffer capacity: %d events", buffer.get_capacity())
        logger.info("Current events in buffer: %d", buffer.size())
        logger.info("Event structure size: %d bytes", ctypes.sizeof(NixlTelemetryEvent))

        event_count = 0

        while running:
            # Try to read an event from the buffer
            event = buffer.pop()
            if event:
                event_count += 1
                print_telemetry_event(event)
            else:
                # No events available, sleep briefly
                time.sleep(0.5)

        logger.info("\nTotal events read: %d", event_count)
        logger.info("Final buffer size: %d events", buffer.size())

    except Exception as e:
        logger.error("Error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
