#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import signal
import time
import mmap
import ctypes
from ctypes import *
from datetime import datetime
from pathlib import Path
import argparse
import errno

# Constants from nixl_types.h
TELEMETRY_PREFIX = "nixl_telemetry"
TELEMETRY_VERSION = 1
TELEMETRY_BUFFER_SIZE = 4096
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

def signal_handler(signum, frame):
    """Signal handler for Ctrl+C"""
    global running
    if signum == signal.SIGINT:
        print("\nReceived Ctrl+C, shutting down...")
        running = False



class NixlTelemetryEvent(ctypes.Structure):
    """Python equivalent of nixlTelemetryEvent struct"""
    # Remove _pack_ = 1 to match C++ alignment
    _pack_ = 1
    _fields_ = [
        ("timestamp_us", ctypes.c_uint64),                          # Event timestamp in microseconds
        ("category", ctypes.c_int),                                 # Main event category for filtering
        ("event_name", ctypes.c_char * MAX_EVENT_NAME_LEN),         # Detailed event name/identifier
        ("_padding", ctypes.c_uint32),                              # Padding to match 8-byte alignment
        ("value", ctypes.c_uint64)                                # Numeric value associated with the event
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
        ("mask", ctypes.c_size_t)
    ]

class SharedRingBuffer:
    """Python wrapper for the C++ SharedRingBuffer class"""
    
    def __init__(self, file_path, version=1):
        self.file_path = file_path
        self.version = version
        self.file_fd = None
        self.mmap_obj = None
        self.header = None
        self.data = None
        self.buffer_size = None  # Will be auto-detected from header
        
        self._open_file()
        self._map_memory()
    
    def _open_file(self):
        """Open existing file"""
        self.file_fd = os.open(self.file_path, os.O_RDWR)
        if self.file_fd == -1:
            raise RuntimeError(f"Failed to open file for shared memory: {os.strerror(errno.errno)}")
    
    def _map_memory(self):
        """Map the file into memory"""
        # For reading, first map just the header to get the size
        self._map_header_only()
    
    def _map_header_only(self):
        """Map only the header to read buffer size"""
        # Map just the header first
        header_mmap = mmap.mmap(self.file_fd, ctypes.sizeof(BufferHeader), 
                               mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        
        # Read the header to get buffer size
        temp_header = BufferHeader.from_buffer(header_mmap)
        
        # Check version compatibility
        if temp_header.version != self.version:
            del temp_header
            header_mmap.close()
            raise RuntimeError(f"Version mismatch: expected {self.version}, got {temp_header.version}")
        
        # Get buffer size from header
        self.buffer_size = temp_header.capacity
        print(f"Auto-detected buffer size: {self.buffer_size}")
        
        # delete so we can close the mmap object and open with correct size
        del temp_header
        header_mmap.close()
        
        # Now map the entire buffer
        total_size = ctypes.sizeof(BufferHeader) + ctypes.sizeof(NixlTelemetryEvent) * self.buffer_size
        self.mmap_obj = mmap.mmap(self.file_fd, total_size, 
                                 mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        
        # Create ctypes pointers to the mapped memory
        self.header = BufferHeader.from_buffer(self.mmap_obj)
        data_offset = ctypes.sizeof(BufferHeader)
        self.data = (NixlTelemetryEvent * self.buffer_size).from_buffer(
            self.mmap_obj, data_offset)
    
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
        
        # Check if buffer is empty
        if read_pos == self.header.write_pos:
            return None  # Buffer empty
        
        # Read data
        event = self.data[read_pos]
        
        # Update read position
        next_read = (read_pos + 1) & self.header.mask
        self.header.read_pos = next_read
        
        return event
    
    def __del__(self):
        """Cleanup resources"""
        # if self.mmap_obj:
        #     self.mmap_obj.close()
        if self.file_fd is not None and self.file_fd != -1:
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
        NIXL_TELEMETRY_CUSTOM: "CUSTOM"
    }
    return category_strings.get(category, f"UNKNOWN_CATEGORY_{category}")

def print_telemetry_event(event):
    """Print telemetry event in a formatted way"""
    print("\n=== NIXL Telemetry Event ===")
    print(f"Timestamp: {format_timestamp(event.timestamp_us)}")
    
    # Decode event name
    event_name = event.event_name.decode('utf-8').rstrip('\x00')
    category_str = get_telemetry_category_string(event.category)
    
    print(f"Category: {category_str}")
    print(f"Event: {event_name}")
    print(f"Value: {event.value}")
    print("===========================")

def check_if_process_running(pid):
    """Check if a process is running"""
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        if e.errno == errno.EPERM:
            return True
        return False

def look_for_stat_active_telemetry_files(telemetry_path, read_any_file=False):
    """Look for active telemetry files in the given directory"""
    module_name_size = len(TELEMETRY_PREFIX)
    pid_offset = module_name_size + 1  # file name is like TELEMETRY_PREFIX.pid
    
    stats_path = Path(telemetry_path)
    if not stats_path.exists() or not stats_path.is_dir():
        print(f"Cannot open directory {stats_path}")
        return ""
    
    for entry in stats_path.iterdir():
        filename = entry.name
        # Check if file name starts with TELEMETRY_PREFIX
        if filename.startswith(TELEMETRY_PREFIX):
            if read_any_file:
                return str(entry)
            
            # Get pid from file name
            try:
                pid_str = filename[pid_offset:]
                pid = int(pid_str)
                
                if not check_if_process_running(pid):
                    continue
                return str(entry)
            except (ValueError, IndexError):
                continue
    
    return ""

def usage():
    """Print usage information"""
    print("Usage: telemetry_reader.py <telemetry_folder_path> [read_any_file]")
    print("Options:")
    print("  <telemetry_folder_path>    Path to the telemetry folder")
    print("  <read_any_file>            Read telemetry data from any file (not only active")
    print("                             process) in the folder (0 - false, 1 - true, default: 0)")
    sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='NIXL Telemetry Reader')
    parser.add_argument('--telemetry_path', help='Path to the telemetry folder', required=True)
    parser.add_argument('--read-any-file', type=int, default=0, 
                       help='Read telemetry data from any file (0 - false, 1 - true)')

    args = parser.parse_args()
    
    print(f"Telemetry path: {args.telemetry_path}")
    read_any_file = bool(args.read_any_file)
    if os.path.isfile(args.telemetry_path):
        telemetry_file_name = args.telemetry_path
    else:
        telemetry_file_name = look_for_stat_active_telemetry_files(args.telemetry_path, read_any_file)
    if not telemetry_file_name:
        print("No active telemetry files found")
        return 1
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print(f"Opening telemetry buffer: {telemetry_file_name}")
        print("Press Ctrl+C to stop reading telemetry...")
        
        # Open the shared memory buffer for reading
        buffer = SharedRingBuffer(telemetry_file_name, version=TELEMETRY_VERSION)
        
        print(f"Successfully opened telemetry buffer (version: {buffer.get_version()})")
        print(f"Buffer capacity: {buffer.get_capacity()} events")
        print(f"Current events in buffer: {buffer.size()}")
        print(f"Event structure size: {ctypes.sizeof(NixlTelemetryEvent)} bytes")
        print(f"Expected size without packing: 8+4+32+8 = 52 bytes")
        print(f"Expected size with 8-byte alignment: 8+4+4(padding)+32+8 = 56 bytes")
        
        # Test structure layout
        test_event = NixlTelemetryEvent()
        # print(f"Field offsets: timestamp={ctypes.addressof(test_event.timestamp_us)}, category={ctypes.addressof(test_event.category)}, event_name={ctypes.addressof(test_event.event_name)}, value={ctypes.addressof(test_event.value)}")
        
        event_count = 0
        
        while running:
            # Try to read an event from the buffer
            event = buffer.pop()
            if event:
                event_count += 1
                print_telemetry_event(event)
            else:
                # No events available, sleep briefly
                time.sleep(0.1)
        
        print(f"\nTotal events read: {event_count}")
        print(f"Final buffer size: {buffer.size()} events")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 