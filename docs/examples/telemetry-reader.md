---
title: Telemetry Reader
description: Read and process NIXL telemetry events programmatically using the shared memory telemetry buffer.
---

The examples below are taken from the `examples/` directory in the [NIXL repository](https://github.com/ai-dynamo/nixl), annotated with inline explanations.

**What you'll learn:** How to read and process NIXL telemetry events programmatically using the shared memory telemetry buffer.

NIXL writes telemetry events to a shared memory ring buffer. The telemetry reader examples show how to open this buffer, read events as they arrive, and format them for display or processing. This is useful for monitoring transfers, debugging performance issues, and building custom telemetry dashboards.

<CodeBlocks>
```python title="Python"
import ctypes
import mmap
import os
import signal
import time
from datetime import datetime

# Constants matching the C++ telemetry event structure
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

# Step 1: Define the telemetry event structure
# This must match the C++ nixlTelemetryEvent struct layout
class NixlTelemetryEvent(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp_us", ctypes.c_uint64),
        ("category", ctypes.c_int),
        ("event_name", ctypes.c_char * MAX_EVENT_NAME_LEN),
        ("_padding", ctypes.c_uint32),
        ("value", ctypes.c_uint64),
    ]

# Step 2: Define the ring buffer header structure
class BufferHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("write_pos", ctypes.c_size_t),
        ("read_pos", ctypes.c_size_t),
        ("version", ctypes.c_uint32),
        ("expected_version", ctypes.c_uint32),
        ("capacity", ctypes.c_size_t),
        ("mask", ctypes.c_size_t),
    ]

# Step 3: Open and memory-map the telemetry buffer
# The telemetry file is created by a Transfer Agent when telemetry is enabled
telemetry_path = "/tmp/nixl_telemetry"  # Default path
fd = os.open(telemetry_path, os.O_RDWR)

# Map the header first to read the buffer capacity
header_mmap = mmap.mmap(
    fd, ctypes.sizeof(BufferHeader),
    mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
)
header = BufferHeader.from_buffer(header_mmap)
buffer_size = header.capacity
del header
header_mmap.close()

# Map the entire buffer (header + event data)
total_size = (
    ctypes.sizeof(BufferHeader)
    + ctypes.sizeof(NixlTelemetryEvent) * buffer_size
)
mmap_obj = mmap.mmap(
    fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
)

header = BufferHeader.from_buffer(mmap_obj)
data_offset = ctypes.sizeof(BufferHeader)
events = (NixlTelemetryEvent * buffer_size).from_buffer(mmap_obj, data_offset)

# Step 4: Read events in a loop
# Events are consumed from the ring buffer using read_pos/write_pos
category_names = {
    0: "MEMORY", 1: "TRANSFER", 2: "CONNECTION", 3: "BACKEND",
    4: "ERROR", 5: "PERFORMANCE", 6: "SYSTEM", 7: "CUSTOM",
}

running = True
event_count = 0

while running:
    read_pos = header.read_pos
    if read_pos != header.write_pos:
        # Pop the next event from the ring buffer
        event = events[read_pos]
        header.read_pos = (read_pos + 1) & header.mask
        event_count += 1

        # Step 5: Format and display the event
        timestamp = datetime.fromtimestamp(event.timestamp_us / 1_000_000)
        event_name = event.event_name.decode("utf-8").rstrip("\x00")
        category = category_names.get(event.category, "UNKNOWN")

        print(f"[{timestamp}] {category}: {event_name} = {event.value}")
    else:
        time.sleep(0.5)  # No events available, wait

print(f"Total events read: {event_count}")
os.close(fd)
```

```cpp title="C++"
#include <iostream>
#include <signal.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>
#include <filesystem>
#include <string>

#include "common/cyclic_buffer.h"
#include "telemetry_event.h"

namespace fs = std::filesystem;

volatile sig_atomic_t g_running = true;

// Signal handler for graceful shutdown with Ctrl+C
void signal_handler(int signal) {
    if (signal == SIGINT) {
        g_running = false;
    }
}

// Format a microsecond timestamp to a readable string
std::string format_timestamp(uint64_t timestamp_us) {
    auto time_point =
        std::chrono::system_clock::time_point(
            std::chrono::microseconds(timestamp_us));
    auto time_t = std::chrono::system_clock::to_time_t(time_point);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(6)
       << (timestamp_us % 1000000);
    return ss.str();
}

// Print a single telemetry event
void print_telemetry_event(const nixlTelemetryEvent& event) {
    std::cout << "\n=== NIXL Telemetry Event ===" << std::endl;
    std::cout << "Timestamp: "
              << format_timestamp(event.timestampUs_) << std::endl;
    std::cout << "Category: "
              << nixlEnumStrings::telemetryCategoryStr(event.category_)
              << std::endl;
    std::cout << "Event name: " << event.eventName_ << std::endl;
    std::cout << "Value: " << event.value_ << std::endl;
    std::cout << "===========================" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: telemetry_reader <telemetry_file_path>"
                  << std::endl;
        return 0;
    }

    // Step 1: Verify the telemetry file exists
    auto telemetry_path = argv[1];
    if (!fs::exists(telemetry_path)) {
        std::cerr << "Telemetry file does not exist: "
                  << telemetry_path << std::endl;
        return 1;
    }

    signal(SIGINT, signal_handler);

    // Step 2: Open the shared ring buffer
    // The buffer is memory-mapped from the telemetry file created by
    // a Transfer Agent. The false parameter means we are a reader (not writer).
    sharedRingBuffer<nixlTelemetryEvent> buffer(
        telemetry_path, false, TELEMETRY_VERSION);

    std::cout << "Buffer capacity: " << buffer.capacity()
              << " events" << std::endl;

    // Step 3: Read events in a loop until Ctrl+C
    nixlTelemetryEvent event;
    uint64_t event_count = 0;

    while (g_running) {
        // pop() returns true if an event was available
        if (buffer.pop(event)) {
            event_count++;
            print_telemetry_event(event);
        } else {
            // No events available, sleep briefly
            std::this_thread::sleep_for(
                std::chrono::milliseconds(500));
        }
    }

    // Step 4: Print summary on exit
    std::cout << "\nTotal events read: " << event_count << std::endl;
    std::cout << "Final buffer size: " << buffer.size()
              << " events" << std::endl;

    return 0;
}
```
</CodeBlocks>

**Expected output:**

```
=== NIXL Telemetry Event ===
Timestamp: 2025-06-15 14:30:22.123456
Category: TRANSFER
Event name: xfer_posted
Value: 1024
===========================

=== NIXL Telemetry Event ===
Timestamp: 2025-06-15 14:30:22.124789
Category: TRANSFER
Event name: xfer_completed
Value: 1024
===========================

Total events read: 2
```

<Tip>
For the full telemetry architecture, event categories, Prometheus integration, and configuration details, see the [Telemetry Guide](../user-guide/telemetry).
</Tip>
