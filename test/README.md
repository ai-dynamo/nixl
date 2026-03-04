# Test Descriptions

Here are all the explained tests in this directory. There are more specific unit tests in unit/plugins and unit/utils.

- test/agent_example.cpp - Single threaded test of the nixlAgent API
- test/desc_example.cpp - Test of nixl descriptors and DescList
- test/metadata_streamer.cpp - Single or Multi node test of nixl metadata streamer
- test/nixl_test.cpp - Single or Multi node test of nixlAgent API
- test/ucx_backend_test.cpp - Single threaded test of all the ucxBackendEngine functionality
- test/ucx_backend_multi.cpp - Multi threaded test of UCX connection setup/teardown
- test/python/nixl_bindings_test.py - single threaded Python test of nixlAgent, nixlBasicDesc, and nixlDescList python bindings

## Google Test Framework (gtest)

The project includes comprehensive unit tests using Google Test framework located in `test/gtest/`:

- test/gtest/telemetry_test.cpp - Comprehensive tests for NIXL telemetry functionality including initialization, data tracking, thread safety, and edge cases
- test/gtest/query_mem.cpp - Tests for memory query functionality
- test/gtest/error_handling.cpp - Tests for error handling and status codes
- test/gtest/test_transfer.cpp - Tests for data transfer operations
- test/gtest/plugin_manager.cpp - Tests for plugin management
- test/gtest/multi_threading.cpp - Multi-threaded test scenarios
- test/gtest/metadata_exchange.cpp - Tests for metadata exchange functionality

To run the gtest suite:
```bash
# Build the project
meson setup build
meson compile -C build

# Run all tests
cd build
./gtest

# Run specific test categories
./gtest --gtest_filter="TelemetryTest*"  # Run only telemetry tests
./gtest --gtest_filter="QueryMemTest*"   # Run only query memory tests
```
