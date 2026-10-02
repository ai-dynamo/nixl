# DOCA_MEMOS Backend Unit Tests

This directory contains comprehensive unit tests for the DOCA_MEMOS backend plugin with mocked DOCA library dependencies.

## Overview

The test suite uses mocked DOCA functions to test the backend without requiring actual DOCA hardware or libraries. This enables:

- **Fast execution**: No hardware initialization delays
- **Deterministic behavior**: Controlled test conditions
- **Complete coverage**: Test error paths that are hard to trigger with real hardware
- **CI/CD integration**: Run tests in any environment

## Test Structure

### Files

- **`doca_memos_mocks.h`**: Mock DOCA type definitions and function declarations
- **`doca_memos_mocks.cpp`**: Mock DOCA function implementations
- **`doca_memos_backend_test.cpp`**: Comprehensive test cases using Google Test
- **`meson.build`**: Build configuration for the test executable

### Mock Architecture

The mock layer provides:

1. **Mock DOCA types**: Lightweight replacements for DOCA structures
1. **Mock DOCA functions**: Implementations that simulate DOCA behavior
1. **Mock control interface**: `DocaMockControl` singleton for configuring mock behavior
1. **Simulated KV store**: In-memory map for testing STORE/RETRIEVE operations

### Mock Control

The `DocaMockControl` singleton allows tests to:

- **Configure return values**: Make any DOCA function succeed or fail
- **Simulate callbacks**: Register and trigger completion/error callbacks
- **Track submissions**: Monitor which tasks were submitted
- **Auto-complete tasks**: Automatically process tasks on `doca_pe_progress()`
- **Simulate KV storage**: Store and retrieve data in a mock key-value store

Example usage:

```cpp
// Make PE creation fail
DocaMockControl::instance().pe_create_result = DOCA_ERROR_NO_MEMORY;

// Enable auto-completion of tasks
DocaMockControl::instance().auto_complete_tasks = true;

// Pre-populate the KV store
DocaMockControl::instance().kv_store["test_key"] = {0x01, 0x02, 0x03};

// Reset all state
DocaMockControl::instance().reset();
```

## Test Categories

Use `--gtest_list_tests` to see the authoritative list. The groupings below
summarise the suite as currently implemented.

### Constructor / Destructor

- `ConstructorSuccess`, `ConstructorWithProgressThread`
- `ConstructorMissingDeviceName`, `ConstructorDocaKvdevCreateFailure`,
  `ConstructorDocaPECreateFailure` (assert on `getInitErr()`)
- `DestructorCleansPendingDeletes`

### Memory Registration

- `RegisterMemObjSeg`, `RegisterMemObjSegWithoutMetaInfo`
- `RegisterMemUnsupportedType`

### Transfer Path

- `PostXferHandleReuse` — completed handle re-used for a second submission
- `EmptyDescriptorList` — `prepXfer` rejects empty `local`/`remote`
- `KeyNotFoundOnRetrieve`, `KeyNotFoundOnRetrieveIgnored`
- `DeviceIOErrorOnStore`, `StoreReturnsTooBig`, `KeyTooLarge`

### Query Memory

- Assume-success modes: `QueryMemAssumeSuccessDefault`,
  `QueryMemAssumeSuccessExplicit`, `QueryMemInvalidMode`
- Actual EXIST mode: `QueryMemActualKeyExists`,
  `QueryMemActualKeyDoesNotExist`, `QueryMemActualMixed`

### Threading Modes

- `SingleThreadedMode`, `MultiThreadedModeWithProgressThread`,
  `MultiThreadedModeWithSyncMode`
- `ThreadedEngineQueuedRequestsComplete`

### Resource Exhaustion / Backpressure

- `TaskPoolExhaustion`, `TaskQueueFull`, `DeviceNotFound`
- Queued-request semantics: `QueuedRequestSucceedsAfterResourcesFreed`,
  `MultipleQueuedRequestsFIFOOrdering`, `QueuedRequestsWithSubmitFailures`,
  `ManyQueuedRequestsConcurrent`, `QueuedRequestNoPartialSubmission`,
  `QueuedRequestPartialRetryForwardProgress`

### Release / Cancel

- `ReleaseInFlightRequest`, `ReleasePendingRequest`, `ReleaseCompletedRequest`
- `ThreadedCancelBeforeProgressPickup`,
  `CancelAfterFatalSubmissionError_NoThread`,
  `CancelAfterFatalSubmissionError_Threaded`,
  `CancelWithInflightAndPendingTasks_Threaded`

## Running the Tests

### Build

```bash
meson setup build
meson compile -C build
```

### Run All DOCA KV Tests

```bash
meson test -C build --suite doca_memos
```

### Run with Verbose Output

```bash
meson test -C build --suite doca_memos --verbose
```

### Run Specific Test

```bash
./build/test/unit/plugins/doca_memos/doca_memos_backend_test --gtest_filter=DocMemosBackendTest.ConstructorSuccess
```

### Run with GTest Options

```bash
# List all tests
./build/test/unit/plugins/doca_memos/doca_memos_backend_test --gtest_list_tests

# Run tests matching a pattern
./build/test/unit/plugins/doca_memos/doca_memos_backend_test --gtest_filter="*RegisterMem*"

# Repeat tests
./build/test/unit/plugins/doca_memos/doca_memos_backend_test --gtest_repeat=100
```

## Extending the Tests

### Adding a New Test Case

1. **Add the test** to `doca_memos_backend_test.cpp`:

```cpp
TEST_F(DocMemosBackendTest, YourNewTest) {
    // Setup mock behavior
    DocaMockControl::instance().some_config = some_value;

    // Create engine
    createEngine();

    // Execute test logic
    // ...

    // Verify expectations
    EXPECT_EQ(actual, expected);
}
```

2. **Configure mocks** as needed:

```cpp
// Make a function fail
DocaMockControl::instance().task_submit_result = DOCA_ERROR_BAD_STATE;

// Enable auto-completion
DocaMockControl::instance().auto_complete_tasks = true;

// Add data to mock KV store
DocaMockControl::instance().kv_store["key"] = {0xAA, 0xBB};
```

3. **Use test fixtures** for common setup:

```cpp
// The SetUp() method runs before each test
void SetUp() override {
    DocaMockControl::instance().reset();
    setupInitParams();
}
```

### Adding Mock Functionality

If you need to mock a new DOCA function:

1. **Add the function signature** to `doca_memos_mocks.h`:

```cpp
extern "C" {
doca_error_t doca_new_function(doca_some_type *arg);
}
```

2. **Add control variables** to `DocaMockControl`:

```cpp
doca_error_t new_function_result = DOCA_SUCCESS;
```

3. **Implement the mock** in `doca_memos_mocks.cpp`:

```cpp
doca_error_t doca_new_function(doca_some_type *arg) {
    return DocaMockControl::instance().new_function_result;
}
```

4. **Reset in `DocaMockControl::reset()`**:

```cpp
new_function_result = DOCA_SUCCESS;
```

## Debugging Tips

### Enable Verbose Logging

The backend uses `NIXL_DEBUG`, `NIXL_INFO`, `NIXL_ERROR` macros. Configure logging level to see detailed execution flow.

### Inspect Mock State

Add debug output in tests:

```cpp
auto& mock = DocaMockControl::instance();
std::cout << "Submitted tasks: " << mock.submitted_tasks.size() << std::endl;
std::cout << "KV store size: " << mock.kv_store.size() << std::endl;
```

### Use GDB

```bash
gdb --args ./build/test/unit/plugins/doca_memos/doca_memos_backend_test --gtest_filter=SpecificTest
```

### Breakpoint in Test

```cpp
TEST_F(DocMemosBackendTest, DebugTest) {
    createEngine();
    __builtin_trap();  // Breakpoint here
    // ... rest of test
}
```

## Known Limitations

1. **Async completion**: mock auto-completion runs synchronously inside
   `doca_pe_progress()` rather than from a real device interrupt path.
1. **Threading**: the mock does not model device-side scheduling delays.
1. **Memory corruption**: the mock does not detect invalid pointers or
   out-of-bounds access into registered buffers.

## Contributing

When adding new functionality to the DOCA_MEMOS backend:

1. **Write tests first** (TDD approach)
1. **Cover error paths** explicitly
1. **Test threading modes** (single-threaded and multi-threaded)
1. **Document test intent** with clear comments
1. **Use descriptive test names**: `TEST_F(DocMemosBackendTest, DescriptiveNameOfWhatIsBeingTested)`

All backend changes should maintain or improve test coverage.
