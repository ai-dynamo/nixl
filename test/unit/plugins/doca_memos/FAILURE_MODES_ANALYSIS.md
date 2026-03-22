# DOCA KV Backend Failure Modes Analysis

This document analyzes all DOCA API calls in the backend and their possible failure modes to ensure comprehensive test coverage.

## DOCA Error Codes Reference

From `doca_error.h`, DOCA can return 32 different error codes:

| Code | Name | Description |
|------|------|-------------|
| 0 | `DOCA_SUCCESS` | Success |
| 1 | `DOCA_ERROR_UNKNOWN` | Unknown error |
| 2 | `DOCA_ERROR_NOT_PERMITTED` | Operation not permitted |
| 3 | `DOCA_ERROR_IN_USE` | Resource already in use |
| 4 | `DOCA_ERROR_NOT_SUPPORTED` | Operation not supported |
| 5 | `DOCA_ERROR_AGAIN` | Resource temporarily unavailable |
| 6 | `DOCA_ERROR_INVALID_VALUE` | Invalid input |
| 7 | `DOCA_ERROR_NO_MEMORY` | Memory allocation failure |
| 8 | `DOCA_ERROR_INITIALIZATION` | Resource initialization failure |
| 9 | `DOCA_ERROR_TIME_OUT` | Timer expired |
| 10 | `DOCA_ERROR_SHUTDOWN` | Shutdown in process |
| 11 | `DOCA_ERROR_CONNECTION_RESET` | Connection reset by peer |
| 12 | `DOCA_ERROR_CONNECTION_ABORTED` | Connection aborted |
| 13 | `DOCA_ERROR_CONNECTION_INPROGRESS` | Connection in progress |
| 14 | `DOCA_ERROR_NOT_CONNECTED` | Not connected |
| 15 | `DOCA_ERROR_NO_LOCK` | Unable to acquire lock |
| 16 | `DOCA_ERROR_NOT_FOUND` | Resource not found |
| 17 | `DOCA_ERROR_IO_FAILED` | I/O operation failed |
| 18 | `DOCA_ERROR_BAD_STATE` | Bad state |
| 19 | `DOCA_ERROR_UNSUPPORTED_VERSION` | Unsupported version |
| 20 | `DOCA_ERROR_OPERATING_SYSTEM` | OS call failure |
| 21 | `DOCA_ERROR_DRIVER` | Driver call failure |
| 22 | `DOCA_ERROR_UNEXPECTED` | Unexpected scenario |
| 23 | `DOCA_ERROR_ALREADY_EXIST` | Resource already exists |
| 24 | `DOCA_ERROR_FULL` | No more space |
| 25 | `DOCA_ERROR_EMPTY` | No entry available |
| 26 | `DOCA_ERROR_IN_PROGRESS` | Operation in progress |
| 27 | `DOCA_ERROR_TOO_BIG` | Operation too big |
| 28 | `DOCA_ERROR_AUTHENTICATION` | Authentication failure |
| 29 | `DOCA_ERROR_BAD_CONFIG` | Invalid configuration |
| 30 | `DOCA_ERROR_SKIPPED` | Data was dropped |
| 31 | `DOCA_ERROR_DEVICE_FATAL_ERROR` | Device fatal error |

## API Call Analysis

### 1. Initialization Path (Constructor)

#### 1.1 `doca_nvme_kernel_kvdev_create()`

**Location**: Line 247
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters or invalid device name
- `DOCA_ERROR_NO_MEMORY` - Allocation failure
- `DOCA_ERROR_NOT_FOUND` - Device doesn't exist
- `DOCA_ERROR_OPERATING_SYSTEM` - System call failed (e.g., open device)
- `DOCA_ERROR_DRIVER` - Driver communication failure
- `DOCA_ERROR_INITIALIZATION` - Failed to initialize device

**Current Tests**: ✅

- `ConstructorDocaKvdevCreateFailure` - Tests `DOCA_ERROR_INVALID_VALUE`

**Missing Tests**: ⚠️

- No tests for `DOCA_ERROR_NOT_FOUND` (device doesn't exist)
- No tests for `DOCA_ERROR_OPERATING_SYSTEM` (system failure)
- No tests for `DOCA_ERROR_NO_MEMORY` (allocation failure)

#### 1.2 `doca_nvme_kernel_kvdev_as_kvdev()`

**Location**: Line 255
**Possible Errors**:

- Returns NULL on failure (not a doca_error_t)

**Current Tests**: ✅ Implicitly tested in constructor success test

#### 1.3 `doca_kvdev_start()`

**Location**: Line 265
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - kvdev is NULL or invalid params
- `DOCA_ERROR_BAD_STATE` - Already started
- `DOCA_ERROR_NO_MEMORY` - Allocation failure
- `DOCA_ERROR_IO_FAILED` - Device I/O error
- `DOCA_ERROR_DEVICE_FATAL_ERROR` - Device is broken

**Current Tests**: ❌ No specific tests

**Missing Tests**: ⚠️

- Need tests for device start failures

#### 1.4 `doca_pe_create()`

**Location**: Line 293
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL output parameter
- `DOCA_ERROR_NO_MEMORY` - Allocation failure
- `DOCA_ERROR_INITIALIZATION` - PE initialization failed

**Current Tests**: ✅

- `ConstructorDocaPECreateFailure` - Tests `DOCA_ERROR_NO_MEMORY`

#### 1.5 `doca_kvdev_io_create()`

**Location**: Line 305
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters
- `DOCA_ERROR_NO_MEMORY` - Allocation failure
- `DOCA_ERROR_INITIALIZATION` - IO context init failed

**Current Tests**: ❌ No specific tests

**Missing Tests**: ⚠️

- Need tests for IO context creation failures

#### 1.6 `doca_kvdev_io_set_conf()`

**Location**: Line 320
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters or invalid num_tasks
- `DOCA_ERROR_BAD_STATE` - Context already started

**Current Tests**: ❌ No specific tests

**Missing Tests**: ⚠️

- Need tests for configuration failures

#### 1.7 `doca_kvdev_io_as_ctx()`

**Location**: Line 336
**Possible Errors**:

- Returns NULL on failure

**Current Tests**: ❌ No specific tests

#### 1.8 `doca_pe_connect_ctx()`

**Location**: Line 352
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters
- `DOCA_ERROR_IN_USE` - Context already connected
- `DOCA_ERROR_BAD_STATE` - Invalid state

**Current Tests**: ❌ No specific tests

**Missing Tests**: ⚠️

- Need tests for connection failures

#### 1.9 `doca_ctx_start()`

**Location**: Line 369
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL context
- `DOCA_ERROR_BAD_STATE` - Already started
- `DOCA_ERROR_INITIALIZATION` - Start failed

**Current Tests**: ❌ No specific tests

**Missing Tests**: ⚠️

- Need tests for context start failures

### 2. Destruction Path (Destructor)

#### 2.1 `doca_ctx_stop()`

**Location**: Line 429
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL context
- `DOCA_ERROR_BAD_STATE` - Already stopped or tasks pending

**Current Tests**: ❌ No specific tests

#### 2.2 `doca_kvdev_io_destroy()`

**Location**: Line 437
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameter
- `DOCA_ERROR_BAD_STATE` - Context not idle

**Current Tests**: ❌ No specific tests

#### 2.3 `doca_pe_destroy()`

**Location**: Line 450
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameter
- `DOCA_ERROR_BAD_STATE` - PE still has contexts

**Current Tests**: ❌ No specific tests

#### 2.4 `doca_kvdev_stop()`

**Location**: Line 457
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL kvdev
- `DOCA_ERROR_BAD_STATE` - Already stopped

**Current Tests**: ❌ No specific tests

#### 2.5 `doca_nvme_kernel_kvdev_destroy()`

**Location**: Line 461
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameter
- `DOCA_ERROR_IN_USE` - Still has references

**Current Tests**: ❌ No specific tests

### 3. Data Path Operations

#### 3.1 `doca_kv_task_alloc_init()`

**Location**: Line 764 (postXfer), Line 638 (queryMem)
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters
- `DOCA_ERROR_NO_MEMORY` - Task pool exhausted
- `DOCA_ERROR_BAD_STATE` - IO context not started

**Current Tests**: ✅ Partially

- `PostXferTaskAllocationFailure` - Tests task allocation failure

**Missing Tests**: ⚠️

- Need test for task pool exhaustion scenario
- Need test with different error codes

#### 3.2 `doca_kv_task_store_set_conf()`

**Location**: Line 776
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters, invalid key/value sizes
- `DOCA_ERROR_TOO_BIG` - Key or value too large
- `DOCA_ERROR_BAD_STATE` - Task not in correct state

**Current Tests**: ❌ No specific tests

**Missing Tests**: ⚠️

- Need tests for key too long
- Need tests for value too large
- Need tests for invalid parameters

#### 3.3 `doca_kv_task_retrieve_set_conf()`

**Location**: Line 783
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters, invalid key/buffer sizes
- `DOCA_ERROR_TOO_BIG` - Key too large
- `DOCA_ERROR_BAD_STATE` - Task not in correct state

**Current Tests**: ❌ No specific tests

#### 3.4 `doca_kv_task_exist_set_conf()`

**Location**: Line 651
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters, invalid key size
- `DOCA_ERROR_TOO_BIG` - Key too large
- `DOCA_ERROR_BAD_STATE` - Task not in correct state

**Current Tests**: ❌ No specific tests

#### 3.5 `doca_task_submit()`

**Location**: Line 800 (postXfer), Line 666 (queryMem)
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL task
- `DOCA_ERROR_BAD_STATE` - Task not configured or context not started
- `DOCA_ERROR_FULL` - Task queue full
- `DOCA_ERROR_IO_FAILED` - Submission hardware error

**Current Tests**: ✅ Partially

- `PostXferTaskSubmissionFailure` - Tests submission failure

**Missing Tests**: ⚠️

- Need test for `DOCA_ERROR_FULL` (queue full)
- Need test for `DOCA_ERROR_BAD_STATE`

#### 3.6 `doca_pe_progress()`

**Location**: Line 172 (progress thread), Line 698 (queryMem)
**Possible Errors**:

- Returns 0 (no progress) or >0 (number of completions)
- No error codes, but can trigger completion callbacks with errors

**Current Tests**: ✅ Implicitly tested via auto_complete_tasks

#### 3.7 `doca_task_free()`

**Location**: Line 80, 108 (callbacks)
**Possible Errors**:

- No documented errors (void function)

#### 3.8 `doca_pe_get_notification_handle()`

**Location**: Line 685
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL parameters
- `DOCA_ERROR_NOT_SUPPORTED` - Notification not supported
- `DOCA_ERROR_OPERATING_SYSTEM` - Failed to get FD

**Current Tests**: ❌ No specific tests

**Missing Tests**: ⚠️

- Need test for `QueryMemNotificationHandleFailure`

#### 3.9 `doca_pe_request_notification()`

**Location**: Line 706
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL PE
- `DOCA_ERROR_BAD_STATE` - Invalid state

**Current Tests**: ❌ No specific tests

#### 3.10 `doca_pe_clear_notification()`

**Location**: Line 714
**Possible Errors**:

- `DOCA_ERROR_INVALID_VALUE` - NULL PE or invalid handle
- `DOCA_ERROR_OPERATING_SYSTEM` - Failed to clear

**Current Tests**: ❌ No specific tests

### 4. Task Completion Callbacks

#### 4.1 Task Completion (Success)

**Location**: Lines 59-83
**Possible Scenarios**:

- Task completes successfully
- All tasks in batch complete
- Partial completion

**Current Tests**: ✅

- Auto-complete mechanism tests this

#### 4.2 Task Error (Failure)

**Location**: Lines 87-113
**Possible Scenarios**:

- Task fails with various error codes
- `DOCA_ERROR_NOT_FOUND` - Key doesn't exist (RETRIEVE/EXIST)
- `DOCA_ERROR_IO_FAILED` - Device I/O error
- `DOCA_ERROR_TOO_BIG` - Value too large
- `DOCA_ERROR_DEVICE_FATAL_ERROR` - Device failure
- Any other DOCA error

**Current Tests**: ✅ Partially

- `ErrorCallback` test exists but marked TODO

**Missing Tests**: ⚠️

- Need tests for specific error scenarios
- Need test for key not found
- Need test for device errors

## Test Coverage Summary

### ✅ Well Covered

1. Basic constructor success
1. Task allocation failure
1. Task submission failure
1. Basic completion callbacks

### ⚠️ Partially Covered

1. Constructor failures (only 2 of 9 failure points tested)
1. Data path errors (only allocation/submission tested)

### ❌ Not Covered

1. Destructor error paths
1. Device start/stop failures
1. Context configuration failures
1. Task configuration errors (key too long, value too big)
1. Task queue full scenarios
1. Event notification failures
1. Specific task error types (NOT_FOUND, IO_FAILED, etc.)
1. Resource exhaustion scenarios
1. State machine violations (double start, stop before idle, etc.)

## Recommendations

### Priority 1 (Critical Paths)

1. **Task Pool Exhaustion**: Test `doca_kv_task_alloc_init()` with `DOCA_ERROR_FULL`
1. **Key Not Found**: Test RETRIEVE/EXIST with `DOCA_ERROR_NOT_FOUND`
1. **Device I/O Errors**: Test with `DOCA_ERROR_IO_FAILED`
1. **Task Queue Full**: Test `doca_task_submit()` with `DOCA_ERROR_FULL`

### Priority 2 (Initialization Robustness)

5. **Device Not Found**: Test with non-existent device name
1. **Device Start Failure**: Test `doca_kvdev_start()` failures
1. **IO Context Creation**: Test `doca_kvdev_io_create()` failures
1. **Context Start**: Test `doca_ctx_start()` failures

### Priority 3 (Edge Cases)

9. **Key Too Long**: Test with oversized keys
1. **Value Too Large**: Test with oversized values
1. **Bad State Transitions**: Test double-start, early destroy, etc.
1. **Notification Failures**: Test event-driven mode failures

### Priority 4 (Cleanup Robustness)

13. **Destructor Error Handling**: While less critical (destructor is best-effort), test cleanup failures

## Implementation Plan

1. **Expand MockControl**: Add error injection for all DOCA APIs
1. **Add Error Scenario Tests**: Create tests for each Priority 1-3 item
1. **Add Stress Tests**: Test resource exhaustion scenarios
1. **Add State Machine Tests**: Test invalid state transitions
1. **Document Coverage**: Update test README with coverage matrix
