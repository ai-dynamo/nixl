// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nixl_sys::{BackendSyncable, SyncManager};
use std::cell::Cell;
use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
enum TestError {
    SyncFailed,
}

#[derive(Debug, Clone)]
struct DummyBackend {
    sync_calls: Rc<Cell<u32>>,
}

#[derive(Debug, Clone)]
struct DummyData {
    value: Cell<i32>,
    data_calls: Rc<Cell<u32>>,
    should_fail: Cell<bool>,
}

impl BackendSyncable for DummyData {
    type Backend = DummyBackend;
    type Error = TestError;

    fn sync_to_backend(&self, backend: &Self::Backend) -> Result<(), Self::Error> {
        if self.should_fail.get() {
            return Err(TestError::SyncFailed);
        }
        self.data_calls.set(self.data_calls.get() + 1);
        backend
            .sync_calls
            .set(backend.sync_calls.get() + 1);
        Ok(())
    }
}

#[test]
fn sync_manager_first_access_triggers_single_sync() {
    let data_calls = Rc::new(Cell::new(0));
    let backend_calls = Rc::new(Cell::new(0));

    let data = DummyData {
        value: Cell::new(0),
        data_calls: data_calls.clone(),
        should_fail: Cell::new(false),
    };
    let backend = DummyBackend {
        sync_calls: backend_calls.clone(),
    };
    let mgr = SyncManager::new(data, backend);

    // First access -> sync happens
    let _ = mgr.backend().unwrap().sync_calls.get();
    assert_eq!(data_calls.get(), 1);
    assert_eq!(backend_calls.get(), 1);

    // Second access without modification -> no sync
    let _ = mgr.backend().unwrap().sync_calls.get();
    assert_eq!(data_calls.get(), 1);
    assert_eq!(backend_calls.get(), 1);
}

#[test]
fn sync_manager_resyncs_after_modify() {
    let data_calls = Rc::new(Cell::new(0));
    let backend_calls = Rc::new(Cell::new(0));

    let data = DummyData {
        value: Cell::new(0),
        data_calls: data_calls.clone(),
        should_fail: Cell::new(false),
    };
    let backend = DummyBackend {
        sync_calls: backend_calls.clone(),
    };
    let mut mgr = SyncManager::new(data, backend);

    // Ensure initial sync
    let _ = mgr.backend().unwrap();
    assert_eq!(data_calls.get(), 1);

    // Modify -> marks dirty
    mgr.data_mut().value.set(42);

    // Next access -> resync
    let _ = mgr.backend().unwrap();
    assert_eq!(data_calls.get(), 2);
    assert_eq!(backend_calls.get(), 2);
}

#[test]
fn sync_manager_backend_accessor_triggers_sync() {
    let data_calls = Rc::new(Cell::new(0));
    let backend_calls = Rc::new(Cell::new(0));

    let data = DummyData {
        value: Cell::new(0),
        data_calls: data_calls.clone(),
        should_fail: Cell::new(false),
    };
    let backend = DummyBackend {
        sync_calls: backend_calls.clone(),
    };
    let mgr = SyncManager::new(data, backend);

    // Read-only data access should not sync
    let _ = mgr.data();
    assert_eq!(data_calls.get(), 0);
    assert_eq!(backend_calls.get(), 0);

    // Backend accessor should sync
    let _ = mgr.backend().unwrap();
    assert_eq!(data_calls.get(), 1);
    assert_eq!(backend_calls.get(), 1);
}

#[test]
fn sync_manager_error_is_propagated_and_retry_succeeds() {
    let data_calls = Rc::new(Cell::new(0));
    let backend_calls = Rc::new(Cell::new(0));

    let data = DummyData {
        value: Cell::new(0),
        data_calls: data_calls.clone(),
        should_fail: Cell::new(false),
    };
    let backend = DummyBackend {
        sync_calls: backend_calls.clone(),
    };
    let mut mgr = SyncManager::new(data, backend);

    // Configure to fail on first sync
    mgr.data_mut().should_fail.set(true);

    // Attempting to access backend should error
    let err = mgr.backend().unwrap_err();
    assert_eq!(err, TestError::SyncFailed);
    // No successful sync occurred
    assert_eq!(data_calls.get(), 0);
    assert_eq!(backend_calls.get(), 0);

    // Clear failure and try again
    mgr.data_mut().should_fail.set(false);
    let _ = mgr.backend().unwrap();
    assert_eq!(data_calls.get(), 1);
    assert_eq!(backend_calls.get(), 1);
}
