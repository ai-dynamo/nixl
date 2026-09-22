// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! RAII guard for serializing and restoring environment variables in tests.

#![allow(dead_code)]

use std::cell::RefCell;
use std::env;
use std::sync::{Mutex, MutexGuard};

/// Process-global lock serializing all `EnvGuard` users, so tests that mutate
/// shared environment variables cannot race under the parallel test runner.
/// `Mutex::new` is `const`, so a const-initialized static suffices (no lazy init).
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard for tests that mutate process-global environment variables.
///
/// On construction it acquires a shared lock (serializing all `EnvGuard` users)
/// and snapshots the current values of the requested variables. `set`/`remove`
/// also snapshot a key on first mutation, so any variable they touch is restored
/// even if it was not passed to `new`. On drop it restores each recorded variable
/// to its previous value, or removes it if it was unset. Restoration runs on drop,
/// so it is panic-safe: a failing assertion still leaves the environment clean for
/// subsequent tests.
pub struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: RefCell<Vec<(String, Option<String>)>>,
}

impl EnvGuard {
    /// Locks the shared env mutex and snapshots the given variables.
    pub fn new<I, S>(vars: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let saved: Vec<(String, Option<String>)> = vars
            .into_iter()
            .map(|var| {
                let key = var.into();
                let prev = env::var(&key).ok();
                (key, prev)
            })
            .collect();
        Self {
            _lock: lock,
            saved: RefCell::new(saved),
        }
    }

    /// Sets a variable for the lifetime of the guard.
    pub fn set(&self, key: &str, value: &str) {
        self.snapshot(key);
        env::set_var(key, value);
    }

    /// Removes a variable for the lifetime of the guard.
    pub fn remove(&self, key: &str) {
        self.snapshot(key);
        env::remove_var(key);
    }

    /// Records `key`'s current value once, so Drop can restore it even when the
    /// key was not passed to `new`.
    fn snapshot(&self, key: &str) {
        let mut saved = self.saved.borrow_mut();
        if !saved.iter().any(|(k, _)| k == key) {
            saved.push((key.to_string(), env::var(key).ok()));
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, prev) in self.saved.get_mut().iter() {
            match prev {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EnvGuard;
    use std::env;

    // Each test uses a unique throwaway key; EnvGuard's own mutex serializes
    // these with every other guard user so they cannot race.

    // A key not listed in new() is still snapshotted on first mutation, so an
    // originally-unset key is removed again on drop (the leak the guard prevents).
    #[test]
    fn set_restores_unsnapshotted_key_to_unset() {
        let key = "NIXL_ENVGUARD_SELFTEST_UNSNAPSHOTTED";
        env::remove_var(key);
        {
            let guard = EnvGuard::new(Vec::<String>::new());
            guard.set(key, "value");
            assert_eq!(env::var(key).as_deref(), Ok("value"));
        }
        assert!(env::var(key).is_err(), "key must be unset after drop");
    }

    // A pre-existing value is restored after the guard mutates it.
    #[test]
    fn set_restores_preexisting_value() {
        let key = "NIXL_ENVGUARD_SELFTEST_PREEXISTING";
        env::set_var(key, "original");
        {
            let guard = EnvGuard::new([key]);
            guard.set(key, "changed");
            assert_eq!(env::var(key).as_deref(), Ok("changed"));
        }
        assert_eq!(env::var(key).as_deref(), Ok("original"));
        env::remove_var(key);
    }

    // remove() inside the guard is undone on drop.
    #[test]
    fn remove_restores_value_on_drop() {
        let key = "NIXL_ENVGUARD_SELFTEST_REMOVE";
        env::set_var(key, "original");
        {
            let guard = EnvGuard::new([key]);
            guard.remove(key);
            assert!(env::var(key).is_err());
        }
        assert_eq!(env::var(key).as_deref(), Ok("original"));
        env::remove_var(key);
    }
}
