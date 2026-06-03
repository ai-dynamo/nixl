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

use std::env;
use std::sync::{Mutex, MutexGuard};

/// Process-global lock serializing all `EnvGuard` users, so tests that mutate
/// shared environment variables cannot race under the parallel test runner.
/// `Mutex::new` is `const`, so a const-initialized static suffices (no lazy init).
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard for tests that mutate process-global environment variables.
///
/// On construction it acquires a shared lock (serializing all `EnvGuard` users)
/// and snapshots the current values of the requested variables. On drop it
/// restores each variable to its previous value, or removes it if it was unset.
/// Restoration runs on drop, so it is panic-safe: a failing assertion still
/// leaves the environment clean for subsequent tests.
pub struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(String, Option<String>)>,
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
        let saved = vars
            .into_iter()
            .map(|var| {
                let key = var.into();
                let prev = env::var(&key).ok();
                (key, prev)
            })
            .collect();
        Self { _lock: lock, saved }
    }

    /// Sets a variable for the lifetime of the guard.
    pub fn set(&self, key: &str, value: &str) {
        env::set_var(key, value);
    }

    /// Removes a variable for the lifetime of the guard.
    pub fn remove(&self, key: &str) {
        env::remove_var(key);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, prev) in &self.saved {
            match prev {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
    }
}
