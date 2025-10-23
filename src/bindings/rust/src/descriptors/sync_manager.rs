// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synchronization manager for frontend-backend data consistency

use std::cell::Cell;

/// Trait for types that can synchronize their state to a backend
pub trait BackendSyncable {
    type Backend;
    type Error;
    fn sync_to_backend(&self, backend: &Self::Backend) -> Result<(), Self::Error>;
}

/// Manager that enforces correct synchronization between frontend and backend
pub struct SyncManager<T: BackendSyncable> {
    data: T,
    backend: T::Backend,
    dirty: Cell<bool>,
}

impl<T: BackendSyncable> SyncManager<T> {
    /// Creates a new sync manager (starts dirty to ensure first sync)
    pub fn new(data: T, backend: T::Backend) -> Self {
        Self {
            data,
            backend,
            dirty: Cell::new(true),
        }
    }

    /// Mutates the frontend data (marks as dirty)
    pub fn mutate<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        self.dirty.set(true);
        f(&mut self.data)
    }

    /// Provides access to both data and backend after ensuring synchronization
    pub fn with_backend<F, R>(&self, f: F) -> Result<R, T::Error>
    where
        F: FnOnce(&T, &T::Backend) -> R,
    {
        self.ensure_synced()?;
        Ok(f(&self.data, &self.backend))
    }

    /// Provides read-only access to the frontend data (no sync)
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Provides read-only access to the backend after ensuring synchronization
    pub fn backend(&self) -> Result<&T::Backend, T::Error> {
        self.ensure_synced()?;
        Ok(&self.backend)
    }

    fn ensure_synced(&self) -> Result<(), T::Error> {
        if self.dirty.get() {
            self.data.sync_to_backend(&self.backend)?;
            self.dirty.set(false);
        }
        Ok(())
    }
}

