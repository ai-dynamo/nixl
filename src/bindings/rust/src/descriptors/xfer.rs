// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use super::*;
use std::cell::Cell;

/// Public transfer descriptor used for indexing and comparisons
#[derive(Debug, Clone, PartialEq)]
pub struct XferDescriptor {
    pub addr: usize,
    pub len: usize,
    pub dev_id: u64,
}

/// A safe wrapper around a NIXL transfer descriptor list
pub struct XferDescList<'a> {
    inner: NonNull<bindings::nixl_capi_xfer_dlist_s>,
    _phantom: PhantomData<&'a dyn NixlDescriptor>,
    // Track descriptors internally for comparison purposes
    descriptors: Vec<XferDescriptor>,
    mem_type: MemType,
    // Private: backend C list needs to be resynchronized from descriptors
    dirty: Cell<bool>,
}

impl<'a> XferDescList<'a> {
    /// Creates a new transfer descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status = unsafe {
            nixl_capi_create_xfer_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist)
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, dlist is non-null
                let inner = unsafe { NonNull::new_unchecked(dlist) };
                Ok(Self {
                    inner,
                    _phantom: PhantomData,
                    descriptors: Vec::new(),
                    mem_type,
                    dirty: Cell::new(false),
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    pub fn as_ptr(&self) -> *mut bindings::nixl_capi_xfer_dlist_s {
        let _ = self.ensure_synced();
        self.inner.as_ptr()
    }

    /// Returns the memory type of the transfer descriptor list
    pub fn get_type(&self) -> Result<MemType, NixlError> { Ok(self.mem_type) }

    /// Adds a descriptor to the list
    pub fn add_desc(&mut self, addr: usize, len: usize, dev_id: u64) -> Result<(), NixlError> {
        // Vector is the source of truth; rebuild backend lazily
        self.descriptors.push(XferDescriptor { addr, len, dev_id });
        self.dirty.set(true);
        Ok(())
    }

    /// Returns true if the list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Returns the number of descriptors in the list
    pub fn desc_count(&self) -> Result<usize, NixlError> { Ok(self.descriptors.len()) }

    /// Returns the number of descriptors in the list
    pub fn len(&self) -> Result<usize, NixlError> { Ok(self.descriptors.len()) }

     /// Trims the list to the given size
     pub fn trim(&mut self) -> Result<(), NixlError> { self.descriptors.shrink_to_fit(); self.dirty.set(true); Ok(()) }

    /// Removes the descriptor at the given index
    pub fn rem_desc(&mut self, index: i32) -> Result<(), NixlError> { if index < 0 { return Err(NixlError::InvalidParam); } let idx = index as usize; if idx >= self.descriptors.len() { return Err(NixlError::InvalidParam); } self.descriptors.remove(idx); self.dirty.set(true); Ok(()) }

    /// Clears all descriptors from the list
    pub fn clear(&mut self) -> Result<(), NixlError> { self.descriptors.clear(); self.dirty.set(true); Ok(()) }

    /// Prints the list contents
    pub fn print(&self) -> Result<(), NixlError> {
        self.ensure_synced()?;
        let status = unsafe { nixl_capi_xfer_dlist_print(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Resizes the list to the given size
    pub fn resize(&mut self, _new_size: usize) -> Result<(), NixlError> { self.dirty.set(true); Ok(()) }

    /// Add a descriptor from a type implementing NixlDescriptor
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The descriptor remains valid for the lifetime of the list
    /// - The memory region pointed to by the descriptor remains valid
    pub fn add_storage_desc<D: NixlDescriptor + 'a>(
        &mut self,
        desc: &'a D,
    ) -> Result<(), NixlError> {
        // Validate memory type matches
        let desc_mem_type = desc.mem_type();
        let list_mem_type = if self.len().unwrap_or(0) > 0 { self.get_type().unwrap() } else { desc_mem_type };

        if desc_mem_type != list_mem_type && list_mem_type != MemType::Unknown {
            return Err(NixlError::InvalidParam);
        }

        // Get descriptor details
        let addr = unsafe { desc.as_ptr() } as usize;
        let len = desc.size();
        let dev_id = desc.device_id();

        // Add to list
        self.add_desc(addr, len, dev_id)
    }

    /// Ensure backend list matches our vector; lazily rebuild if needed
    fn ensure_synced(&self) -> Result<(), NixlError> {
        if !self.dirty.get() { return Ok(()); }
        // Clear backend
        let status = unsafe { nixl_capi_xfer_dlist_clear(self.inner.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => {}
            NIXL_CAPI_ERROR_INVALID_PARAM => return Err(NixlError::InvalidParam),
            _ => return Err(NixlError::BackendError),
        }
        // Re-add all descriptors
        for d in &self.descriptors {
            let status = unsafe {
                nixl_capi_xfer_dlist_add_desc(self.inner.as_ptr(), d.addr as uintptr_t, d.len, d.dev_id)
            };
            match status {
                NIXL_CAPI_SUCCESS => {}
                NIXL_CAPI_ERROR_INVALID_PARAM => return Err(NixlError::InvalidParam),
                _ => return Err(NixlError::BackendError),
            }
        }
        self.dirty.set(false);
        Ok(())
    }

    /// Find index of first descriptor matching the predicate
    pub fn get_index<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(&XferDescriptor) -> bool,
    {
        self.descriptors.iter().position(predicate)
    }

    /// Find index of a specific descriptor (exact match)
    pub fn index_of(&self, desc: &XferDescriptor) -> Option<usize> {
        self.descriptors.iter().position(|d| d == desc)
    }

    pub(crate) fn handle(&self) -> *mut bindings::nixl_capi_xfer_dlist_s {
        let _ = self.ensure_synced();
        self.inner.as_ptr()
    }
}

impl std::fmt::Debug for XferDescList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_type = self.get_type().unwrap_or(MemType::Unknown);
        let len = self.len().unwrap_or(0);
        let desc_count = self.desc_count().unwrap_or(0);

        f.debug_struct("XferDescList")
            .field("mem_type", &mem_type)
            .field("len", &len)
            .field("desc_count", &desc_count)
            .finish()
    }
}

impl PartialEq for XferDescList<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Compare memory types first
        if self.mem_type != other.mem_type {
            return false;
        }

        // Compare internal descriptor tracking
        // This gives us accurate comparison of actual descriptor contents
        self.descriptors == other.descriptors
    }
}

impl Drop for XferDescList<'_> {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_xfer_dlist(self.inner.as_ptr());
        }
    }
}

use std::ops::{Index, IndexMut};

impl Index<usize> for XferDescList<'_> {
    type Output = XferDescriptor;

    fn index(&self, index: usize) -> &Self::Output {
        &self.descriptors[index]
    }
}

impl IndexMut<usize> for XferDescList<'_> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // Mutating an element requires backend resync to stay consistent
        self.dirty.set(true);
        &mut self.descriptors[index]
    }
}
