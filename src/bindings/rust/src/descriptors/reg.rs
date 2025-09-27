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

/// Public registration descriptor used for indexing and comparisons
#[derive(Debug, Clone, PartialEq)]
pub struct RegDescriptor {
    pub addr: usize,
    pub len: usize,
    pub dev_id: u64,
    pub metadata: Vec<u8>,
}

/// A safe wrapper around a NIXL registration descriptor list
pub struct RegDescList<'a> {
    inner: NonNull<bindings::nixl_capi_reg_dlist_s>,
    _phantom: PhantomData<&'a dyn NixlDescriptor>,
    // Track descriptors internally for comparison purposes
    descriptors: Vec<RegDescriptor>,
    mem_type: MemType,
    // Private: backend C list needs to be resynchronized from descriptors
    dirty: Cell<bool>,
}

impl<'a> RegDescList<'a> {
    /// Creates a new registration descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status = unsafe {
            nixl_capi_create_reg_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist)
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                if dlist.is_null() {
                    tracing::error!("Failed to create registration descriptor list");
                    return Err(NixlError::RegDescListCreationFailed);
                }
                let ptr = NonNull::new(dlist).ok_or(NixlError::RegDescListCreationFailed)?;

                Ok(Self {
                    inner: ptr,
                    _phantom: PhantomData,
                    descriptors: Vec::new(),
                    mem_type,
                    dirty: Cell::new(false),
                })
            }
            _ => Err(NixlError::RegDescListCreationFailed),
        }
    }

    pub fn get_type(&self) -> Result<MemType, NixlError> { Ok(self.mem_type) }

    /// Adds a descriptor to the list
    pub fn add_desc(&mut self, addr: usize, len: usize, dev_id: u64) -> Result<(), NixlError> {
        self.add_desc_with_meta(addr, len, dev_id, &[])
    }

    /// Add a descriptor with metadata
    pub fn add_desc_with_meta(
        &mut self,
        addr: usize,
        len: usize,
        dev_id: u64,
        metadata: &[u8],
    ) -> Result<(), NixlError> {
        // Push to vector; backend will be built lazily
        self.descriptors.push(RegDescriptor {
            addr,
            len,
            dev_id,
            metadata: metadata.to_vec(),
        });
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
    pub fn trim(&mut self) -> Result<(), NixlError> {
        self.descriptors.shrink_to_fit();
        // Backend capacity may differ; force rebuild on next use
        self.dirty.set(true);
        Ok(())
    }

    /// Removes the descriptor at the given index
    pub fn rem_desc(&mut self, index: i32) -> Result<(), NixlError> {
        if index < 0 { return Err(NixlError::InvalidParam); }
        let idx = index as usize;
        if idx >= self.descriptors.len() { return Err(NixlError::InvalidParam); }
        self.descriptors.remove(idx);
        self.dirty.set(true);
        Ok(())
    }

    /// Prints the list contents
    pub fn print(&self) -> Result<(), NixlError> {
        self.ensure_synced()?;
        let status = unsafe { nixl_capi_reg_dlist_print(self.inner.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Clears all descriptors from the list
    pub fn clear(&mut self) -> Result<(), NixlError> {
        self.descriptors.clear();
        self.dirty.set(true);
        Ok(())
    }

    /// Resizes the list to the given size
    pub fn resize(&mut self, _new_size: usize) -> Result<(), NixlError> {
        // No-op for now; backend capacity will be adjusted on next sync
        self.dirty.set(true);
        Ok(())
    }

    /// Add a descriptor from a type implementing NixlDescriptor
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The descriptor remains valid for the lifetime of the list
    /// - The memory region pointed to by the descriptor remains valid
    pub fn add_storage_desc(&mut self, desc: &'a dyn NixlDescriptor) -> Result<(), NixlError> {
        // Validate memory type matches
        let desc_mem_type = desc.mem_type();
        let list_mem_type = unsafe {
            // Get the memory type from the list by checking first descriptor
            let mut len = 0;
            match nixl_capi_reg_dlist_len(self.inner.as_ptr(), &mut len) {
                0 => Ok(()),
                -1 => Err(NixlError::InvalidParam),
                _ => Err(NixlError::BackendError),
            }?;
            if len > 0 {
                self.get_type()?
            } else {
                desc_mem_type
            }
        };

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
        if !self.dirty.get() {
            return Ok(());
        }
        // Clear backend
        let status = unsafe { nixl_capi_reg_dlist_clear(self.inner.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => {}
            NIXL_CAPI_ERROR_INVALID_PARAM => return Err(NixlError::InvalidParam),
            _ => return Err(NixlError::BackendError),
        }
        // Re-add all descriptors
        for d in &self.descriptors {
            let status = unsafe {
                nixl_capi_reg_dlist_add_desc(
                    self.inner.as_ptr(),
                    d.addr as uintptr_t,
                    d.len,
                    d.dev_id,
                    d.metadata.as_ptr() as *const std::ffi::c_void,
                    d.metadata.len(),
                )
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
        P: Fn(&RegDescriptor) -> bool,
    {
        self.descriptors.iter().position(predicate)
    }

    /// Find index of a specific descriptor (exact match)
    pub fn index_of(&self, desc: &RegDescriptor) -> Option<usize> {
        self.descriptors.iter().position(|d| d == desc)
    }

    /// Deserialize a descriptor list from a byte vector using custom binary format
    pub fn deserialize(data: &[u8]) -> Result<Self, NixlError> {
        if data.is_empty() {
            return Err(NixlError::InvalidParam);
        }

        // Use shared codec readers

        let mut offset = 0;

        // Read memory type
        let (mem_type, desc_count) = super::codec::read_header(data, &mut offset)?;

        // Create new descriptor list (but we can't use the C API constructor directly)
        // We need to create it and then populate the internal tracking
        let mut new_list = Self::new(mem_type)?;

        // Read each descriptor
        for _ in 0..desc_count {
            // Read addr, len, dev_id (each 8 bytes)
            let addr = super::codec::read_u64(data, &mut offset)? as usize;
            let len = super::codec::read_u64(data, &mut offset)? as usize;
            let dev_id = super::codec::read_u64(data, &mut offset)?;

            // Read metadata length
            let metadata_len = super::codec::read_u64(data, &mut offset)? as usize;

            // Read metadata
            let metadata = super::codec::read_bytes(data, &mut offset, metadata_len)?.to_vec();

            // Add descriptor to the list
            new_list.add_desc_with_meta(addr, len, dev_id, &metadata)?;
        }

        Ok(new_list)
    }

    /// Serialize the descriptor list to a byte vector using custom binary format
    pub fn serialize(&self) -> Result<Vec<u8>, NixlError> {
        // Precompute capacity: 1 (mem_type) + 8 (count) + per-desc (24 + 8 + meta.len())
        let mut total_size = 1usize + 8usize;
        for d in &self.descriptors {
            total_size = total_size
                .saturating_add(24) // addr,len,dev_id
                .saturating_add(8)  // metadata_len
                .saturating_add(d.metadata.len());
        }

        let mut buffer = Vec::new();
        super::codec::reserve_capacity(&mut buffer, total_size);

        // Format: [mem_type: u8][descriptor_count: u64][descriptors...]
        // Each descriptor: [addr: u64][len: u64][dev_id: u64][metadata_len: u64][metadata: bytes]

        super::codec::write_header(&mut buffer, self.mem_type, self.descriptors.len());

        // Write each descriptor
        for desc in &self.descriptors {
            // Write addr, len, dev_id
            super::codec::write_u64(&mut buffer, desc.addr as u64);
            super::codec::write_u64(&mut buffer, desc.len as u64);
            super::codec::write_u64(&mut buffer, desc.dev_id);

            // Write metadata length and data
            let metadata_len_u64 = desc.metadata.len() as u64;
            super::codec::write_u64(&mut buffer, metadata_len_u64);
            buffer.extend_from_slice(&desc.metadata);
        }

        Ok(buffer)
    }

    pub(crate) fn handle(&self) -> *mut bindings::nixl_capi_reg_dlist_s {
        // SAFETY: backend must be in sync before usage
        let _ = self.ensure_synced();
        self.inner.as_ptr()
    }
}

impl std::fmt::Debug for RegDescList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_type = self.get_type().unwrap_or(MemType::Unknown);
        let len = self.len().unwrap_or(0);
        let desc_count = self.desc_count().unwrap_or(0);

        f.debug_struct("RegDescList")
            .field("mem_type", &mem_type)
            .field("len", &len)
            .field("desc_count", &desc_count)
            .finish()
    }
}

impl PartialEq for RegDescList<'_> {
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

impl Drop for RegDescList<'_> {
    fn drop(&mut self) {
        tracing::trace!("Dropping registration descriptor list");
        unsafe {
            nixl_capi_destroy_reg_dlist(self.inner.as_ptr());
        }
        tracing::trace!("Registration descriptor list dropped");
    }
}

use std::ops::{Index, IndexMut};

impl Index<usize> for RegDescList<'_> {
    type Output = RegDescriptor;

    fn index(&self, index: usize) -> &Self::Output {
        &self.descriptors[index]
    }
}

impl IndexMut<usize> for RegDescList<'_> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // Mutating an element requires backend resync to stay consistent
        self.dirty.set(true);
        &mut self.descriptors[index]
    }
}
