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

/// A safe wrapper around a NIXL transfer descriptor list
pub struct XferDescList<'a> {
    inner: NonNull<bindings::nixl_capi_xfer_dlist_s>,
    _phantom: PhantomData<&'a dyn NixlDescriptor>,
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
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    pub fn as_ptr(&self) -> *mut bindings::nixl_capi_xfer_dlist_s {
        self.inner.as_ptr()
    }

    /// Returns the memory type of the transfer descriptor list
    pub fn get_type(&self) -> Result<MemType, NixlError> {
        let mut mem_type = 0;
        let status = unsafe { nixl_capi_xfer_dlist_get_type(self.inner.as_ptr(), &mut mem_type) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(MemType::from(mem_type)),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Adds a descriptor to the list
    pub fn add_desc(&mut self, addr: usize, len: usize, dev_id: u64) -> Result<(), NixlError> {
        let status = unsafe {
            nixl_capi_xfer_dlist_add_desc(self.inner.as_ptr(), addr as uintptr_t, len, dev_id)
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Returns the number of descriptors in the list
    pub fn desc_count(&self) -> Result<usize, NixlError> {
        let mut count = 0;
        let status = unsafe { nixl_capi_xfer_dlist_desc_count(self.inner.as_ptr(), &mut count) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(count),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of descriptors in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut len = 0;
        let status = unsafe { nixl_capi_xfer_dlist_len(self.inner.as_ptr(), &mut len) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(len),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

     /// Trims the list to the given size
     pub fn trim(&mut self) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_xfer_dlist_trim(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Removes the descriptor at the given index
    pub fn rem_desc(&mut self, index: i32) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_xfer_dlist_rem_desc(self.inner.as_ptr(), index) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Clears all descriptors from the list
    pub fn clear(&mut self) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_xfer_dlist_clear(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Prints the list contents
    pub fn print(&self) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_xfer_dlist_print(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Resizes the list to the given size
    pub fn resize(&mut self, new_size: usize) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_xfer_dlist_resize(self.inner.as_ptr(), new_size) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

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
        let list_mem_type = unsafe {
            // Get the memory type from the list by checking first descriptor
            let mut len = 0;
            match nixl_capi_xfer_dlist_len(self.inner.as_ptr(), &mut len) {
                0 => Ok(()),
                -1 => Err(NixlError::InvalidParam),
                _ => Err(NixlError::BackendError),
            }?;
            if len > 0 {
                self.get_type().unwrap()
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

    pub(crate) fn handle(&self) -> *mut bindings::nixl_capi_xfer_dlist_s {
        self.inner.as_ptr()
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
