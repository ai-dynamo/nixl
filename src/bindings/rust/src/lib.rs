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

//! Raw FFI bindings to the NIXL library
//!
//! This crate provides low-level bindings to the NIXL C++ library.
//! It is not meant to be used directly, but rather through the higher-level
//! `nixl` crate.

use libc::uintptr_t;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::ptr::NonNull;
use std::sync::{Arc, RwLock};
use thiserror::Error;

// Include the generated bindings
mod bindings {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// Re-export types from the included bindings
use bindings::{
    nixl_capi_create_agent, nixl_capi_create_backend, nixl_capi_create_notif_map,
    nixl_capi_create_opt_args, nixl_capi_create_reg_dlist, nixl_capi_create_xfer_dlist,
    nixl_capi_deregister_mem, nixl_capi_destroy_agent, nixl_capi_destroy_backend,
    nixl_capi_destroy_mem_list, nixl_capi_destroy_notif_map, nixl_capi_destroy_opt_args,
    nixl_capi_destroy_params, nixl_capi_destroy_reg_dlist, nixl_capi_destroy_string_list,
    nixl_capi_destroy_xfer_dlist, nixl_capi_get_available_plugins, nixl_capi_get_backend_params,
    nixl_capi_get_local_md, nixl_capi_get_notifs, nixl_capi_get_plugin_params,
    nixl_capi_get_xfer_status, nixl_capi_invalidate_remote_md, nixl_capi_load_remote_md,
    nixl_capi_mem_list_get, nixl_capi_mem_list_is_empty, nixl_capi_mem_list_size,
    nixl_capi_mem_type_t, nixl_capi_mem_type_to_string, nixl_capi_notif_map_get_agent_at,
    nixl_capi_notif_map_get_notif, nixl_capi_notif_map_get_notifs_size, nixl_capi_notif_map_size,
    nixl_capi_opt_args_add_backend, nixl_capi_opt_args_get_has_notif,
    nixl_capi_opt_args_get_notif_msg, nixl_capi_opt_args_get_skip_desc_merge,
    nixl_capi_opt_args_set_has_notif, nixl_capi_opt_args_set_notif_msg,
    nixl_capi_opt_args_set_skip_desc_merge, nixl_capi_params_create_iterator,
    nixl_capi_params_destroy_iterator, nixl_capi_params_is_empty, nixl_capi_params_iterator_next,
    nixl_capi_post_xfer_req, nixl_capi_reg_dlist_add_desc, nixl_capi_reg_dlist_clear,
    nixl_capi_reg_dlist_has_overlaps, nixl_capi_reg_dlist_len, nixl_capi_reg_dlist_resize,
    nixl_capi_register_mem, nixl_capi_string_list_get, nixl_capi_string_list_size,
    nixl_capi_xfer_dlist_add_desc, nixl_capi_xfer_dlist_clear, nixl_capi_xfer_dlist_has_overlaps,
    nixl_capi_xfer_dlist_len, nixl_capi_xfer_dlist_resize, nixl_capi_xfer_dlist_get_type,
    nixl_capi_reg_dlist_get_type, nixl_capi_xfer_dlist_verify_sorted, nixl_capi_reg_dlist_verify_sorted,
    nixl_capi_xfer_dlist_is_empty, nixl_capi_reg_dlist_is_empty, nixl_capi_xfer_dlist_is_sorted,
    nixl_capi_reg_dlist_is_sorted, nixl_capi_xfer_dlist_desc_count, nixl_capi_reg_dlist_desc_count,
    nixl_capi_xfer_dlist_trim, nixl_capi_reg_dlist_trim, nixl_capi_xfer_dlist_rem_desc,
    nixl_capi_reg_dlist_rem_desc, nixl_capi_xfer_dlist_print, nixl_capi_reg_dlist_print,
    nixl_capi_agent_make_connection, nixl_capi_agent_prep_xfer_dlist,
    nixl_capi_agent_make_xfer_req, nixl_capi_create_xfer_dlist_handle,
    nixl_capi_destroy_xfer_dlist_handle, nixl_capi_estimate_xfer_cost, nixl_capi_gen_notif
};

// Re-export status codes
pub use bindings::{
    nixl_capi_status_t_NIXL_CAPI_ERROR_BACKEND as NIXL_CAPI_ERROR_BACKEND,
    nixl_capi_status_t_NIXL_CAPI_ERROR_INVALID_PARAM as NIXL_CAPI_ERROR_INVALID_PARAM,
    nixl_capi_status_t_NIXL_CAPI_IN_PROG as NIXL_CAPI_IN_PROG,
    nixl_capi_status_t_NIXL_CAPI_SUCCESS as NIXL_CAPI_SUCCESS,
};

/// Errors that can occur when using NIXL
#[derive(Error, Debug)]
pub enum NixlError {
    #[error("Invalid parameter provided to NIXL")]
    InvalidParam,
    #[error("Backend error occurred")]
    BackendError,
    #[error("Failed to create CString from input: {0}")]
    StringConversionError(#[from] std::ffi::NulError),
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Invalid data pointer")]
    InvalidDataPointer,
    #[error("Failed to create XferRequest")]
    FailedToCreateXferRequest,
    #[error("Failed to create registration descriptor list")]
    RegDescListCreationFailed,
    #[error("Failed to add registration descriptor")]
    RegDescAddFailed,
}

/// Memory types supported by NIXL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemType {
    Dram,
    Vram,
    Block,
    Object,
    File,
    Unknown,
}

impl From<nixl_capi_mem_type_t> for MemType {
    fn from(mem_type: nixl_capi_mem_type_t) -> Self {
        match mem_type {
            0 => MemType::Dram,
            1 => MemType::Vram,
            2 => MemType::Block,
            3 => MemType::Object,
            4 => MemType::File,
            _ => MemType::Unknown,
        }
    }
}

impl fmt::Display for MemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: We know the memory type is valid and the string will be available
        let mut str_ptr = ptr::null();
        unsafe {
            let mem_type = match self {
                MemType::Dram => 0,
                MemType::Vram => 1,
                MemType::Block => 2,
                MemType::Object => 3,
                MemType::File => 4,
                MemType::Unknown => 5,
            };
            nixl_capi_mem_type_to_string(mem_type, &mut str_ptr);
            let c_str = CStr::from_ptr(str_ptr);
            write!(f, "{}", c_str.to_str().unwrap())
        }
    }
}

/// A safe wrapper around a list of strings from NIXL
pub struct StringList {
    inner: NonNull<bindings::nixl_capi_string_list_s>,
}

impl StringList {
    /// Returns the number of strings in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;
        let status = unsafe { nixl_capi_string_list_size(self.inner.as_ptr(), &mut size) };

        match status {
            0 => Ok(size),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list contains no strings
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Returns the string at the given index
    pub fn get(&self, index: usize) -> Result<&str, NixlError> {
        let mut str_ptr = ptr::null();
        let status = unsafe { nixl_capi_string_list_get(self.inner.as_ptr(), index, &mut str_ptr) };

        match status {
            0 => {
                // SAFETY: If status is 0, str_ptr points to a valid null-terminated string
                let c_str = unsafe { CStr::from_ptr(str_ptr) };
                Ok(c_str.to_str().unwrap()) // Safe because NIXL strings are valid UTF-8
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the strings in the list
    pub fn iter(&self) -> StringListIterator<'_> {
        StringListIterator {
            list: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }
}

impl Drop for StringList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_string_list(self.inner.as_ptr());
        }
    }
}

/// An iterator over the strings in a StringList
pub struct StringListIterator<'a> {
    list: &'a StringList,
    index: usize,
    length: usize,
}

impl<'a> Iterator for StringListIterator<'a> {
    type Item = Result<&'a str, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let result = self.list.get(self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

/// A safe wrapper around NIXL memory list
pub struct MemList {
    inner: NonNull<bindings::nixl_capi_mem_list_s>,
}

impl Drop for MemList {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_mem_list(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around NIXL parameters
pub struct Params {
    inner: NonNull<bindings::nixl_capi_params_s>,
}

impl Drop for Params {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_params(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around a NIXL transfer descriptor list
pub struct XferDescList<'a> {
    inner: NonNull<bindings::nixl_capi_xfer_dlist_s>,
    _phantom: PhantomData<&'a dyn NixlDescriptor>,
}

impl<'a> XferDescList<'a> {
    /// Creates a new transfer descriptor list for the given memory type
    fn _new(mem_type: MemType, sorted: bool) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status =
            unsafe { nixl_capi_create_xfer_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist, sorted) };

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

    /// Creates a new transfer descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        Self::_new(mem_type, false)
    }

    /// Creates a new sorted transfer descriptor list for the given memory type
    pub fn new_sorted(mem_type: MemType) -> Result<Self, NixlError> {
        Self::_new(mem_type, true)
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

    /// Returns true if the list is sorted
    fn verify_sorted_inner(inner: NonNull<bindings::nixl_capi_xfer_dlist_s>) -> Result<bool, NixlError>   {
        let mut is_sorted = false;
        let status = unsafe { nixl_capi_xfer_dlist_verify_sorted(inner.as_ptr(), &mut is_sorted) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(is_sorted),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list is sorted
    pub fn verify_sorted(&self) -> Result<bool, NixlError> {
        Self::verify_sorted_inner(self.inner)
    }

    /// Returns the number of descriptors in the list
    /// Deprecated: Use desc_count instead
    #[deprecated(since = "0.1.0", note = "Use desc_count instead")]
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut len = 0;
        let status = unsafe { nixl_capi_xfer_dlist_len(self.inner.as_ptr(), &mut len) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(len),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
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

    /// Returns true if the list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        let mut is_empty: bool = false;
        let status = unsafe { nixl_capi_xfer_dlist_is_empty(self.inner.as_ptr(), &mut is_empty) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(is_empty),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list is sorted
    pub fn is_sorted(&self) -> Result<bool, NixlError> {
        let mut is_sorted = false;
        let status = unsafe { nixl_capi_xfer_dlist_is_sorted(self.inner.as_ptr(), &mut is_sorted) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(is_sorted),
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

    /// Returns true if any descriptors in the list overlap
    fn has_overlaps(&self) -> Result<bool, NixlError> {
        let mut has_overlaps = false;
        let status =
            unsafe { nixl_capi_xfer_dlist_has_overlaps(self.inner.as_ptr(), &mut has_overlaps) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(has_overlaps),
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
    fn resize(&mut self, new_size: usize) -> Result<(), NixlError> {
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
                // TODO: Add API to get descriptor memory type
                MemType::Unknown
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
}

impl<'a> Drop for XferDescList<'a> {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_xfer_dlist(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around a NIXL transfer descriptor list handle
pub struct XferDescListHandle {
    inner: NonNull<bindings::nixl_capi_xfer_dlist_handle_s>,
}

impl XferDescListHandle {
    pub fn new() -> Result<Self, NixlError> {
        let mut handle = ptr::null_mut();
        let status = unsafe { nixl_capi_create_xfer_dlist_handle(&mut handle) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(Self { inner: unsafe { NonNull::new_unchecked(handle) } }),
            _ => Err(NixlError::BackendError),
        }
    }
}

impl Drop for XferDescListHandle {
    fn drop(&mut self) {
        unsafe { nixl_capi_destroy_xfer_dlist_handle(self.inner.as_ptr()) };
    }
}


/// A safe wrapper around a NIXL registration descriptor list
pub struct RegDescList<'a> {
    inner: NonNull<bindings::nixl_capi_reg_dlist_s>,
    _phantom: PhantomData<&'a dyn NixlDescriptor>,
}

impl<'a> RegDescList<'a> {
    /// Creates a new registration descriptor list for the given memory type
    fn _new(mem_type: MemType, sorted: bool) -> Result<Self, NixlError> {
        let mut dlist = ptr::null_mut();
        let status =
            unsafe { nixl_capi_create_reg_dlist(mem_type as nixl_capi_mem_type_t, &mut dlist, sorted) };

        match status {
            NIXL_CAPI_SUCCESS => {
                if dlist.is_null() {
                    tracing::error!("Failed to create registration descriptor list");
                    return Err(NixlError::RegDescListCreationFailed);
                }

                // SAFETY: If status is NIXL_CAPI_SUCCESS, dlist is non-null
                let ptr = unsafe { NonNull::new_unchecked(dlist) };

                Ok(Self {
                    inner: ptr,
                    _phantom: PhantomData,
                })
            }
            _ => Err(NixlError::RegDescListCreationFailed),
        }
    }

    /// Creates a new registration descriptor list for the given memory type
    pub fn new(mem_type: MemType) -> Result<Self, NixlError> {
        Self::_new(mem_type, false)
    }

    /// Creates a new sorted registration descriptor list for the given memory type
    pub fn new_sorted(mem_type: MemType) -> Result<Self, NixlError> {
        Self::_new(mem_type, true)
    }

    pub fn get_type(&self) -> Result<MemType, NixlError> {
        let mut mem_type = 0;
        let status = unsafe { nixl_capi_reg_dlist_get_type(self.inner.as_ptr(), &mut mem_type) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(MemType::from(mem_type)),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Adds a descriptor to the list
    pub fn add_desc(&mut self, addr: usize, len: usize, dev_id: u64) -> Result<(), NixlError> {
        let status = unsafe {
            nixl_capi_reg_dlist_add_desc(self.inner.as_ptr(), addr as uintptr_t, len, dev_id)
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list is sorted
    fn verify_sorted_inner(inner: NonNull<bindings::nixl_capi_reg_dlist_s>) -> Result<bool, NixlError> {
        let mut is_sorted = false;
        let status = unsafe { nixl_capi_reg_dlist_verify_sorted(inner.as_ptr(), &mut is_sorted) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(is_sorted),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list is sorted
    pub fn verify_sorted(&self) -> Result<bool, NixlError> {
        Self::verify_sorted_inner(self.inner)
    }

    /// Returns the number of descriptors in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut len = 0;
        let status = unsafe { nixl_capi_reg_dlist_len(self.inner.as_ptr(), &mut len) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(len),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of descriptors in the list
    pub fn desc_count(&self) -> Result<usize, NixlError> {
        let mut count = 0;
        let status = unsafe { nixl_capi_reg_dlist_desc_count(self.inner.as_ptr(), &mut count) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(count),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if the list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        let mut is_empty = false;
        let status = unsafe { nixl_capi_reg_dlist_is_empty(self.inner.as_ptr(), &mut is_empty) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(is_empty),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    pub fn is_sorted(&self) -> Result<bool, NixlError> {
        let mut is_sorted = false;
        let status = unsafe { nixl_capi_reg_dlist_is_sorted(self.inner.as_ptr(), &mut is_sorted) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(is_sorted),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Trims the list to the given size
    pub fn trim(&mut self) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_reg_dlist_trim(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Removes the descriptor at the given index
    pub fn rem_desc(&mut self, index: i32) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_reg_dlist_rem_desc(self.inner.as_ptr(), index) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if any descriptors in the list overlap
    fn has_overlaps(&self) -> Result<bool, NixlError> {
        let mut has_overlaps = false;
        let status =
            unsafe { nixl_capi_reg_dlist_has_overlaps(self.inner.as_ptr(), &mut has_overlaps) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(has_overlaps),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Clears all descriptors from the list
    pub fn clear(&mut self) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_reg_dlist_clear(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Prints the list contents
    pub fn print(&self) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_reg_dlist_print(self.inner.as_ptr()) };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Resizes the list to the given size
    fn resize(&mut self, new_size: usize) -> Result<(), NixlError> {
        let status = unsafe { nixl_capi_reg_dlist_resize(self.inner.as_ptr(), new_size) };

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
                // TODO: Add API to get descriptor memory type
                MemType::Unknown
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
}

impl<'a> Drop for RegDescList<'a> {
    fn drop(&mut self) {
        tracing::trace!("Dropping registration descriptor list");
        unsafe {
            bindings::nixl_capi_destroy_reg_dlist(self.inner.as_ptr());
        }
        tracing::trace!("Registration descriptor list dropped");
    }
}

/// Inner state for an agent that manages the raw pointer
#[derive(Debug)]
struct AgentInner {
    name: String,
    handle: NonNull<bindings::nixl_capi_agent_s>,
    backends: HashMap<String, NonNull<bindings::nixl_capi_backend_s>>,
    remotes: HashSet<String>,
}

unsafe impl Send for AgentInner {}
unsafe impl Sync for AgentInner {}

impl AgentInner {
    fn new(handle: NonNull<bindings::nixl_capi_agent_s>, name: String) -> Self {
        Self {
            name,
            handle,
            backends: HashMap::new(),
            remotes: HashSet::new(),
        }
    }

    fn get_backend(&self, name: &str) -> Option<NonNull<bindings::nixl_capi_backend_s>> {
        self.backends.get(name).cloned()
    }

    fn invalidate_remote_md(&mut self, remote_agent: &str) -> Result<(), NixlError> {
        unsafe {
            if self.remotes.remove(remote_agent) {
                nixl_capi_invalidate_remote_md(self.handle.as_ptr(), remote_agent.as_ptr().cast());
            } else {
                return Err(NixlError::InvalidParam);
            }
        }
        Ok(())
    }

    fn invalidate_all_remotes(&mut self) -> Result<(), NixlError> {
        unsafe {
            for remote in self.remotes.drain() {
                nixl_capi_invalidate_remote_md(self.handle.as_ptr(), remote.as_ptr().cast());
            }
        }
        Ok(())
    }
}

impl Drop for AgentInner {
    fn drop(&mut self) {
        tracing::trace!("Dropping NIXL agent");
        unsafe {
            // invalidate all remotes
            for remote in self.remotes.iter() {
                tracing::trace!(remote.agent = %remote, "Invalidating remote agent");
                nixl_capi_invalidate_remote_md(self.handle.as_ptr(), remote.as_ptr().cast());
            }

            // destroy all backends
            for backend in self.backends.values() {
                tracing::trace!("Destroying backend");
                nixl_capi_destroy_backend(backend.as_ptr());
            }

            nixl_capi_destroy_agent(self.handle.as_ptr());
        }
        tracing::trace!("NIXL agent dropped");
    }
}

/// A NIXL agent that can create backends and manage memory
#[derive(Debug, Clone)]
pub struct Agent {
    inner: Arc<RwLock<AgentInner>>,
}

impl Agent {
    /// Creates a new agent with the given name
    pub fn new(name: &str) -> Result<Self, NixlError> {
        tracing::trace!(agent.name = %name, "Creating new NIXL agent");
        let c_name = CString::new(name)?;
        let mut agent = ptr::null_mut();
        let status = unsafe { nixl_capi_create_agent(c_name.as_ptr(), &mut agent) };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, agent is non-null
                let handle = unsafe { NonNull::new_unchecked(agent) };
                tracing::trace!(agent.name = %name, "Successfully created NIXL agent");
                Ok(Self {
                    inner: Arc::new(RwLock::new(AgentInner::new(handle, name.to_string()))),
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(agent.name = %name, error = "invalid_param", "Failed to create NIXL agent");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(agent.name = %name, error = "backend_error", "Failed to create NIXL agent");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Gets the name of the agent
    pub fn name(&self) -> String {
        self.inner.read().unwrap().name.clone()
    }

    /// Gets the list of available plugins
    pub fn get_available_plugins(&self) -> Result<StringList, NixlError> {
        tracing::trace!("Getting available NIXL plugins");
        let mut plugins = ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_get_available_plugins(
                self.inner.write().unwrap().handle.as_ptr(),
                &mut plugins,
            )
        };

        match status {
            0 => {
                // SAFETY: If status is 0, plugins was successfully created and is non-null
                let inner = unsafe { NonNull::new_unchecked(plugins) };
                tracing::trace!("Successfully retrieved NIXL plugins");
                Ok(StringList { inner })
            }
            -1 => {
                tracing::trace!(error = "invalid_param", "Failed to get NIXL plugins");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(error = "backend_error", "Failed to get NIXL plugins");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Gets the parameters for a plugin
    ///
    /// # Arguments
    /// * `plugin_name` - The name of the plugin
    ///
    /// # Returns
    /// The plugin's memory list and parameters
    ///
    /// # Errors
    /// Returns a NixlError if:
    /// * The plugin name contains interior nul bytes
    /// * The operation fails
    pub fn get_plugin_params(&self, plugin_name: &str) -> Result<(MemList, Params), NixlError> {
        let plugin_name = CString::new(plugin_name)?;
        let mut mems = ptr::null_mut();
        let mut params = ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_get_plugin_params(
                self.inner.read().unwrap().handle.as_ptr(),
                plugin_name.as_ptr(),
                &mut mems,
                &mut params,
            )
        };

        match status {
            0 => {
                // SAFETY: If status is 0, both pointers were successfully created and are non-null
                let mems_inner = unsafe { NonNull::new_unchecked(mems) };
                let params_inner = unsafe { NonNull::new_unchecked(params) };
                Ok((
                    MemList { inner: mems_inner },
                    Params {
                        inner: params_inner,
                    },
                ))
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Gets the parameters and memory types for a backend after initialization
    pub fn get_backend_params(&self, backend: &Backend) -> Result<(MemList, Params), NixlError> {
        let mut mem_list = ptr::null_mut();
        let mut params = ptr::null_mut();

        let status = unsafe {
            nixl_capi_get_backend_params(
                self.inner.read().unwrap().handle.as_ptr(),
                backend.inner.as_ptr(),
                &mut mem_list,
                &mut params,
            )
        };

        if status != NIXL_CAPI_SUCCESS {
            return Err(NixlError::BackendError);
        }

        // SAFETY: If status is NIXL_CAPI_SUCCESS, both pointers are non-null
        unsafe {
            Ok((
                MemList {
                    inner: NonNull::new_unchecked(mem_list),
                },
                Params {
                    inner: NonNull::new_unchecked(params),
                },
            ))
        }
    }

    /// Creates a new backend for the given plugin using the provided parameters
    pub fn create_backend(&self, plugin: &str, params: &Params) -> Result<Backend, NixlError> {
        tracing::trace!(plugin.name = %plugin, "Creating new NIXL backend");
        let c_plugin = CString::new(plugin).map_err(|_| NixlError::InvalidParam)?;
        let name = c_plugin.to_string_lossy().to_string();
        let mut backend = ptr::null_mut();
        let status = unsafe {
            nixl_capi_create_backend(
                self.inner.write().unwrap().handle.as_ptr(),
                c_plugin.as_ptr(),
                params.inner.as_ptr(),
                &mut backend,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                let backend_handle = NonNull::new(backend).ok_or(NixlError::BackendError)?;
                self.inner
                    .write()
                    .unwrap()
                    .backends
                    .insert(name.clone(), backend_handle.clone());
                tracing::trace!(plugin.name = %plugin, "Successfully created NIXL backend");
                Ok(Backend {
                    inner: backend_handle,
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(plugin.name = %plugin, error = "invalid_param", "Failed to create NIXL backend");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(plugin.name = %plugin, error = "backend_error", "Failed to create NIXL backend");
                Err(NixlError::BackendError)
            }
        }
    }

    pub fn get_backend(&self, name: &str) -> Option<Backend> {
        if let Some(backend) = self.inner.read().unwrap().get_backend(name) {
            Some(Backend { inner: backend })
        } else {
            None
        }
    }

    pub fn register_memory(
        &self,
        descriptor: &impl NixlDescriptor,
        opt_args: Option<&OptArgs>,
    ) -> Result<RegistrationHandle, NixlError> {
        let mut reg_dlist = RegDescList::new(descriptor.mem_type())?;
        unsafe {
            reg_dlist.add_storage_desc(descriptor)?;

            nixl_capi_register_mem(
                self.inner.write().unwrap().handle.as_ptr(),
                reg_dlist.inner.as_ptr(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            );
        }
        Ok(RegistrationHandle {
            agent: Some(self.inner.clone()),
            ptr: unsafe { descriptor.as_ptr() } as usize,
            size: descriptor.size(),
            dev_id: descriptor.device_id(),
            mem_type: descriptor.mem_type(),
        })
    }

    pub fn make_connection(&self, remote_agent: &str) -> Result<(), NixlError> {
        let remote_agent = CString::new(remote_agent)?;
        let status = unsafe {
            nixl_capi_agent_make_connection(
                self.inner.write().unwrap().handle.as_ptr(),
                remote_agent.as_ptr(),
                std::ptr::null_mut(),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    pub fn prep_xfer_dlist(&self, agent_name: &str, descs: &XferDescList, handle: &mut XferDescListHandle,
                           opt_args: &OptArgs) -> Result<(), NixlError> {
        let agent_name = CString::new(agent_name)?;
        let status = unsafe {
            nixl_capi_agent_prep_xfer_dlist(
                self.inner.write().unwrap().handle.as_ptr(),
                agent_name.as_ptr(),
                descs.inner.as_ptr(),
                handle.inner.as_ptr(),
                opt_args.inner.as_ptr(),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    pub fn make_xfer_req(&self, operation: XferOp, local_descs: &XferDescList, remote_descs: &XferDescList, remote_agent: &str, opt_args: &OptArgs) -> Result<XferRequest, NixlError> {
        let remote_agent = CString::new(remote_agent)?;
        let mut req = std::ptr::null_mut();

        let status = unsafe {
            nixl_capi_agent_make_xfer_req(
                self.inner.write().unwrap().handle.as_ptr(),
                operation as bindings::nixl_capi_xfer_op_t,
                local_descs.inner.as_ptr(),
                remote_descs.inner.as_ptr(),
                remote_agent.as_ptr(),
                &mut req,
                opt_args.inner.as_ptr(),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                let inner = NonNull::new(req).ok_or(NixlError::FailedToCreateXferRequest)?;
                Ok(XferRequest {
                    inner,
                    agent: self.inner.clone(),
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Creates a transfer request between local and remote descriptors
    ///
    /// # Arguments
    /// * `operation` - The transfer operation (read or write)
    /// * `local_descs` - The local descriptor list
    /// * `remote_descs` - The remote descriptor list
    /// * `remote_agent` - The name of the remote agent
    /// * `opt_args` - Optional arguments for the transfer
    ///
    /// # Returns
    /// A handle to the transfer request
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn create_xfer_req(
        &self,
        operation: XferOp,
        local_descs: &XferDescList,
        remote_descs: &XferDescList,
        remote_agent: &str,
        opt_args: Option<&OptArgs>,
    ) -> Result<XferRequest, NixlError> {
        let remote_agent = CString::new(remote_agent)?;
        let mut req = std::ptr::null_mut();

        // SAFETY: All pointers are guaranteed to be valid
        let status = unsafe {
            bindings::nixl_capi_create_xfer_req(
                self.inner.read().unwrap().handle.as_ptr(),
                operation as bindings::nixl_capi_xfer_op_t,
                local_descs.inner.as_ptr(),
                remote_descs.inner.as_ptr(),
                remote_agent.as_ptr(),
                &mut req,
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, req is guaranteed to be non-null
                let inner = NonNull::new(req).ok_or(NixlError::FailedToCreateXferRequest)?;
                Ok(XferRequest {
                    inner,
                    agent: self.inner.clone(),
                })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::FailedToCreateXferRequest),
        }
    }

    /// Estimates the cost of a transfer request
    ///
    /// # Arguments
    /// * `req` - Transfer request handle
    /// * `opt_args` - Optional arguments for the estimation
    ///
    /// # Returns
    /// A tuple containing (duration in microseconds, error margin in microseconds, cost method)
    ///
    /// # Errors
    /// Returns a NixlError if the operation fails
    pub fn estimate_xfer_cost(
        &self,
        req: &XferRequest,
        opt_args: Option<&OptArgs>,
    ) -> Result<(i64, i64, CostMethod), NixlError> {
        let mut duration_us: i64 = 0;
        let mut err_margin_us: i64 = 0;
        let mut method: u32 = 0;

        let status = unsafe {
            nixl_capi_estimate_xfer_cost(
                self.inner.write().unwrap().handle.as_ptr(),
                req.inner.as_ptr(),
                opt_args.map_or(ptr::null_mut(), |args| args.inner.as_ptr()),
                &mut duration_us,
                &mut err_margin_us,
                &mut method as *mut u32 as *mut bindings::nixl_capi_cost_t,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok((duration_us, err_margin_us, CostMethod::from(method))),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Posts a transfer request to initiate a transfer
    ///
    /// After this, the transfer state can be checked asynchronously until completion.
    /// For small transfers that complete within the call, the function returns `Ok(false)`.
    /// Otherwise, it returns `Ok(true)` to indicate the transfer is in progress.
    ///
    /// # Arguments
    /// * `req` - Transfer request handle obtained from `create_xfer_req`
    /// * `opt_args` - Optional arguments for the transfer request
    ///
    /// # Returns
    /// * `Ok(false)` - If the transfer completed immediately
    /// * `Ok(true)` - If the transfer is in progress
    /// * `Err` - If there was an error posting the transfer request
    pub fn post_xfer_req(
        &self,
        req: &XferRequest,
        opt_args: Option<&OptArgs>,
    ) -> Result<bool, NixlError> {
        tracing::trace!("Posting transfer request");
        let status = unsafe {
            nixl_capi_post_xfer_req(
                self.inner.write().unwrap().handle.as_ptr(),
                req.inner.as_ptr(),
                opt_args.map_or(ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!(
                    status = "completed",
                    "Transfer request completed immediately"
                );
                Ok(false)
            }
            NIXL_CAPI_IN_PROG => {
                tracing::trace!(status = "in_progress", "Transfer request in progress");
                Ok(true)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(error = "invalid_param", "Failed to post transfer request");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(error = "backend_error", "Failed to post transfer request");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Checks the status of a transfer request
    ///
    /// Returns `Ok(true)` if the transfer is still in progress, `Ok(false)` if it completed successfully.
    ///
    /// # Arguments
    /// * `req` - Transfer request handle after `post_xfer_req`
    pub fn get_xfer_status(&self, req: &XferRequest) -> Result<bool, NixlError> {
        let status = unsafe {
            nixl_capi_get_xfer_status(
                self.inner.write().unwrap().handle.as_ptr(),
                req.inner.as_ptr(),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => Ok(false), // Transfer completed
            NIXL_CAPI_IN_PROG => Ok(true),  // Transfer in progress
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Gets notifications from other agents
    ///
    /// # Arguments
    /// * `notifs` - Notification map to populate with notifications
    /// * `opt_args` - Optional arguments to filter notifications by backend
    pub fn get_notifications(
        &self,
        notifs: &mut NotificationMap,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), NixlError> {
        tracing::trace!("Getting notifications");
        let status = unsafe {
            nixl_capi_get_notifs(
                self.inner.write().unwrap().handle.as_ptr(),
                notifs.inner.as_ptr(),
                opt_args.map_or(ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully retrieved notifications");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(error = "invalid_param", "Failed to get notifications");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(error = "backend_error", "Failed to get notifications");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Send a notification to a remote agent
    ///
    /// # Arguments
    /// * `remote_agent` - Name of the remote agent to send notification to
    /// * `message` - The notification message to send
    /// * `backend` - Optional backend to use for sending the notification
    ///
    /// # Returns
    /// `Ok(())` if the notification was sent successfully
    pub fn send_notification(
        &self,
        remote_agent: &str,
        message: &[u8],
        backend: Option<&Backend>,
    ) -> Result<(), NixlError> {
        tracing::trace!(remote_agent = %remote_agent, "Sending notification");

        let c_remote_name = CString::new(remote_agent)?;

        let opt_args = if backend.is_some() {
            let mut args = OptArgs::new()?;
            if let Some(b) = backend {
                args.add_backend(b)?;
            }
            Some(args)
        } else {
            None
        };

        let status = unsafe {
            nixl_capi_gen_notif(
                self.inner.write().unwrap().handle.as_ptr(),
                c_remote_name.as_ptr(),
                message.as_ptr() as *const std::ffi::c_void,
                message.len(),
                opt_args
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!(remote_agent = %remote_agent, "Successfully sent notification");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(error = "invalid_param", remote_agent = %remote_agent, "Failed to send notification");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(error = "backend_error", remote_agent = %remote_agent, "Failed to send notification");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Gets the local metadata for this agent as a byte array
    pub fn get_local_md(&self) -> Result<Vec<u8>, NixlError> {
        tracing::trace!("Getting local metadata");
        let mut data = std::ptr::null_mut();
        let mut len = 0;

        let status = unsafe {
            nixl_capi_get_local_md(
                self.inner.write().unwrap().handle.as_ptr(),
                &mut data as *mut *mut _ as *mut *mut std::ffi::c_void,
                &mut len,
            )
        };

        let data = data as *const u8;

        if data.is_null() {
            tracing::trace!(
                error = "invalid_data_pointer",
                "Failed to get local metadata"
            );
            return Err(NixlError::InvalidDataPointer);
        }

        match status {
            NIXL_CAPI_SUCCESS => {
                let bytes = unsafe {
                    let slice = std::slice::from_raw_parts(data, len);
                    let vec = slice.to_vec();
                    libc::free(data as *mut libc::c_void);
                    vec
                };
                tracing::trace!(metadata.size = len, "Successfully retrieved local metadata");
                Ok(bytes)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(error = "invalid_param", "Failed to get local metadata");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(error = "backend_error", "Failed to get local metadata");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Loads remote metadata from a byte slice
    pub fn load_remote_md(&self, metadata: &[u8]) -> Result<String, NixlError> {
        tracing::trace!(metadata.size = metadata.len(), "Loading remote metadata");
        let mut agent_name = std::ptr::null_mut();

        let status = unsafe {
            nixl_capi_load_remote_md(
                self.inner.write().unwrap().handle.as_ptr(),
                metadata.as_ptr() as *const std::ffi::c_void,
                metadata.len(),
                &mut agent_name,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                let name = unsafe {
                    let c_str = std::ffi::CStr::from_ptr(agent_name);
                    let s = c_str.to_str().unwrap().to_string();
                    libc::free(agent_name as *mut libc::c_void);
                    s
                };
                self.inner.write().unwrap().remotes.insert(name.clone());
                tracing::trace!(remote.agent = %name, "Successfully loaded remote metadata");
                Ok(name)
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(error = "invalid_param", "Failed to load remote metadata");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(error = "backend_error", "Failed to load remote metadata");
                Err(NixlError::BackendError)
            }
        }
    }

    pub fn invalidate_remote_md(&self, remote_agent: &str) -> Result<(), NixlError> {
        self.inner
            .write()
            .unwrap()
            .invalidate_remote_md(remote_agent)
    }

    pub fn invalidate_all_remotes(&self) -> Result<(), NixlError> {
        self.inner.write().unwrap().invalidate_all_remotes()
    }

    /// Send this agent's metadata to etcd
    ///
    /// This enables other agents to discover this agent's metadata via etcd.
    ///
    /// # Arguments
    /// * `opt_args` - Optional arguments for sending metadata
    pub fn send_local_md(&self, opt_args: Option<&OptArgs>) -> Result<(), NixlError> {
        tracing::trace!("Sending local metadata to etcd");
        let status = unsafe {
            bindings::nixl_capi_send_local_md(
                self.inner.write().unwrap().handle.as_ptr(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully sent local metadata to etcd");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(
                    error = "invalid_param",
                    "Failed to send local metadata to etcd"
                );
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(
                    error = "backend_error",
                    "Failed to send local metadata to etcd"
                );
                Err(NixlError::BackendError)
            }
        }
    }

    /// Fetch a remote agent's metadata from etcd
    ///
    /// Once fetched, the metadata will be loaded and cached locally, enabling
    /// communication with the remote agent.
    ///
    /// # Arguments
    /// * `remote_name` - Name of the remote agent to fetch metadata for
    /// * `opt_args` - Optional arguments for fetching metadata
    pub fn fetch_remote_md(
        &self,
        remote_name: &str,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), NixlError> {
        tracing::trace!(remote_agent = %remote_name, "Fetching remote metadata from etcd");

        let c_remote_name = CString::new(remote_name)?;
        let status = unsafe {
            bindings::nixl_capi_fetch_remote_md(
                self.inner.write().unwrap().handle.as_ptr(),
                c_remote_name.as_ptr(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                self.inner
                    .write()
                    .unwrap()
                    .remotes
                    .insert(remote_name.to_string());
                tracing::trace!(remote_agent = %remote_name, "Successfully fetched remote metadata from etcd");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(error = "invalid_param", remote_agent = %remote_name, "Failed to fetch remote metadata from etcd");
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(error = "backend_error", remote_agent = %remote_name, "Failed to fetch remote metadata from etcd");
                Err(NixlError::BackendError)
            }
        }
    }

    /// Invalidate this agent's metadata in etcd
    ///
    /// This signals to other agents that this agent's metadata is no longer valid.
    ///
    /// # Arguments
    /// * `opt_args` - Optional arguments for invalidating metadata
    pub fn invalidate_local_md(&self, opt_args: Option<&OptArgs>) -> Result<(), NixlError> {
        tracing::trace!("Invalidating local metadata in etcd");
        let status = unsafe {
            bindings::nixl_capi_invalidate_local_md(
                self.inner.write().unwrap().handle.as_ptr(),
                opt_args.map_or(std::ptr::null_mut(), |args| args.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully invalidated local metadata in etcd");
                Ok(())
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => {
                tracing::trace!(
                    error = "invalid_param",
                    "Failed to invalidate local metadata in etcd"
                );
                Err(NixlError::InvalidParam)
            }
            _ => {
                tracing::trace!(
                    error = "backend_error",
                    "Failed to invalidate local metadata in etcd"
                );
                Err(NixlError::BackendError)
            }
        }
    }

    /// Check if remote metadata for a specific agent is available
    ///
    /// This function checks if the metadata for the specified remote agent has been
    /// loaded and if specific descriptors can be found in the metadata.
    ///
    /// # Arguments
    /// * `remote_agent` - Name of the remote agent to check
    /// * `descs` - Optional descriptor list to check against the remote metadata.
    ///            If None, only checks if any metadata exists for the agent.
    ///
    /// # Returns
    /// `true` if the remote agent's metadata is available (and descriptors are found if provided),
    /// `false` otherwise
    pub fn check_remote_metadata(&self, remote_agent: &str, descs: Option<&XferDescList>) -> bool {
        tracing::trace!(remote_agent = %remote_agent, "Checking remote metadata");

        let c_remote_name = match CString::new(remote_agent) {
            Ok(name) => name,
            Err(_) => {
                tracing::trace!(
                    error = "invalid_param",
                    remote_agent = %remote_agent,
                    "Invalid remote agent name"
                );
                return false;
            }
        };

        let status = unsafe {
            bindings::nixl_capi_check_remote_md(
                self.inner.read().unwrap().handle.as_ptr(),
                c_remote_name.as_ptr(),
                descs.map_or(std::ptr::null_mut(), |d| d.inner.as_ptr()),
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!(remote_agent = %remote_agent, "Remote metadata is available");
                true
            }
            _ => {
                tracing::trace!(remote_agent = %remote_agent, "Remote metadata is not available");
                false
            }
        }
    }
}

impl Drop for NotificationMap {
    fn drop(&mut self) {
        tracing::trace!("Dropping notification map");
        unsafe {
            nixl_capi_destroy_notif_map(self.inner.as_ptr());
        }
        tracing::trace!("Notification map dropped");
    }
}

#[derive(Debug)]
pub struct RegistrationHandle {
    agent: Option<Arc<RwLock<AgentInner>>>,
    ptr: usize,
    size: usize,
    dev_id: u64,
    mem_type: MemType,
}

impl RegistrationHandle {
    pub fn agent_name(&self) -> Option<String> {
        self.agent
            .as_ref()
            .map(|agent| agent.read().unwrap().name.clone())
    }

    pub fn deregister(&mut self) -> Result<(), NixlError> {
        if let Some(agent) = self.agent.take() {
            tracing::trace!(
                ptr = self.ptr,
                size = self.size,
                dev_id = self.dev_id,
                mem_type = ?self.mem_type,
                "Deregistering memory"
            );
            let mut reg_dlist = RegDescList::new(self.mem_type)?;
            unsafe {
                reg_dlist.add_desc(self.ptr, self.size, self.dev_id)?;
                let _opt_args = OptArgs::new().unwrap();
                nixl_capi_deregister_mem(
                    agent.write().unwrap().handle.as_ptr(),
                    reg_dlist.inner.as_ptr(),
                    _opt_args.inner.as_ptr(),
                );
            }
            tracing::trace!("Memory deregistered successfully");
        }
        Ok(())
    }
}

impl Drop for RegistrationHandle {
    fn drop(&mut self) {
        tracing::trace!(
            ptr = self.ptr,
            size = self.size,
            dev_id = self.dev_id,
            mem_type = ?self.mem_type,
            "Dropping registration handle"
        );
        if let Err(e) = self.deregister() {
            tracing::debug!(error = ?e, "Failed to deregister memory");
        }
    }
}

/// A NIXL backend that can be used for data transfer
#[derive(Debug)]
pub struct Backend {
    inner: NonNull<bindings::nixl_capi_backend_s>,
}

unsafe impl Send for Backend {}
unsafe impl Sync for Backend {}

/// A safe wrapper around NIXL optional arguments
pub struct OptArgs {
    inner: NonNull<bindings::nixl_capi_opt_args_s>,
}

impl OptArgs {
    /// Creates a new empty optional arguments struct
    pub fn new() -> Result<Self, NixlError> {
        let mut args = ptr::null_mut();

        let status = unsafe { nixl_capi_create_opt_args(&mut args) };

        match status {
            0 => {
                // SAFETY: If status is 0, args was successfully created and is non-null
                let inner = unsafe { NonNull::new_unchecked(args) };
                Ok(Self { inner })
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Add a backend to the optional arguments
    pub fn add_backend(&mut self, backend: &Backend) -> Result<(), NixlError> {
        let status =
            unsafe { nixl_capi_opt_args_add_backend(self.inner.as_ptr(), backend.inner.as_ptr()) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Set the notification message
    pub fn set_notification_message(&mut self, message: &[u8]) -> Result<(), NixlError> {
        let status = unsafe {
            nixl_capi_opt_args_set_notif_msg(
                self.inner.as_ptr(),
                message.as_ptr() as *const _,
                message.len(),
            )
        };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Get the notification message
    pub fn get_notification_message(&self) -> Result<Vec<u8>, NixlError> {
        let mut data = ptr::null_mut();
        let mut len = 0;
        let status =
            unsafe { nixl_capi_opt_args_get_notif_msg(self.inner.as_ptr(), &mut data, &mut len) };

        match status {
            NIXL_CAPI_SUCCESS => {
                if data.is_null() {
                    Ok(Vec::new())
                } else {
                    // SAFETY: If status is 0 and data is not null, it points to valid memory of size len
                    let message = unsafe {
                        let slice = std::slice::from_raw_parts(data as *const u8, len);
                        let vec = slice.to_vec();
                        libc::free(data as *mut _);
                        vec
                    };
                    Ok(message)
                }
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Set whether notification is enabled
    pub fn set_has_notification(&mut self, has_notification: bool) -> Result<(), NixlError> {
        let status =
            unsafe { nixl_capi_opt_args_set_has_notif(self.inner.as_ptr(), has_notification) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Get whether notification is enabled
    pub fn has_notification(&self) -> Result<bool, NixlError> {
        let mut has_notification = false;
        let status =
            unsafe { nixl_capi_opt_args_get_has_notif(self.inner.as_ptr(), &mut has_notification) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(has_notification),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Set whether to skip descriptor merging
    pub fn set_skip_descriptor_merge(&mut self, skip_merge: bool) -> Result<(), NixlError> {
        let status =
            unsafe { nixl_capi_opt_args_set_skip_desc_merge(self.inner.as_ptr(), skip_merge) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(()),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Get whether descriptor merging is skipped
    pub fn skip_descriptor_merge(&self) -> Result<bool, NixlError> {
        let mut skip_merge = false;
        let status =
            unsafe { nixl_capi_opt_args_get_skip_desc_merge(self.inner.as_ptr(), &mut skip_merge) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(skip_merge),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

impl Drop for OptArgs {
    fn drop(&mut self) {
        tracing::trace!("Dropping optional arguments");
        unsafe {
            nixl_capi_destroy_opt_args(self.inner.as_ptr());
        }
        tracing::trace!("Optional arguments dropped");
    }
}

/// A key-value pair in the parameters
#[derive(Debug)]
pub struct ParamPair<'a> {
    pub key: &'a str,
    pub value: &'a str,
}

/// An iterator over parameter key-value pairs
pub struct ParamIterator<'a> {
    iter: NonNull<bindings::nixl_capi_param_iter_s>,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Iterator for ParamIterator<'a> {
    type Item = Result<ParamPair<'a>, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut key_ptr = ptr::null();
        let mut value_ptr = ptr::null();
        let mut has_next = false;

        // SAFETY: self.iter is guaranteed to be valid by NonNull
        let status = unsafe {
            nixl_capi_params_iterator_next(
                self.iter.as_ptr(),
                &mut key_ptr,
                &mut value_ptr,
                &mut has_next,
            )
        };

        match status {
            0 if !has_next => None,
            0 => {
                // SAFETY: If status is 0, both pointers are valid null-terminated strings
                let result = unsafe {
                    let key = CStr::from_ptr(key_ptr).to_str().unwrap();
                    let value = CStr::from_ptr(value_ptr).to_str().unwrap();
                    Ok(ParamPair { key, value })
                };
                Some(result)
            }
            -1 => Some(Err(NixlError::InvalidParam)),
            _ => Some(Err(NixlError::BackendError)),
        }
    }
}

impl<'a> Drop for ParamIterator<'a> {
    fn drop(&mut self) {
        // SAFETY: self.iter is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_params_destroy_iterator(self.iter.as_ptr());
        }
    }
}

impl Params {
    /// Returns true if the parameters are empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        let mut is_empty = false;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_params_is_empty(self.inner.as_ptr(), &mut is_empty) };

        match status {
            0 => Ok(is_empty),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the parameter key-value pairs
    pub fn iter(&self) -> Result<ParamIterator<'_>, NixlError> {
        let mut iter = ptr::null_mut();

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_params_create_iterator(self.inner.as_ptr(), &mut iter) };

        match status {
            0 => {
                // SAFETY: If status is 0, iter was successfully created and is non-null
                let iter = unsafe { NonNull::new_unchecked(iter) };
                Ok(ParamIterator {
                    iter,
                    _phantom: std::marker::PhantomData,
                })
            }
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

impl MemList {
    /// Returns true if the memory list is empty
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        let mut is_empty = false;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_is_empty(self.inner.as_ptr(), &mut is_empty) };

        match status {
            0 => Ok(is_empty),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of memory types in the list
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_size(self.inner.as_ptr(), &mut size) };

        match status {
            0 => Ok(size),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the memory type at the given index
    pub fn get(&self, index: usize) -> Result<MemType, NixlError> {
        let mut mem_type = 0;

        // SAFETY: self.inner is guaranteed to be valid by NonNull
        let status = unsafe { nixl_capi_mem_list_get(self.inner.as_ptr(), index, &mut mem_type) };

        match status {
            0 => Ok(MemType::from(mem_type)),
            -1 => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the memory types
    pub fn iter(&self) -> MemListIterator<'_> {
        MemListIterator {
            list: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }
}

/// An iterator over memory types in a MemList
pub struct MemListIterator<'a> {
    list: &'a MemList,
    index: usize,
    length: usize,
}

impl<'a> Iterator for MemListIterator<'a> {
    type Item = Result<MemType, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let result = self.list.get(self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

/// A trait for storage types that can be used with NIXL
pub trait MemoryRegion: std::fmt::Debug + Send + Sync {
    /// Get a raw pointer to the storage
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The pointer is not used after the storage is dropped
    /// - Access patterns respect the storage's thread safety model
    unsafe fn as_ptr(&self) -> *const u8;

    /// Returns the total size of the storage in bytes
    fn size(&self) -> usize;
}

/// A trait for types that can be added to NIXL descriptor lists
pub trait NixlDescriptor: MemoryRegion {
    /// Get the memory type for this descriptor
    fn mem_type(&self) -> MemType;

    /// Get the device ID for this memory region
    fn device_id(&self) -> u64;
}

/// A trait for types that can be registered with NIXL
pub trait NixlRegistration: NixlDescriptor {
    fn register(&mut self, agent: &Agent, opt_args: Option<&OptArgs>) -> Result<(), NixlError>;
}

/// System memory storage implementation using a Vec<u8>
#[derive(Debug)]
pub struct SystemStorage {
    data: Vec<u8>,
    handle: Option<RegistrationHandle>,
}

impl SystemStorage {
    /// Create a new system storage with the given size
    pub fn new(size: usize) -> Result<Self, NixlError> {
        let mut data = Vec::with_capacity(size);
        // Initialize to zero to ensure consistent behavior
        data.resize(size, 0);
        Ok(Self { data, handle: None })
    }

    /// Fill the storage with a specific byte value
    pub fn memset(&mut self, value: u8) {
        self.data.fill(value);
    }

    /// Get a slice of the underlying data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

impl MemoryRegion for SystemStorage {
    fn size(&self) -> usize {
        self.data.len()
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

impl NixlDescriptor for SystemStorage {
    fn mem_type(&self) -> MemType {
        MemType::Dram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

impl NixlRegistration for SystemStorage {
    fn register(&mut self, agent: &Agent, opt_args: Option<&OptArgs>) -> Result<(), NixlError> {
        let handle = agent.register_memory(self, opt_args)?;
        self.handle = Some(handle);
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum XferOp {
    Read = 0,
    Write = 1,
}

/// Methods used for estimating transfer costs
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CostMethod {
    AnalyticalBackend = 0,
    Unknown = 1,
}

impl From<u32> for CostMethod {
    fn from(value: u32) -> Self {
        match value {
            0 => CostMethod::AnalyticalBackend,
            _ => CostMethod::Unknown,
        }
    }
}

/// A handle to a transfer request
pub struct XferRequest {
    inner: NonNull<bindings::nixl_capi_xfer_req_s>,
    agent: Arc<RwLock<AgentInner>>,
}

// SAFETY: XferRequest can be sent between threads safely
unsafe impl Send for XferRequest {}
// SAFETY: XferRequest can be shared between threads safely
unsafe impl Sync for XferRequest {}

impl Drop for XferRequest {
    fn drop(&mut self) {
        unsafe {
            bindings::nixl_capi_release_xfer_req(
                self.agent.write().unwrap().handle.as_ptr(),
                self.inner.as_ptr(),
            );

            bindings::nixl_capi_destroy_xfer_req(self.inner.as_ptr());
        }
    }
}

/// A safe wrapper around NIXL notification map
pub struct NotificationMap {
    inner: NonNull<bindings::nixl_capi_notif_map_s>,
}

impl NotificationMap {
    /// Creates a new empty notification map
    pub fn new() -> Result<Self, NixlError> {
        let mut map = ptr::null_mut();
        let status = unsafe { nixl_capi_create_notif_map(&mut map) };
        match status {
            NIXL_CAPI_SUCCESS => {
                // SAFETY: If status is NIXL_CAPI_SUCCESS, map is non-null
                let inner = unsafe { NonNull::new_unchecked(map) };
                Ok(Self { inner })
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of agents that have notifications
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;
        let status = unsafe { nixl_capi_notif_map_size(self.inner.as_ptr(), &mut size) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(size),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns true if there are no notifications
    pub fn is_empty(&self) -> Result<bool, NixlError> {
        Ok(self.len()? == 0)
    }

    /// Returns an iterator over the agent names that have notifications
    pub fn agents(&self) -> NotificationMapAgentIterator<'_> {
        NotificationMapAgentIterator {
            map: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }

    /// Returns the number of notifications for a given agent
    pub fn get_notifications_size(&self, agent_name: &str) -> Result<usize, NixlError> {
        let mut size = 0;
        let c_name = CString::new(agent_name)?;
        let status = unsafe {
            nixl_capi_notif_map_get_notifs_size(self.inner.as_ptr(), c_name.as_ptr(), &mut size)
        };
        match status {
            NIXL_CAPI_SUCCESS => Ok(size),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over the notifications for a given agent
    pub fn get_notifications(
        &self,
        agent_name: &str,
    ) -> Result<NotificationIterator<'_>, NixlError> {
        let size = self.get_notifications_size(agent_name)?;
        Ok(NotificationIterator {
            map: self,
            agent_name: agent_name.to_string(),
            index: 0,
            length: size,
        })
    }

    /// Returns a specific notification for a given agent
    pub fn get_notification(&self, agent_name: &str, index: usize) -> Result<Vec<u8>, NixlError> {
        let c_name = CString::new(agent_name)?;
        let mut data: *const u8 = ptr::null();
        let mut len = 0;
        let status = unsafe {
            nixl_capi_notif_map_get_notif(
                self.inner.as_ptr(),
                c_name.as_ptr(),
                index,
                &mut data as *mut *const _ as *mut *const std::ffi::c_void,
                &mut len,
            )
        };
        match status {
            NIXL_CAPI_SUCCESS => {
                if data.is_null() {
                    Ok(Vec::new())
                } else {
                    // SAFETY: If status is NIXL_CAPI_SUCCESS, data points to valid memory of size len
                    let bytes = unsafe {
                        let slice = std::slice::from_raw_parts(data as *const u8, len);
                        slice.to_vec()
                    };
                    Ok(bytes)
                }
            }
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }
}

/// An iterator over agent names in a NotificationMap
pub struct NotificationMapAgentIterator<'a> {
    map: &'a NotificationMap,
    index: usize,
    length: usize,
}

impl<'a> Iterator for NotificationMapAgentIterator<'a> {
    type Item = Result<&'a str, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let mut agent_name = ptr::null();
            let status = unsafe {
                nixl_capi_notif_map_get_agent_at(
                    self.map.inner.as_ptr(),
                    self.index,
                    &mut agent_name,
                )
            };
            self.index += 1;
            match status {
                NIXL_CAPI_SUCCESS => {
                    // SAFETY: If status is NIXL_CAPI_SUCCESS, agent_name points to a valid C string
                    let name = unsafe { CStr::from_ptr(agent_name) };
                    Some(name.to_str().map_err(|_| NixlError::InvalidParam))
                }
                NIXL_CAPI_ERROR_INVALID_PARAM => Some(Err(NixlError::InvalidParam)),
                _ => Some(Err(NixlError::BackendError)),
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

/// An iterator over notifications for a specific agent
pub struct NotificationIterator<'a> {
    map: &'a NotificationMap,
    agent_name: String,
    index: usize,
    length: usize,
}

impl<'a> Iterator for NotificationIterator<'a> {
    type Item = Result<Vec<u8>, NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let result = self.map.get_notification(&self.agent_name, self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests;
