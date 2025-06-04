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

/// A safe wrapper around NIXL parameters
pub struct Params {
    inner: NonNull<bindings::nixl_capi_params_s>,
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

impl Drop for ParamIterator<'_> {
    fn drop(&mut self) {
        // SAFETY: self.iter is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_params_destroy_iterator(self.iter.as_ptr());
        }
    }
}

impl Params {
    pub(crate) fn new(inner: NonNull<bindings::nixl_capi_params_s>) -> Self {
        Self { inner }
    }

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

    pub(crate) fn handle(&self) -> *mut bindings::nixl_capi_params_s {
        self.inner.as_ptr()
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        // SAFETY: self.inner is guaranteed to be valid by NonNull
        unsafe {
            nixl_capi_destroy_params(self.inner.as_ptr());
        }
    }
}
