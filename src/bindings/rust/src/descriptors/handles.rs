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

    pub(crate) fn as_ptr(&self) -> *mut bindings::nixl_capi_xfer_dlist_handle_s {
        self.inner.as_ptr()
    }
}

impl Drop for XferDescListHandle {
    fn drop(&mut self) {
        unsafe { nixl_capi_destroy_xfer_dlist_handle(self.inner.as_ptr()) };
    }
}
