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

/// Reusable container for per-entry transfer events.
/// Create once, pass to `get_xfer_status_with_events` each poll to avoid allocation.
#[derive(Debug)]
pub struct XferEntryEvents {
    inner: NonNull<bindings::nixl_capi_xfer_entry_events_s>,
}

impl XferEntryEvents {
    /// Creates a new empty events container.
    pub fn new() -> Result<Self, NixlError> {
        let mut events = ptr::null_mut();
        let status = unsafe { nixl_capi_xfer_entry_events_create(&mut events) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(Self {
                inner: NonNull::new(events).ok_or(NixlError::BackendError)?,
            }),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the number of events.
    pub fn len(&self) -> Result<usize, NixlError> {
        let mut size = 0;
        let status = unsafe { nixl_capi_xfer_entry_events_size(self.inner.as_ptr(), &mut size) };
        match status {
            NIXL_CAPI_SUCCESS => Ok(size),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns the event at the given index as (descriptor_index, status).
    pub fn get(&self, index: usize) -> Result<(usize, i32), NixlError> {
        let mut idx_out = 0;
        let mut status_out = 0;
        let status = unsafe {
            nixl_capi_xfer_entry_events_get(
                self.inner.as_ptr(),
                index,
                &mut idx_out,
                &mut status_out,
            )
        };
        match status {
            NIXL_CAPI_SUCCESS => Ok((idx_out, status_out)),
            NIXL_CAPI_ERROR_INVALID_PARAM => Err(NixlError::InvalidParam),
            _ => Err(NixlError::BackendError),
        }
    }

    /// Returns an iterator over (index, status) events.
    pub fn iter(&self) -> XferEntryEventsIterator<'_> {
        XferEntryEventsIterator {
            events: self,
            index: 0,
            length: self.len().unwrap_or(0),
        }
    }
}

impl Default for XferEntryEvents {
    fn default() -> Self {
        Self::new().expect("Failed to create XferEntryEvents")
    }
}

impl Drop for XferEntryEvents {
    fn drop(&mut self) {
        unsafe {
            nixl_capi_xfer_entry_events_destroy(self.inner.as_ptr());
        }
    }
}

/// Iterator over per-entry transfer events.
pub struct XferEntryEventsIterator<'a> {
    events: &'a XferEntryEvents,
    index: usize,
    length: usize,
}

impl Iterator for XferEntryEventsIterator<'_> {
    type Item = Result<(usize, i32), NixlError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.length {
            None
        } else {
            let result = self.events.get(self.index);
            self.index += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.length - self.index;
        (remaining, Some(remaining))
    }
}

unsafe impl Send for XferEntryEvents {}
unsafe impl Sync for XferEntryEvents {}

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

impl XferRequest {
    pub(crate) fn new(
        inner: NonNull<bindings::nixl_capi_xfer_req_s>,
        agent: Arc<RwLock<AgentInner>>,
    ) -> Self {
        Self { inner, agent }
    }

    pub(crate) fn handle(&self) -> *mut bindings::nixl_capi_xfer_req_s {
        self.inner.as_ptr()
    }

    /// Gets telemetry data for this transfer request
    ///
    /// # Returns
    /// Transfer telemetry data containing timing and performance metrics
    ///
    /// # Errors
    /// * `NoTelemetry`  - If telemetry is not enabled or transfer is not complete
    /// * `InvalidParam` - If the request handle is invalid
    /// * `BackendError` - If there was an error retrieving telemetry data
    pub fn get_telemetry(&self) -> Result<XferTelemetry, NixlError> {
        tracing::trace!("Getting transfer telemetry from request");
        let mut telemetry = bindings::nixl_capi_xfer_telemetry_s {
            start_time_us: 0,
            post_duration_us: 0,
            xfer_duration_us: 0,
            total_bytes: 0,
            desc_count: 0,
        };

        let status = unsafe {
            nixl_capi_get_xfer_telemetry(
                self.agent.write().unwrap().handle.as_ptr(),
                self.handle(),
                &mut telemetry,
            )
        };

        match status {
            NIXL_CAPI_SUCCESS => {
                tracing::trace!("Successfully retrieved transfer telemetry from request");
                Ok(XferTelemetry {
                    start_time_us: telemetry.start_time_us,
                    post_duration_us: telemetry.post_duration_us,
                    xfer_duration_us: telemetry.xfer_duration_us,
                    total_bytes: telemetry.total_bytes,
                    desc_count: telemetry.desc_count,
                })
            },
            NIXL_CAPI_IN_PROG => {
                tracing::error!(error = "transfer_not_complete", "Transfer not complete");
                Err(NixlError::NoTelemetry)
            },
            NIXL_CAPI_ERROR_NO_TELEMETRY => {
                tracing::error!(error = "telemetry_not_enabled", "Telemetry not enabled");
                Err(NixlError::NoTelemetry)
            },
            _ => {
                tracing::error!(error = "backend_error", "Failed to get transfer telemetry from request");
                Err(NixlError::BackendError)
            }
        }
    }
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
