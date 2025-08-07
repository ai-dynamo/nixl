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
