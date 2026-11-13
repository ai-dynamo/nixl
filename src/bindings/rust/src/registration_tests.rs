// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn unregistered_handle(storage: &SystemStorage) -> Result<RegistrationHandle, NixlError> {
    let name = CString::new("deregistration_without_backend")?;
    let mut agent = ptr::null_mut();
    let status = unsafe { nixl_capi_create_agent(name.as_ptr(), &mut agent) };
    if status != NIXL_CAPI_SUCCESS {
        return Err(NixlError::BackendError);
    }
    let handle = NonNull::new(agent).ok_or(NixlError::BackendError)?;
    Ok(RegistrationHandle {
        agent: Some(Arc::new(RwLock::new(AgentInner {
            name: name.to_string_lossy().into_owned(),
            handle,
            backends: HashMap::new(),
            remotes: HashSet::new(),
        }))),
        ptr: storage.data.as_ptr() as usize,
        size: storage.data.len(),
        dev_id: 0,
        mem_type: MemType::Dram,
    })
}

#[test]
fn deregistration_failure_preserves_owner_for_retry() -> Result<(), NixlError> {
    let storage = SystemStorage::new(64)?;
    let mut handle = unregistered_handle(&storage)?;
    let owner = handle.agent.as_ref().unwrap().clone();
    for _ in 0..2 {
        assert!(matches!(
            handle.deregister(),
            Err(NixlError::NotFound | NixlError::BackendError)
        ));
        assert!(Arc::ptr_eq(handle.agent.as_ref().unwrap(), &owner));
        assert_eq!(Arc::strong_count(&owner), 2);
    }
    // No registration exists, so the test can release the synthetic handle.
    handle.agent = None;
    Ok(())
}

#[test]
fn poisoned_agent_lock_preserves_owner_for_retry() -> Result<(), NixlError> {
    let storage = SystemStorage::new(64)?;
    let mut handle = unregistered_handle(&storage)?;
    let owner = handle.agent.as_ref().unwrap().clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = owner.write().unwrap();
        panic!("poison the agent lock");
    }));
    assert!(result.is_err());
    assert!(matches!(
        handle.deregister(),
        Err(NixlError::AgentLockPoisoned)
    ));
    assert!(Arc::ptr_eq(handle.agent.as_ref().unwrap(), &owner));
    owner.clear_poison();
    assert!(matches!(
        handle.deregister(),
        Err(NixlError::NotFound | NixlError::BackendError)
    ));
    handle.agent = None;
    Ok(())
}

#[test]
fn handle_drop_retains_owner_after_failure() -> Result<(), NixlError> {
    let storage = SystemStorage::new(64)?;
    let handle = unregistered_handle(&storage)?;
    let owner = handle.agent.as_ref().unwrap().clone();
    drop(handle);
    assert_eq!(Arc::strong_count(&owner), 2);
    Ok(())
}

#[test]
fn storage_drop_retains_owner_after_failure() -> Result<(), NixlError> {
    let mut storage = SystemStorage::new(64)?;
    let handle = unregistered_handle(&storage)?;
    let owner = handle.agent.as_ref().unwrap().clone();
    storage.handle = Some(handle);
    drop(storage);
    assert_eq!(Arc::strong_count(&owner), 2);
    Ok(())
}

#[test]
fn successful_deregistration_releases_owner_and_is_idempotent() -> Result<(), NixlError> {
    let agent = Agent::new("successful_deregistration")?;
    let (_, params) = agent.get_plugin_params("UCX")?;
    let _backend = agent.create_backend("UCX", &params)?;
    let storage = SystemStorage::new(64)?;
    let mut handle = agent.register_memory(&storage, None)?;
    let owner = handle.agent.as_ref().unwrap().clone();
    let count = Arc::strong_count(&owner);
    handle.deregister()?;
    assert!(handle.agent.is_none());
    assert_eq!(Arc::strong_count(&owner), count - 1);
    handle.deregister()?;
    assert_eq!(Arc::strong_count(&owner), count - 1);
    Ok(())
}
