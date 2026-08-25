// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


// Tests for the memory view API. Memory views are consumed device-side, so
// these allocate VRAM, mirroring test/gtest/device_api/single_write_test.cu.
// Compiled only where build.rs found a CUDA toolkit.
#![cfg(has_cuda)]

use nixl_sys::*;
use std::time::{Duration, Instant};
use std::thread;

mod common;
use common::*;

#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
}

fn has_cuda_gpu() -> bool {
    let mut count = 0;
    unsafe { cudaGetDeviceCount(&mut count) == 0 && count > 0 }
}

/// Selects the CUDA device before any agent is created, so UCX picks the NICs
/// closest to it. Otherwise the endpoint gets lanes on every NIC and none
/// matches the buffer's system device.
fn select_gpu() {
    assert_eq!(unsafe { cudaSetDevice(0) }, 0, "Failed to set CUDA device 0");
}

#[derive(Debug)]
struct VramStorage {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

unsafe impl Send for VramStorage {}
unsafe impl Sync for VramStorage {}

impl VramStorage {
    fn new(size: usize) -> Option<Self> {
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr, size) };
        if rc != 0 || ptr.is_null() {
            return None;
        }
        Some(Self { ptr, size })
    }
}

impl Drop for VramStorage {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr) };
    }
}

impl MemoryRegion for VramStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl NixlDescriptor for VramStorage {
    fn mem_type(&self) -> MemType {
        MemType::Vram
    }

    fn device_id(&self) -> u64 {
        0
    }
}

/// Allocates one VRAM buffer and registers it with the agent. `None` when the
/// allocation fails.
fn vram_buffer(agent: &Agent, opt_args: &OptArgs) -> Option<(VramStorage, RegistrationHandle)> {
    let storage = VramStorage::new(4096)?;

    let handle = agent
        .register_memory(&storage, Some(opt_args))
        .expect("Failed to register VRAM");
    Some((storage, handle))
}

fn vram_dlist(storage: &VramStorage) -> XferDescList<'_> {
    let mut dlist = XferDescList::new(MemType::Vram).expect("Failed to create XferDescList");
    unsafe { dlist.add_desc(storage.as_ptr() as usize, storage.size(), 0) };
    dlist
}

fn vram_remote_desc<'a>(
    storage: &VramStorage,
    remote_agent: Option<&'a str>,
) -> RemoteDescriptor<'a> {
    RemoteDescriptor {
        addr: unsafe { storage.as_ptr() } as usize,
        len: storage.size(),
        dev_id: 0,
        remote_agent,
    }
}

fn vram_remote_dlist(descriptor: &RemoteDescriptor<'_>) -> RemoteDescList {
    let mut dlist = RemoteDescList::new(MemType::Vram).expect("Failed to create RemoteDescList");
    dlist
        .add_desc(descriptor)
        .expect("Failed to add remote descriptor");
    dlist
}

#[test]
fn test_prep_mem_view_local() {
    if !has_cuda_gpu() {
        eprintln!("skipping test_prep_mem_view_local: no CUDA-capable GPU");
        return;
    }
    select_gpu();

    let (agent, opt_args) = create_agent_with_backend("mem_view_local").expect("Failed to create agent");
    let (storage, _handle) = vram_buffer(&agent, &opt_args).expect("Failed to allocate VRAM");

    // SAFETY: storage outlives the view
    let view = unsafe { agent.prep_mem_view_local(&vram_dlist(&storage), Some(&opt_args)) }
        .expect("prep_mem_view_local failed");
    assert!(!view.as_ptr().is_null());
}

#[test]
fn test_prep_mem_view_local_with_custom_param() {
    if !has_cuda_gpu() {
        eprintln!("skipping test_prep_mem_view_local_with_custom_param: no CUDA-capable GPU");
        return;
    }
    select_gpu();

    let (agent, mut opt_args) = create_agent_with_backend("mem_view_custom_param").expect("Failed to create agent");
    let (storage, _handle) = vram_buffer(&agent, &opt_args).expect("Failed to allocate VRAM");

    opt_args
        .set_custom_param(b"worker_id=0")
        .expect("Failed to set custom param");
    assert_eq!(
        opt_args.get_custom_param().expect("Failed to get custom param"),
        b"worker_id=0"
    );

    // SAFETY: storage outlives the view
    let view = unsafe { agent.prep_mem_view_local(&vram_dlist(&storage), Some(&opt_args)) }
        .expect("prep_mem_view_local failed");
    assert!(!view.as_ptr().is_null());
}

/// A remote memory view has no device lane until the endpoint is wired up.
/// Mirrors `DeviceApiTestBase::completeWireup`.
fn complete_wireup(from: &Agent, to: &Agent, to_name: &str) {
    from.send_notification(to_name, b"wireup", None)
        .expect("Failed to send wireup notification");

    let mut notifs = NotificationMap::new().expect("Failed to create notification map");
    let deadline = Instant::now() + Duration::from_secs(10);
    while notifs.is_empty().expect("Failed to query notification map") {
        assert!(Instant::now() < deadline, "Timed out waiting for wireup notification");
        to.get_notifications(&mut notifs, None)
            .expect("Failed to get notifications");
        thread::sleep(Duration::from_millis(50));
    }
}

/// Opt-in: needs a device-capable RDMA lane, which UCX offers only over
/// accelerated IB. Run with:
///     NIXL_TEST_DEVICE_LANE=1 cargo test --test mem_view -- --test-threads=1
#[test]
fn test_prep_mem_view_remote() {
    if std::env::var_os("NIXL_TEST_DEVICE_LANE").is_none() {
        eprintln!(
            "skipping test_prep_mem_view_remote: set NIXL_TEST_DEVICE_LANE=1 to run it, \
             on a host with accelerated IB"
        );
        return;
    }
    if !has_cuda_gpu() {
        eprintln!("skipping test_prep_mem_view_remote: no CUDA-capable GPU");
        return;
    }
    select_gpu();

    let (agent1, opt_args1) = create_agent_with_backend("mem_view_initiator").expect("Failed to create agent");
    let (agent2, opt_args2) = create_agent_with_backend("mem_view_target").expect("Failed to create agent");

    let (local, _local_handle) = vram_buffer(&agent1, &opt_args1).expect("Failed to allocate VRAM");
    let (remote, _remote_handle) = vram_buffer(&agent2, &opt_args2).expect("Failed to allocate VRAM");

    exchange_metadata(&agent1, &agent2).expect("Failed to exchange metadata");
    complete_wireup(&agent1, &agent2, "mem_view_target");

    // SAFETY: local and remote outlive the views
    let _local_view =
        unsafe { agent1.prep_mem_view_local(&vram_dlist(&local), Some(&opt_args1)) }
            .expect("prep_mem_view_local failed");

    let remote_desc = vram_remote_desc(&remote, Some("mem_view_target"));
    let view = unsafe {
        agent1.prep_mem_view_remote(&vram_remote_dlist(&remote_desc), Some(&opt_args1))
    }
    .expect("prep_mem_view_remote failed");
    assert!(!view.as_ptr().is_null());
}

#[test]
fn test_prep_mem_view_remote_unknown_agent() {
    if !has_cuda_gpu() {
        eprintln!("skipping test_prep_mem_view_remote_unknown_agent: no CUDA-capable GPU");
        return;
    }
    select_gpu();

    let (agent, opt_args) = create_agent_with_backend("mem_view_unknown").expect("Failed to create agent");
    let (storage, _handle) = vram_buffer(&agent, &opt_args).expect("Failed to allocate VRAM");

    // SAFETY: storage outlives the call
    let desc = vram_remote_desc(&storage, Some("no_such_agent"));
    let result =
        unsafe { agent.prep_mem_view_remote(&vram_remote_dlist(&desc), Some(&opt_args)) };
    assert!(
        matches!(result, Err(NixlError::NotFound)),
        "Expected NotFound for an unknown remote agent"
    );
}

/// A null-agent descriptor is a placeholder, not an addressed peer, so a list
/// holding only placeholders has no backend to prepare against. `nixl_ep` mixes
/// them with real agents to keep descriptor indices aligned with rank numbers.
#[test]
fn test_prep_mem_view_remote_null_agent_only() {
    if !has_cuda_gpu() {
        eprintln!("skipping test_prep_mem_view_remote_null_agent_only: no CUDA-capable GPU");
        return;
    }
    select_gpu();

    let (agent, opt_args) = create_agent_with_backend("mem_view_null_agent").expect("Failed to create agent");
    let (storage, _handle) = vram_buffer(&agent, &opt_args).expect("Failed to allocate VRAM");

    // SAFETY: storage outlives the call
    let desc = vram_remote_desc(&storage, None);
    let result =
        unsafe { agent.prep_mem_view_remote(&vram_remote_dlist(&desc), Some(&opt_args)) };
    assert!(
        matches!(result, Err(NixlError::NotFound)),
        "Expected NotFound for a list of only null agents"
    );
}
