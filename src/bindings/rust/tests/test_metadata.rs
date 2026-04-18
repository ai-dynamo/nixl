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

use nixl_sys::{MemType, MemoryRegion, NixlDescriptor, NixlObjectMetadata};

// --- Minimal test fixtures ---

/// A bare descriptor that relies on the default metadata() impl.
#[derive(Debug)]
struct BareDescriptor {
    data: Vec<u8>,
}

impl BareDescriptor {
    fn new(size: usize) -> Self {
        Self { data: vec![0u8; size] }
    }
}

impl MemoryRegion for BareDescriptor {
    unsafe fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
    fn size(&self) -> usize { self.data.len() }
}

impl NixlDescriptor for BareDescriptor {
    fn mem_type(&self) -> MemType { MemType::Dram }
    fn device_id(&self) -> u64 { 0 }
    // metadata() intentionally not overridden — tests the default
}

/// A descriptor that carries opaque metadata bytes (e.g. an object storage key).
#[derive(Debug)]
struct DescriptorWithMeta {
    data: Vec<u8>,
    meta: Vec<u8>,
}

impl DescriptorWithMeta {
    fn new(size: usize, meta: &[u8]) -> Self {
        Self { data: vec![0u8; size], meta: meta.to_vec() }
    }
}

impl MemoryRegion for DescriptorWithMeta {
    unsafe fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
    fn size(&self) -> usize { self.data.len() }
}

impl NixlDescriptor for DescriptorWithMeta {
    fn mem_type(&self) -> MemType { MemType::Dram }
    fn device_id(&self) -> u64 { 0 }
    fn metadata(&self) -> Option<Vec<u8>> { Some(self.meta.clone()) }
}

/// A full NixlObject: carries both descriptor info and metadata.
/// Models e.g. an ImmutableBlock that owns a NixlMetadata.
#[derive(Debug)]
struct NixlObject {
    data: Vec<u8>,
    nixl_meta: Option<Vec<u8>>,
}

impl NixlObject {
    fn new(size: usize, meta: Option<&[u8]>) -> Self {
        Self { data: vec![0u8; size], nixl_meta: meta.map(|m| m.to_vec()) }
    }
}

impl MemoryRegion for NixlObject {
    unsafe fn as_ptr(&self) -> *const u8 { self.data.as_ptr() }
    fn size(&self) -> usize { self.data.len() }
}

impl NixlDescriptor for NixlObject {
    fn mem_type(&self) -> MemType { MemType::Dram }
    fn device_id(&self) -> u64 { 0 }
}

impl NixlObjectMetadata for NixlObject {
    fn nixl_metadata(&self) -> Option<Vec<u8>> { self.nixl_meta.clone() }
}

// --- NixlDescriptor::metadata() tests ---

#[test]
fn descriptor_default_metadata_is_none() {
    let desc = BareDescriptor::new(64);
    assert_eq!(desc.metadata(), None);
}

#[test]
fn descriptor_custom_metadata_returns_bytes() {
    let key = b"s3://bucket/object-key";
    let desc = DescriptorWithMeta::new(64, key);
    assert_eq!(desc.metadata(), Some(key.to_vec()));
}

#[test]
fn descriptor_custom_metadata_empty_bytes() {
    let desc = DescriptorWithMeta::new(64, b"");
    assert_eq!(desc.metadata(), Some(vec![]));
}

// --- NixlObjectMetadata tests ---

#[test]
fn nixl_object_metadata_returns_some() {
    let obj = NixlObject::new(64, Some(b"object-key-abc"));
    assert_eq!(obj.nixl_metadata(), Some(b"object-key-abc".to_vec()));
}

#[test]
fn nixl_object_metadata_returns_none() {
    let obj = NixlObject::new(64, None);
    assert_eq!(obj.nixl_metadata(), None);
}

#[test]
fn nixl_object_metadata_is_also_nixl_descriptor() {
    // NixlObjectMetadata: NixlDescriptor — verify the supertrait is accessible
    let obj = NixlObject::new(128, Some(b"key"));
    assert_eq!(obj.mem_type(), MemType::Dram);
    assert_eq!(obj.device_id(), 0);
    assert_eq!(obj.size(), 128);
}

#[test]
fn nixl_object_default_descriptor_metadata_is_none() {
    // NixlObject doesn't override NixlDescriptor::metadata() so it should return None,
    // while nixl_metadata() returns Some. The two methods are independent.
    let obj = NixlObject::new(64, Some(b"key"));
    assert_eq!(obj.metadata(), None);
    assert_eq!(obj.nixl_metadata(), Some(b"key".to_vec()));
}
