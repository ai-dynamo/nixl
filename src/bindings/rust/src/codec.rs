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

//! Wire codec for `nixl-sys` descriptor lists.
//!
//! This module replaces the previously used `bincode` 1.x serializer with
//! `wincode` 0.6. `wincode`'s default wire format is byte-for-byte identical to
//! `bincode` 1.x for the shapes used here: little endian, fixed-width integers,
//! `u64` length prefixes, `u32` enum tags, and `usize` encoded as `u64`. Existing
//! on-the-wire buffers therefore stay readable and mixed-version deployments keep
//! interoperating, while the unmaintained `bincode` dependency
//! (`RUSTSEC-2025-0141`, no patched release) is removed.
//!
//! ## Preallocation limit
//!
//! `wincode`'s `DefaultConfig` caps container preallocation at 4 MiB. `bincode`
//! had no such cap, and a registration/transfer descriptor list is not bounded by
//! 4 MiB (legitimate lists can carry tens of MiB of descriptors and metadata).
//! [`WireConfig`] therefore raises the limit well above the default via
//! [`WIRE_PREALLOC_LIMIT`], while keeping every other wire parameter identical to
//! `bincode` 1.x (little endian, `u64` length prefixes, `u32` enum tags, `usize`
//! as `u64`).
//!
//! The limit stays **finite** on purpose: when decoding, `wincode` validates
//! `len * size_of::<Element>()` against it and returns an error for an implausibly
//! large length instead of panicking or attempting a huge allocation. This holds
//! for both statically- and dynamically-sized elements (e.g. `RegDescriptor`, whose
//! `metadata: Vec<u8>` makes its encoded size dynamic), so a `u64::MAX` descriptor
//! count is rejected up front rather than triggering a `capacity overflow` panic.
//! A finite cap is the only guard, because descriptor lists arrive from the network
//! and are untrusted.

use wincode::config::Configuration;

/// Maximum bytes `wincode` may preallocate while decoding a sequence.
///
/// `len * size_of::<Element>()` is checked against this bound and an error is
/// returned (never a panic or OOM) when a decoded length is implausibly large.
/// 4 MiB (wincode's default) is too small for real NIXL descriptor lists, so we
/// raise it to 256 MiB; it deliberately stays finite to stop untrusted input from
/// driving a huge allocation.
pub(crate) const WIRE_PREALLOC_LIMIT: usize = 256 * 1024 * 1024;

/// Wire configuration shared by every `nixl-sys` (de)serialization.
///
/// `ZERO_COPY_ALIGN_CHECK` is enabled and the preallocation size limit is
/// [`WIRE_PREALLOC_LIMIT`]; length encoding (`BincodeLen`), byte order
/// (`LittleEndian`), integer encoding (`FixInt`) and tag encoding (`u32`) all match
/// the `bincode` 1.x defaults.
pub(crate) type WireConfig = Configuration<true, WIRE_PREALLOC_LIMIT>;

const WIRE_CONFIG: WireConfig =
    Configuration::default().with_preallocation_size_limit::<WIRE_PREALLOC_LIMIT>();

/// Serialize `value` into the `nixl-sys` wire format.
pub(crate) fn serialize<T>(value: &T) -> Result<Vec<u8>, wincode::WriteError>
where
    T: wincode::SchemaWrite<WireConfig, Src = T> + ?Sized,
{
    wincode::config::serialize(value, WIRE_CONFIG)
}

/// Deserialize a value from the `nixl-sys` wire format.
pub(crate) fn deserialize<'de, T>(bytes: &'de [u8]) -> Result<T, wincode::ReadError>
where
    T: wincode::SchemaRead<'de, WireConfig, Dst = T>,
{
    wincode::config::deserialize(bytes, WIRE_CONFIG)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_vec_u32_is_eight_zero_bytes() {
        // BincodeLen uses a u64 length prefix; an empty Vec therefore serializes
        // to eight zero bytes.
        let bytes = serialize(&Vec::<u32>::new()).unwrap();
        assert_eq!(bytes, vec![0u8; 8]);
    }

    #[test]
    fn vec_u32_layout_is_u64_length_prefix_then_le_elements() {
        let bytes = serialize(&vec![1u32, 2, 3]).unwrap();
        // u64 length prefix (= 3) followed by three little-endian u32s.
        assert_eq!(
            bytes,
            vec![
                3, 0, 0, 0, 0, 0, 0, 0, // length prefix
                1, 0, 0, 0, // 1
                2, 0, 0, 0, // 2
                3, 0, 0, 0, // 3
            ]
        );
    }

    #[test]
    fn truncated_input_is_rejected() {
        let bytes = serialize(&vec![1u32, 2, 3]).unwrap();
        // Drop the final element's bytes; the remaining input is shorter than the
        // declared length.
        let truncated = &bytes[..bytes.len() - 2];
        assert!(deserialize::<Vec<u32>>(truncated).is_err());
    }

    #[test]
    fn oversized_length_prefix_is_rejected() {
        // A u64::MAX length prefix with no payload following must be rejected
        // without allocating.
        let malicious = u64::MAX.to_le_bytes().to_vec();
        assert!(deserialize::<Vec<u32>>(&malicious).is_err());
    }

    #[test]
    fn oversized_dynamic_element_count_is_rejected() {
        // `Vec<Vec<u8>>` has dynamically-sized elements, so `wincode` cannot
        // validate the declared length against a static-size window; it relies on
        // the preallocation limit instead. A `u64::MAX` count must be rejected as
        // an error, not a `capacity overflow` panic (this mirrors the untrusted
        // `RegDescData` decoding path, whose element type also has a `Vec<u8>`).
        let mut bytes = serialize(&Vec::<Vec<u8>>::new()).unwrap();
        bytes[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(deserialize::<Vec<Vec<u8>>>(&bytes).is_err());
    }
}
