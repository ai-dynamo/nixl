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

use nixl_sys::{
    Agent, MemType, MemoryRegion, NixlRegistration, NotificationMap, OptArgs, SystemStorage,
    XferDescList, XferOp,
};
use std::env;
use std::error::Error;
use std::thread;
use std::time::Duration;

// Use the same agent names as in working C++ example
const AGENT1_NAME: &str = "Agent001";
const AGENT2_NAME: &str = "Agent002";

fn main() -> Result<(), Box<dyn Error>> {
    println!("NIXL Fixed Single Process Example");
    println!("================================");

    // Set essential UCX environment variables like in the working minimal example
    println!("Setting UCX environment variables...");
    env::set_var("UCX_NET_DEVICES", "enp4s0");
    env::set_var("UCX_TCP_CM_REUSEADDR", "y");
    env::set_var("UCX_TLS", "tcp");
    env::set_var("UCX_WARN_UNUSED_ENV_VARS", "n");

    // Create two agents in same process (like C++ example)
    println!("Creating agents...");
    let agent1 = Agent::new(AGENT1_NAME)?;
    let agent2 = Agent::new(AGENT2_NAME)?;
    println!("Created agents: {} and {}", AGENT1_NAME, AGENT2_NAME);

    // Get available plugins
    let plugins = agent1.get_available_plugins()?;
    println!("Available plugins:");
    for plugin in plugins.iter() {
        println!("{}", plugin?);
    }

    // Create UCX backends
    println!("Creating UCX backends...");
    let (_, params1) = agent1.get_plugin_params("UCX")?;
    let (_, params2) = agent2.get_plugin_params("UCX")?;

    let backend1 = agent1.create_backend("UCX", &params1)?;
    let backend2 = agent2.create_backend("UCX", &params2)?;
    println!("Created UCX backends");

    // Create and register memory
    println!("Allocating and registering memory...");
    let buffer_size = 1024;

    // Agent 1 memory
    let mut storage1 = SystemStorage::new(buffer_size)?;
    storage1.memset(0xaa);

    // Agent 2 memory
    let mut storage2 = SystemStorage::new(buffer_size)?;
    storage2.memset(0xbb);

    // Create opt args with backends
    let mut opt_args1 = OptArgs::new()?;
    opt_args1.add_backend(&backend1)?;

    let mut opt_args2 = OptArgs::new()?;
    opt_args2.add_backend(&backend2)?;

    // Register memory
    storage1.register(&agent1, &opt_args1)?;
    storage2.register(&agent2, &opt_args2)?;

    println!("Registered memory for both agents");
    println!("Agent1's storage address: {:p}", unsafe {
        storage1.as_ptr().unwrap()
    });
    println!("Agent2's storage address: {:p}", unsafe {
        storage2.as_ptr().unwrap()
    });

    // Exchange metadata
    println!("\nMetadata Exchange Demo");
    println!("==========================");

    // 1. Get Local Metadata
    println!("\n1. Getting local metadata...");
    let md1 = agent1.get_local_md()?;
    let md2 = agent2.get_local_md()?;

    println!("Agent1 metadata size: {} bytes", md1.len());
    println!("Agent2 metadata size: {} bytes", md2.len());

    // 2. Exchange Metadata directly
    println!("\n2. Exchanging metadata directly...");
    let agent2_name = agent1.load_remote_md(&md2)?;
    println!("Agent1 loaded Agent2's metadata: {}", agent2_name);

    let agent1_name = agent2.load_remote_md(&md1)?;
    println!("Agent2 loaded Agent1's metadata: {}", agent1_name);

    println!("Metadata exchange successful!");

    // Check initial memory contents
    let data1 = storage1.as_slice();
    let data2 = storage2.as_slice();

    println!("Before transfer:");
    println!("Agent1 memory byte[0]: 0x{:02x}", data1[0]);
    println!("Agent2 memory byte[0]: 0x{:02x}", data2[0]);

    // Try a data transfer - key settings from minimal example:
    // 1. Small transfer size (4 bytes)
    // 2. From beginning of buffer (offset 0)
    // 3. To beginning of buffer (offset 0)
    println!("\nAttempting data transfer...");

    // Define transfer parameters - keep them very simple
    let req_size = 4;
    let src_offset = 0;
    let dst_offset = 0;

    // Create source descriptor for Agent1
    let mut src_desc = XferDescList::new(MemType::Dram)?;
    let src_ptr = unsafe {
        let base = storage1.as_ptr().unwrap() as usize;
        base + src_offset
    };
    src_desc.add_desc(src_ptr, req_size, 0)?;

    // Create destination descriptor for Agent2
    let mut dst_desc = XferDescList::new(MemType::Dram)?;
    let dst_ptr = unsafe {
        let base = storage2.as_ptr().unwrap() as usize;
        base + dst_offset
    };
    dst_desc.add_desc(dst_ptr, req_size, 0)?;

    println!(
        "Transfer request: {} bytes from {:p} to {:p}",
        req_size,
        unsafe { (storage1.as_ptr().unwrap() as usize + src_offset) as *const u8 },
        unsafe { (storage2.as_ptr().unwrap() as usize + dst_offset) as *const u8 }
    );

    // Wait to ensure connections are ready
    println!("Waiting for connections to be established...");
    thread::sleep(Duration::from_secs(1));

    // Create transfer options with notification
    let mut xfer_opt_args = OptArgs::new()?;
    xfer_opt_args.add_backend(&backend1)?;
    xfer_opt_args.set_notification_message(b"notification")?;
    xfer_opt_args.set_has_notification(true)?;

    // Try to create and post transfer
    println!("Creating transfer request...");
    let xfer_req = agent1.create_xfer_req(
        XferOp::Write,
        &src_desc,
        &dst_desc,
        AGENT2_NAME,
        Some(&xfer_opt_args),
    )?;

    println!("Posting transfer request...");
    agent1.post_xfer_req(&xfer_req, Some(&xfer_opt_args))?;
    println!("Transfer request posted successfully!");

    // Wait for notifications and transfer completion
    let mut completed = false;
    let mut received_notification = false;
    let start = std::time::Instant::now();

    println!("Waiting for transfer completion and notification...");
    while (!completed || !received_notification) && start.elapsed() < Duration::from_secs(5) {
        // Check transfer status
        if !completed {
            match agent1.get_xfer_status(&xfer_req) {
                Ok(status) => {
                    completed = !status;
                    if completed {
                        println!("Transfer completed!");
                    }
                }
                Err(e) => {
                    println!("Error checking transfer status: {:?}", e);
                    return Err(Box::new(e));
                }
            }
        }

        // Check for notifications on Agent2
        if !received_notification {
            let mut notifs = NotificationMap::new()?;
            agent2.get_notifications(&mut notifs, None)?;

            if !notifs.is_empty()? {
                println!("Agent2 received notification");
                received_notification = true;
            }
        }

        if !completed || !received_notification {
            thread::sleep(Duration::from_millis(100));
        }
    }

    if !completed || !received_notification {
        println!("Warning: Transfer or notification did not complete within timeout period.");
    } else {
        println!("Transfer and notification verified");
    }

    // Verify the memory content
    let data2_after = storage2.as_slice();
    println!("\nAfter transfer:");
    println!("Agent1 memory bytes[0..4]: {:?}", &data1[0..4]);
    println!("Agent2 memory bytes[0..4]: {:?}", &data2_after[0..4]);

    if data2_after[0] == 0xaa {
        println!("SUCCESSFUL - Memory was updated with Agent1's pattern (0xaa)!");
    } else {
        println!("FAILED - Memory was not updated as expected.");
        println!("  Expected: 0xaa, Got: 0x{:02x}", data2_after[0]);
    }

    println!("\nExample completed.");
    Ok(())
}
