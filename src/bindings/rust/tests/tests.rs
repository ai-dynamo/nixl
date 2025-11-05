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

use nixl_sys::*;
use std::env;

// Helper function to create an agent with error handling
fn create_test_agent(name: &str) -> Result<Agent, NixlError> {
    Agent::new(name)
}

fn setup_agent_with_backend(agent: &Agent) -> Result<OptArgs, NixlError> {
    let plugins = agent.get_available_plugins().expect("Failed to get available plugins");
    let plugin_name = find_plugin(&plugins, "UCX").expect("Failed to find plugin");
    let (_mems, params) = agent.get_plugin_params(&plugin_name).expect("Failed to get plugin params");
    agent.create_backend(&plugin_name, &params).expect("Failed to create backend");

    let mut opt_args = OptArgs::new().expect("Failed to create opt args");
    let _ = opt_args.add_backend(&agent.get_backend("UCX").unwrap());

    Ok(opt_args)
}

fn create_agent_with_backend(name: &str) -> Result<(Agent, OptArgs), NixlError> {
    let agent = Agent::new(name).expect("Failed to create agent");
    let plugins = agent.get_available_plugins().expect("Failed to get available plugins");
    let plugin_name = find_plugin(&plugins, "UCX").expect("Failed to find plugin");
    let (_mems, params) = agent.get_plugin_params(&plugin_name).expect("Failed to get plugin params");
    agent.create_backend(&plugin_name, &params).expect("Failed to create backend");

    let mut opt_args = OptArgs::new().expect("Failed to create opt args");
    let _ = opt_args.add_backend(&agent.get_backend("UCX").unwrap());

    Ok((agent, opt_args))
}


fn create_storage_list(agent: &Agent, opt_args: &OptArgs, size: usize) -> Vec<SystemStorage> {
    let mut storage_list = Vec::new();
    for _ in 0..size {
        let mut storage = SystemStorage::new(1024).unwrap();
        storage.register(agent, Some(opt_args)).expect("Failed to register storage memory");
        storage.memset(0);
        agent.register_memory(&storage, Some(opt_args)).expect("Failed to register storage memory");
        storage_list.push(storage);
    }
    storage_list
}

fn create_dlist<'a>(storage_list: &'a mut Vec<SystemStorage>) -> Result<XferDescList<'a>, NixlError> {
    let mut dlist = XferDescList::new(MemType::Dram).expect("Failed to create XferDescList");
    for storage in storage_list.iter_mut() {
        dlist.add_storage_desc(storage).expect(&format!("Failed to add storage descriptor for storage"));
    }
    Ok(dlist)
}

fn exchange_metadata(agent1: &Agent, agent2: &Agent) -> Result<(), NixlError> {
    let metadata1 = agent1.get_local_md().expect("Failed to get local metadata");
    let metadata2 = agent2.get_local_md().expect("Failed to get local metadata");
    agent1.load_remote_md(&metadata2).expect("Failed to load remote metadata");
    agent2.load_remote_md(&metadata1).expect("Failed to load remote metadata");
    Ok(())
}

// Helper function to find a plugin by name
fn find_plugin(plugins: &StringList, name: &str) -> Result<String, NixlError> {
    plugins
        .iter()
        .filter_map(Result::ok)
        .find(|&plugin| plugin == name)
        .map(ToString::to_string)
        .or_else(|| plugins.get(0).ok().map(ToString::to_string))
        .ok_or(NixlError::InvalidParam)
}

/// Helper function to create and initialize a POSIX backend with optional arguments
/// Returns (backend, opt_args) if POSIX is available, or None if not available
fn create_posix_backend(agent: &Agent) -> Option<(Backend, OptArgs)> {
    // Get available plugins - check if POSIX is available
    let plugins = agent
        .get_available_plugins()
        .expect("Failed to get plugins");

    if !plugins
        .iter()
        .any(|p| p.as_ref().map(|s| *s == "POSIX").unwrap_or(false))
    {
        println!("POSIX plugin not available, skipping test");
        return None;
    }

    // Get plugin parameters and create POSIX backend
    let (_mems, params) = agent
        .get_plugin_params("POSIX")
        .expect("Failed to get POSIX plugin params");

    let backend = agent
        .create_backend("POSIX", &params)
        .expect("Failed to create POSIX backend");

    // Create optional arguments with the backend
    let mut opt_args = OptArgs::new().expect("Failed to create opt args");
    opt_args
        .add_backend(&backend)
        .expect("Failed to add backend");

    Some((backend, opt_args))
}

#[test]
fn create_agent_with_custom_config() {
    // Ensure we can construct with non-default config
    let cfg = AgentConfig {
        enable_listen_thread: true,
        listen_port: 0,
        capture_telemetry: false,
        ..Default::default()
    };

    let agent = Agent::new_configured("cfg_agent", &cfg)
        .expect("Failed to create configured agent");

    // basic sanity: can query available plugins
    let _plugins = agent.get_available_plugins().expect("Failed to get plugins");
}

#[test]
fn test_agent_creation() {
    let agent = Agent::new("test_agent").expect("Failed to create agent");
    drop(agent);
}

#[test]
fn test_agent_invalid_name() {
    let result = Agent::new("test\0agent");
    assert!(matches!(result, Err(NixlError::StringConversionError(_))));
}

#[test]
fn test_get_available_plugins() {
    let agent = Agent::new("test_agent").expect("Failed to create agent");
    let plugins = agent
        .get_available_plugins()
        .expect("Failed to get plugins");

    // Print available plugins
    for plugin in plugins.iter() {
        println!("Found plugin: {}", plugin.unwrap());
    }
}

#[test]
fn test_get_plugin_params() {
    let agent = Agent::new("test_agent").expect("Failed to create agent");
    let (_mems, _params) = agent
        .get_plugin_params("UCX")
        .expect("Failed to get plugin params");
    // MemList and Params will be automatically dropped here
}

#[test]
fn test_backend_creation() {
    let agent = Agent::new("test_agent").expect("Failed to create agent");
    let (_mems, params) = agent
        .get_plugin_params("UCX")
        .expect("Failed to get plugin params");
    let backend = agent
        .create_backend("UCX", &params)
        .expect("Failed to create backend");

    let mut opt_args = OptArgs::new().expect("Failed to create opt args");
    opt_args
        .add_backend(&backend)
        .expect("Failed to add backend");
}

#[test]
fn test_params_iteration() {
    let agent = Agent::new("test_agent").expect("Failed to create agent");
    let (mems, params) = agent
        .get_plugin_params("UCX")
        .expect("Failed to get plugin params");

    println!("Parameters:");
    if !params.is_empty().unwrap() {
        for param in params.iter().unwrap() {
            let param = param.unwrap();
            println!("  {} = {}", param.key, param.value);
        }
    } else {
        println!("  (empty)");
    }

    println!("Memory types:");
    if !mems.is_empty().unwrap() {
        for mem_type in mems.iter() {
            println!("  {}", mem_type.unwrap());
        }
    } else {
        println!("  (empty)");
    }
}

// #[test]
// fn test_get_backend_params() {
//     let agent = Agent::new("test_agent").unwrap();
//     let plugins = agent.get_available_plugins().unwrap();
//     assert!(!plugins.is_empty().unwrap_or(false));

//     let plugin_name = plugins.get(0).unwrap();
//     let (_mems, params) = agent.get_plugin_params(plugin_name).unwrap();
//     let backend = agent.create_backend(plugin_name, &params).unwrap();

//     // Get backend params after initialization
//     let (backend_mems, backend_params) = agent.get_backend_params(&backend).unwrap();

//     // Verify we can access the parameters
//     let param_iter = backend_params.iter().unwrap();
//     for param in param_iter {
//         let param = param.unwrap();
//         println!("Backend param: {} = {}", param.key, param.value);
//     }

//     // Verify we can access the memory types
//     for mem_type in backend_mems.iter() {
//         println!("Backend memory type: {:?}", mem_type.unwrap());
//     }
// }

#[test]
fn test_get_backend_params() -> Result<(), NixlError> {
    let agent = create_test_agent("test_agent")?;
    let plugins = agent.get_available_plugins()?;

    // Ensure we have at least one plugin
    assert!(!plugins.is_empty()?);

    // Try UCX plugin first since it doesn't require GPU
    let plugin_name = find_plugin(&plugins, "UCX")?;
    let (_mems, params) = agent.get_plugin_params(&plugin_name)?;
    let backend = agent.create_backend(&plugin_name, &params)?;

    // Get backend params after initialization
    let (backend_mems, backend_params) = agent.get_backend_params(&backend)?;

    // Print parameters using iterator
    let param_iter = backend_params.iter()?;
    for param in param_iter.flatten() {
        println!("Backend param: {} = {}", param.key, param.value);
    }

    // Print memory types
    for mem_type in backend_mems.iter().flatten() {
        println!("Backend memory type: {:?}", mem_type);
    }

    Ok(())
}

#[test]
fn test_xfer_dlist() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();

    // Add some descriptors
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    dlist.add_desc(0x2000, 0x200, 1).unwrap();

    // Check length
    assert_eq!(dlist.len().unwrap(), 2);

    // Clear list
    dlist.clear().unwrap();
    assert_eq!(dlist.len().unwrap(), 0);

    // Resize list
    dlist.resize(5).unwrap();
}

#[test]
fn test_reg_dlist() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();

    // Add some descriptors
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    dlist.add_desc(0x2000, 0x200, 1).unwrap();

    // Check length
    assert_eq!(dlist.len().unwrap(), 2);

    // Clear list
    dlist.clear().unwrap();
    assert_eq!(dlist.len().unwrap(), 0);

    // Resize list
    dlist.resize(5).unwrap();
}

#[test]
fn test_storage_descriptor_lifetime() {
    // Create storage that outlives the descriptor list
    let storage = SystemStorage::new(1024).unwrap();

    {
        // Create a descriptor list with shorter lifetime
        let mut dlist = XferDescList::new(MemType::Dram).unwrap();
        dlist.add_storage_desc(&storage).unwrap();
        assert_eq!(dlist.len().unwrap(), 1);
        // dlist is dropped here, but storage is still valid
    }

    // MemoryRegion is still valid here
    assert_eq!(<SystemStorage as MemoryRegion>::size(&storage), 1024);
}

#[test]
fn test_multiple_storage_descriptors() {
    let storage1 = SystemStorage::new(1024).unwrap();
    let storage2 = SystemStorage::new(2048).unwrap();

    let mut dlist = XferDescList::new(MemType::Dram).unwrap();

    // Add multiple descriptors
    dlist.add_storage_desc(&storage1).unwrap();
    dlist.add_storage_desc(&storage2).unwrap();

    assert_eq!(dlist.len().unwrap(), 2);
}

#[test]
fn test_memory_registration() {
    let agent = Agent::new("test_agent").unwrap();
    let mut storage = SystemStorage::new(1024).unwrap();

    // Register memory
    storage.register(&agent, None).unwrap();

    // Verify we can still access the memory
    storage.memset(0xAA);
    assert!(storage.as_slice().iter().all(|&x| x == 0xAA));
}

#[test]
fn test_registration_handle_drop() {
    let agent = Agent::new("test_agent").unwrap();
    let mut storage = SystemStorage::new(1024).unwrap();

    // Register memory
    storage.register(&agent, None).unwrap();

    // Drop the storage, which should trigger deregistration
    drop(storage);

    // Create new storage to verify we can register again
    let mut new_storage = SystemStorage::new(1024).unwrap();
    new_storage.register(&agent, None).unwrap();
}

#[test]
fn test_multiple_registrations() {
    let agent = Agent::new("test_agent").unwrap();
    let mut storage1 = SystemStorage::new(1024).unwrap();
    let mut storage2 = SystemStorage::new(2048).unwrap();

    // Register both storages
    storage1.register(&agent, None).unwrap();
    storage2.register(&agent, None).unwrap();

    // Verify we can still access both memories
    storage1.memset(0xAA);
    storage2.memset(0xBB);
    assert!(storage1.as_slice().iter().all(|&x| x == 0xAA));
    assert!(storage2.as_slice().iter().all(|&x| x == 0xBB));
}

#[test]
fn test_make_connection_success() {
    let agent = Agent::new("test_agent").expect("Failed to create agent");
    let remote_agent = Agent::new("remote_agent").expect("Failed to create remote agent");
    let opt_args = setup_agent_with_backend(&agent).expect("Failed to setup agent");
    let _opt_args_remote = setup_agent_with_backend(&remote_agent).expect("Failed to setup agent");

    exchange_metadata(&agent, &remote_agent).expect("Failed to exchange metadata");

    // This should succeed if the agent is valid and the backend is set up
    let result = agent.make_connection(&remote_agent.name(), Some(&opt_args));

    assert!(
        result.is_ok(),
        "Expected Ok got: {:?}",
        result
    );
}

#[test]
fn test_make_connection_invalid_param() {
    let agent = Agent::new("test_agent").expect("Failed to create agent");
    // Null bytes in the name should trigger InvalidParam or StringConversionError
    let result = agent.make_connection("remote\0agent", None);
    assert!(
        matches!(result, Err(NixlError::StringConversionError(_))) ||
        matches!(result, Err(NixlError::InvalidParam)),
        "Expected StringConversionError or InvalidParam, got: {:?}",
        result
    );
}

#[test]
fn test_get_local_md() {
    let agent = Agent::new("test_agent").unwrap();

    // Get available plugins and print their names
    let plugins = agent.get_available_plugins().unwrap();
    for plugin in plugins.iter() {
        println!("Found plugin: {}", plugin.unwrap());
    }

    // Get plugin parameters for both agents
    let (_mem_list, params) = agent.get_plugin_params("UCX").unwrap();

    // Create backends for both agents
    let backend1 = agent.create_backend("UCX", &params).unwrap();

    let md = agent.get_local_md().unwrap();

    // Measure the size
    let initial_size = md.len();
    println!("Local metadata size: {}", initial_size);

    let mut opt_args = OptArgs::new().unwrap();
    opt_args.add_backend(&backend1).unwrap();

    let mut storages = Vec::new();

    for _i in 0..10 {
        // Register some memory regions
        let mut storage = SystemStorage::new(1024).unwrap();
        storage.register(&agent, Some(&opt_args)).unwrap();
        storages.push(storage);
    }

    let md = agent.get_local_md().unwrap();

    // Measure the size again
    let final_size = md.len();
    println!("Local metadata size: {}", final_size);

    // Check if the size has increased
    assert!(final_size > initial_size);
}

#[test]
fn test_metadata_exchange() {
    // Create two agents
    let agent2 = Agent::new("agent2").unwrap();
    let agent1 = Agent::new("agent1").unwrap();

    // Get plugin parameters for both agents
    let (_mem_list, params) = agent1.get_plugin_params("UCX").unwrap();

    // Create backends for both agents
    let _backend1 = agent1.create_backend("UCX", &params).unwrap();
    let _backend2 = agent2.create_backend("UCX", &params).unwrap();

    // Get metadata from agent1
    let md = agent1.get_local_md().unwrap();

    // Load metadata into agent2
    let remote_name = agent2.load_remote_md(&md).unwrap();
    assert_eq!(remote_name, "agent1");
}

#[test]
fn test_basic_agent_lifecycle() {
    // Create two agents
    let agent2 = Agent::new("A2").unwrap();
    let agent1 = Agent::new("A1").unwrap();

    // Get available plugins and print their names
    let plugins = agent1.get_available_plugins().unwrap();
    for plugin in plugins.iter() {
        println!("Found plugin: {}", plugin.unwrap());
    }

    // Get plugin parameters for both agents
    let (_mem_list1, _params) = agent1.get_plugin_params("UCX").unwrap();
    let (_mem_list2, params) = agent2.get_plugin_params("UCX").unwrap();

    // Create backends for both agents
    let backend1 = agent1.create_backend("UCX", &params).unwrap();
    let backend2 = agent2.create_backend("UCX", &params).unwrap();

    // Create optional arguments and add backends
    let mut opt_args = OptArgs::new().unwrap();
    opt_args.add_backend(&backend1).unwrap();
    opt_args.add_backend(&backend2).unwrap();

    // Allocate and initialize memory regions
    let mut storage1 = SystemStorage::new(256).unwrap();
    let mut storage2 = SystemStorage::new(256).unwrap();

    // Initialize memory patterns
    storage1.memset(0xbb);
    storage2.memset(0x00);

    // Verify memory patterns
    assert!(storage1.as_slice().iter().all(|&x| x == 0xbb));
    assert!(storage2.as_slice().iter().all(|&x| x == 0x00));

    // Create registration descriptor lists
    storage1.register(&agent1, None).unwrap();
    storage2.register(&agent2, None).unwrap();

    // Mimic transferring metadata from agent2 to agent1
    let metadata = agent2.get_local_md().unwrap();
    let remote_name = agent1.load_remote_md(&metadata).unwrap();
    assert_eq!(remote_name, "A2");

    let mut local_xfer_dlist = XferDescList::new(MemType::Dram).unwrap();
    local_xfer_dlist.add_storage_desc(&storage1).unwrap();

    let mut remote_xfer_dlist = XferDescList::new(MemType::Dram).unwrap();
    remote_xfer_dlist.add_storage_desc(&storage2).unwrap();

    let mut xfer_args = OptArgs::new().unwrap();
    xfer_args.set_has_notification(true).unwrap();
    xfer_args.set_notification_message(b"notification").unwrap();

    let xfer_req = agent1
        .create_xfer_req(
            XferOp::Write,
            &local_xfer_dlist,
            &remote_xfer_dlist,
            &remote_name,
            Some(&xfer_args),
        )
        .unwrap();

    let _status = agent1.post_xfer_req(&xfer_req, None).unwrap();

    println!("Waiting for local completions");

    loop {
        let status = agent1.get_xfer_status(&xfer_req).unwrap();

        if status.is_success() {
            println!("Xfer req completed");
            break;
        } else {
            println!("Xfer req not completed");
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let mut notifs = NotificationMap::new().unwrap();
    let notify_map;
    println!("Waiting for notifications");
    std::thread::sleep(std::time::Duration::from_millis(100));

    loop {
        agent2.get_notifications(&mut notifs, None).unwrap();
        if !notifs.is_empty().unwrap() {
            notify_map = notifs.take_notifs().unwrap();
            assert_eq!(notify_map.len(), 1);
            assert!(notifs.is_empty().unwrap());
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    println!("Got notifications");

    // Get first notification from first agent
    let vals = notify_map.get("A1").unwrap();
    assert_eq!(vals.len(), 1);
    assert_eq!(vals[0], "notification");

    // Verify memory patterns
    assert!(storage1.as_slice().iter().all(|&x| x == 0xbb));
    assert!(storage2.as_slice().iter().all(|&x| x == 0xbb));
}

#[test]
fn test_etcd_metadata_exchange() -> Result<(), NixlError> {
    // Check if NIXL_ETCD_ENDPOINTS env var is set to skip test if not
    if env::var("NIXL_ETCD_ENDPOINTS").is_err() {
        println!("Skipping etcd test - NIXL_ETCD_ENDPOINTS not set");
        return Ok(());
    }

    // Create two agents for metadata exchange
    let agent1 = Agent::new("EtcdAgent1")?;
    let agent2 = Agent::new("EtcdAgent2")?;

    // Get UCX backend to add to optional arguments
    let plugins = agent1.get_available_plugins()?;
    let plugin_name = find_plugin(&plugins, "UCX")?;
    let (_mems, params) = agent1.get_plugin_params(&plugin_name)?;
    let backend = agent1.create_backend(&plugin_name, &params)?;

    // Create OptArgs with backend
    let mut opt_args = OptArgs::new()?;
    opt_args.add_backend(&backend)?;

    // Send agent1's metadata to etcd
    agent1.send_local_md(Some(&opt_args))?;
    println!("Successfully sent agent1 metadata to etcd");

    // Fetch agent1's metadata from etcd with agent2
    agent2.fetch_remote_md("EtcdAgent1", Some(&opt_args))?;
    println!("Successfully fetched agent1 metadata from etcd");

    // Invalidate agent1's metadata in etcd
    agent1.invalidate_local_md(Some(&opt_args))?;
    println!("Successfully invalidated agent1 metadata in etcd");

    Ok(())
}

#[test]
fn test_send_notification() -> Result<(), NixlError> {
    // Create two agents for notification exchange
    let agent1 = Agent::new("NotifSender")?;
    let agent2 = Agent::new("NotifReceiver")?;

    // Set up backends for both agents
    let (_mem_list, params) = agent1.get_plugin_params("UCX")?;
    let backend1 = agent1.create_backend("UCX", &params)?;
    let backend2 = agent2.create_backend("UCX", &params)?;

    // Exchange metadata
    let metadata = agent2.get_local_md()?;
    agent1.load_remote_md(&metadata)?;

    // Create notification message
    let message = b"Test notification message";

    // Send notification with no backend specified
    agent1.send_notification("NotifReceiver", message, None)?;

    // Send notification with specific backend
    agent1.send_notification("NotifReceiver", message, Some(&backend1))?;

    // Create a notification map to receive notifications
    let mut notifs = NotificationMap::new()?;

    // Receive notifications without backend
    agent2.get_notifications(&mut notifs, None)?;

    // Receive notifications with specific backend
    let mut opt_args = OptArgs::new()?;
    opt_args.add_backend(&backend2)?;
    agent2.get_notifications(&mut notifs, Some(&opt_args))?;

    // Verify notification map contents
    if !notifs.is_empty()? {
        let mut agents = notifs.agents();

        // Should have notifications from NotifSender
        if let Some(Ok(agent_name)) = agents.next() {
            assert_eq!(agent_name, "NotifSender");

            // Verify notification content
            let notifications = notifs.get_notifications(agent_name)?;
            let notif_count = notifs.get_notifications_size(agent_name)?;

            // May have 1 or 2 notifications depending on whether both were processed
            assert!(notif_count > 0, "Should have at least one notification");

            // Check content of notification
            for notification in notifications {
                assert_eq!(notification?, message);
            }
        }
    }

    Ok(())
}

#[test]
fn test_check_remote_metadata() {
    // Create two agents
    let agent1 = Agent::new("agent1").expect("Failed to create agent1");
    let agent2 = Agent::new("agent2").expect("Failed to create agent2");

    // Set up backends for both agents (required before metadata operations)
    let (_mem_list, params) = agent1
        .get_plugin_params("UCX")
        .expect("Failed to get plugin params");
    let _backend1 = agent1
        .create_backend("UCX", &params)
        .expect("Failed to create backend for agent1");
    let _backend2 = agent2
        .create_backend("UCX", &params)
        .expect("Failed to create backend for agent2");

    // Initially, agent1 should not have metadata for agent2
    assert!(!agent1.check_remote_metadata("agent2", None));

    // Get and share metadata
    let metadata = agent2.get_local_md().expect("Failed to get local metadata");
    agent1
        .load_remote_md(&metadata)
        .expect("Failed to load remote metadata");

    // Now agent1 should have metadata for agent2
    assert!(agent1.check_remote_metadata("agent2", None));

    // Test with a descriptor list
    let mut storage = SystemStorage::new(1024).expect("Failed to create storage");
    let opt_args = OptArgs::new().expect("Failed to create opt args");
    storage
        .register(&agent2, Some(&opt_args))
        .expect("Failed to register memory");

    // Create descriptor list with memory that exists in agent2
    let mem_type = MemType::Dram;
    let mut xfer_desc_list =
        XferDescList::new(mem_type).expect("Failed to create xfer desc list");
    xfer_desc_list
        .add_desc(
            unsafe { storage.as_ptr() } as usize,
            storage.size(),
            storage.device_id(),
        )
        .expect("Failed to add descriptor");

    // Update metadata after registration
    let metadata = agent2
        .get_local_md()
        .expect("Failed to get updated local metadata");
    agent1
        .load_remote_md(&metadata)
        .expect("Failed to reload remote metadata");

    // Check with descriptor list - should return true for valid descriptors
    assert!(agent1.check_remote_metadata("agent2", Some(&xfer_desc_list)));

    // Create a descriptor list with invalid memory address
    let mut invalid_desc_list =
        XferDescList::new(mem_type).expect("Failed to create invalid desc list");
    invalid_desc_list
        .add_desc(0xdeadbeef, 1024, 0)
        .expect("Failed to add invalid descriptor");

    // Check with invalid descriptor list - should return false
    assert!(!agent1.check_remote_metadata("agent2", Some(&invalid_desc_list)));

    // Check with non-existent agent name
    assert!(!agent1.check_remote_metadata("non_existent_agent", None));

    // Check with invalid agent name (contains null byte)
    // The function should return false rather than panic
    let invalid_name = "invalid\0agent";
    assert!(!agent1.check_remote_metadata(invalid_name, None));
}

#[test]
fn test_xfer_desc_list_new() {
    let dlist = XferDescList::new(MemType::Dram).unwrap();
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_xfer_desc_list_get_type() {
    let dlist = XferDescList::new(MemType::Vram).unwrap();
    assert_eq!(dlist.get_type().unwrap(), MemType::Vram);
}

#[test]
fn test_xfer_desc_list_get_type_after_add() {
    let mut dlist = XferDescList::new(MemType::Block).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert_eq!(dlist.get_type().unwrap(), MemType::Block);
}

#[test]
fn test_xfer_desc_list_desc_count_basic() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    assert_eq!(dlist.desc_count().unwrap(), 0);
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert_eq!(dlist.desc_count().unwrap(), 1);
}

#[test]
fn test_xfer_desc_list_desc_count_after_clear() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    dlist.clear().unwrap();
    assert_eq!(dlist.desc_count().unwrap(), 0);
}

#[test]
fn test_xfer_desc_list_is_empty_true() {
    let dlist = XferDescList::new(MemType::Dram).unwrap();
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_xfer_desc_list_is_empty_false() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert!(!dlist.is_empty().unwrap());
}

#[test]
fn test_xfer_desc_list_trim_basic() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    dlist.trim().unwrap();
    assert!(dlist.desc_count().unwrap() <= 1);
}

#[test]
fn test_xfer_desc_list_trim_empty() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    assert!(dlist.trim().is_ok());
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_xfer_desc_list_rem_desc_basic() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert!(dlist.rem_desc(0).is_ok());
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_xfer_desc_list_rem_desc_out_of_bounds() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    assert!(dlist.rem_desc(0).is_err());
}

#[test]
fn test_xfer_desc_list_clear_basic() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    dlist.clear().unwrap();
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_xfer_desc_list_clear_empty() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    assert!(dlist.clear().is_ok());
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_xfer_desc_list_print_basic() {
    let dlist = XferDescList::new(MemType::Dram).unwrap();
    assert!(dlist.print().is_ok());
}

#[test]
fn test_xfer_desc_list_print_after_add() {
    let mut dlist = XferDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert!(dlist.print().is_ok());
}

// ----------- RegDescList API TESTS -----------

#[test]
fn test_reg_desc_list_new() {
    let dlist = RegDescList::new(MemType::Dram).unwrap();
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_reg_desc_list_get_type() {
    let dlist = RegDescList::new(MemType::Vram).unwrap();
    assert_eq!(dlist.get_type().unwrap(), MemType::Vram);
}

#[test]
fn test_reg_desc_list_get_type_after_add() {
    let mut dlist = RegDescList::new(MemType::Block).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert_eq!(dlist.get_type().unwrap(), MemType::Block);
}

#[test]
fn test_reg_desc_list_desc_count_basic() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    assert_eq!(dlist.desc_count().unwrap(), 0);
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert_eq!(dlist.desc_count().unwrap(), 1);
}

#[test]
fn test_reg_desc_list_desc_count_after_clear() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    dlist.clear().unwrap();
    assert_eq!(dlist.desc_count().unwrap(), 0);
}

#[test]
fn test_reg_desc_list_is_empty_true() {
    let dlist = RegDescList::new(MemType::Dram).unwrap();
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_reg_desc_list_is_empty_false() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert!(!dlist.is_empty().unwrap());
}

#[test]
fn test_reg_desc_list_trim_basic() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    dlist.trim().unwrap();
    assert!(dlist.desc_count().unwrap() <= 1);
}

#[test]
fn test_reg_desc_list_trim_empty() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    assert!(dlist.trim().is_ok());
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_reg_desc_list_rem_desc_basic() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert!(dlist.rem_desc(0).is_ok());
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_reg_desc_list_rem_desc_out_of_bounds() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    assert!(dlist.rem_desc(0).is_err());
}

#[test]
fn test_reg_desc_list_clear_basic() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    dlist.clear().unwrap();
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_reg_desc_list_clear_empty() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    assert!(dlist.clear().is_ok());
    assert!(dlist.is_empty().unwrap());
}

#[test]
fn test_reg_desc_list_print_basic() {
    let dlist = RegDescList::new(MemType::Dram).unwrap();
    assert!(dlist.print().is_ok());
}

#[test]
fn test_reg_desc_list_print_after_add() {
    let mut dlist = RegDescList::new(MemType::Dram).unwrap();
    dlist.add_desc(0x1000, 0x100, 0).unwrap();
    assert!(dlist.print().is_ok());
}

#[test]
fn test_query_mem_with_files() {
    use std::fs::File;
    use std::io::Write;

    // Constants
    const DESCRIPTOR_ADDR: usize = 0;
    const DESCRIPTOR_SIZE: usize = 1024;
    const DESCRIPTOR_DEV_ID: u64 = 0;
    const NUM_FILES_TO_CREATE: usize = 2;
    const EXPECTED_NUM_RESPONSES: usize = 3;

    // Create a unique temporary directory for this test
    let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
    let temp_dir_path = temp_dir.path();

    // Define test files
    let test_files = vec![
        ("test_query_mem_rust_1.txt", "Test content for file 1"),
        ("test_query_mem_rust_2.txt", "Test content for file 2"),
        ("non_existent_file_rust.txt", ""), // This file won't be created
    ];

    // Create temporary test files
    let mut file_paths: Vec<_> = test_files
        .iter()
        .take(NUM_FILES_TO_CREATE) // Only create the first two files
        .map(|(filename, content)| {
            let file_path = temp_dir_path.join(filename);
            let mut file =
                File::create(&file_path).expect(&format!("Failed to create {}", filename));
            writeln!(file, "{}", content).expect(&format!("Failed to write to {}", filename));
            file_path
        })
        .collect();

    // Add the non-existent file path
    file_paths.push(temp_dir_path.join(test_files[2].0));

    // Create agent
    let agent = Agent::new("test_agent").expect("Failed to create agent");

    // Create POSIX backend
    let (_backend, opt_args) = match create_posix_backend(&agent) {
        Some(result) => result,
        None => return,
    };

    // Create descriptor list with existing and non-existing files
    let mut descs =
        RegDescList::new(MemType::File).expect("Failed to create descriptor list");

    // Add blob descriptors with filenames as metadata
    for (i, file_path) in file_paths.iter().enumerate() {
        descs
            .add_desc_with_meta(
                DESCRIPTOR_ADDR,
                DESCRIPTOR_SIZE,
                DESCRIPTOR_DEV_ID,
                file_path.to_string_lossy().as_bytes(),
            )
            .expect(&format!("Failed to add descriptor for file {}", i + 1));
    }

    // Query memory
    let resp = agent
        .query_mem(&descs, Some(&opt_args))
        .expect("Failed to query mem");

    // Verify results
    assert_eq!(
        resp.len().unwrap(),
        EXPECTED_NUM_RESPONSES,
        "Expected 3 responses"
    );

    // Check responses - current order: existing file, existing file, non-existent file
    let responses: Vec<_> = resp.iter().unwrap().collect();

    assert!(responses[0].has_value().unwrap(), "First file should exist");
    assert!(
        responses[1].has_value().unwrap(),
        "Second file should exist"
    );
    assert!(
        !responses[2].has_value().unwrap(),
        "Third file should not exist"
    );

    // Print parameters for existing files
    for (i, response) in responses.iter().enumerate() {
        if response.has_value().unwrap() {
            if let Some(params) = response.get_params().unwrap() {
                println!("Parameters for response {}:", i);
                for param in params.iter().unwrap() {
                    let param = param.unwrap();
                    println!("  {} = {}", param.key, param.value);
                    // POSIX backend returns mtime and mode parameters
                    if param.key == "mtime" || param.key == "mode" {
                        assert!(
                            !param.value.is_empty(),
                            "Parameter value should not be empty"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_query_mem_empty_list() {
    // Constants
    const EXPECTED_EMPTY_RESPONSES: usize = 0;

    // Create agent
    let agent = Agent::new("test_agent").expect("Failed to create agent");

    // Create POSIX backend
    let (_backend, opt_args) = match create_posix_backend(&agent) {
        Some(result) => result,
        None => return,
    };

    // Create empty descriptor list
    let descs = RegDescList::new(MemType::File).expect("Failed to create descriptor list");

    // Query memory with empty list
    let resp = agent
        .query_mem(&descs, Some(&opt_args))
        .expect("Failed to query mem");

    // Verify results
    let num_responses = resp.len().expect("Failed to get response count");
    assert_eq!(
        num_responses, EXPECTED_EMPTY_RESPONSES,
        "Expected 0 responses for empty descriptor list"
    );
}

// Tests for prep_xfer_dlist API
#[test]
fn test_prep_xfer_dlist_success() {
    const DLIST_SIZE: usize = 10;

    // 1. Create agents and backends
    let (local_agent, opt_args) = create_agent_with_backend("local_agent").expect("Failed to create agent");
    let (remote_agent, _opt_args_remote) = create_agent_with_backend("remote_agent").expect("Failed to create agent");

    // 2. Create memory regions and register them
    let mut storage_list = create_storage_list(&local_agent, &opt_args, DLIST_SIZE);

    {
        // 3. Create transfer descriptor list
        let dlist = create_dlist(&mut storage_list).expect("Failed to create XferDescList");

        // 4. Exchange metadata
        exchange_metadata(&local_agent, &remote_agent).expect("Failed to exchange metadata");

        // 5. Prepare transfer descriptor list
        let result = local_agent.prepare_xfer_dlist("", &dlist, None);
        assert!(result.is_ok(), "prepare_xfer_dlist failed with error: {:?}", result.err());
    }
}

#[test]
fn test_prep_xfer_dlist_invalid_agent() {
    const DLIST_SIZE: usize = 10;

    let agent = Agent::new("test_agent").expect("Failed to create agent");
    let opt_args = setup_agent_with_backend(&agent).expect("Failed to setup agent");
    let mut storage_list = create_storage_list(&agent, &opt_args, DLIST_SIZE);
    {
        let dlist = create_dlist(&mut storage_list).expect("Failed to create XferDescList");

        // Try with invalid agent name
        let result = agent.prepare_xfer_dlist("invalid_agent", &dlist, None);

        assert!(
            result.is_err_and(|e| matches!(e, NixlError::BackendError)),
            "Expected InvalidParam for invalid agent name"
        );
    }
}

// Tests for make_xfer_req API
#[test]
fn test_make_xfer_req_success() {
    const DLIST_SIZE: usize = 10;

    let (local_agent, opt_args) = create_agent_with_backend("local_agent").expect("Failed to create agent");
    let (remote_agent, opt_args_remote) = create_agent_with_backend("remote_agent").expect("Failed to create agent");

    // 2. Create memory regions and register them
    let mut storage_list = create_storage_list(&local_agent, &opt_args, DLIST_SIZE);
    let mut remote_storage_list = create_storage_list(&remote_agent, &opt_args_remote, DLIST_SIZE);

    {
        let dlist = create_dlist(&mut storage_list).expect("Failed to create XferDescList");
        let remote_dlist = create_dlist(&mut remote_storage_list).expect("Failed to create XferDescList");

        // 4. Exchange metadata
        exchange_metadata(&local_agent, &remote_agent).expect("Failed to exchange metadata");

        // Prepare descriptor list handles
        let local_handle: XferDlistHandle = local_agent.prepare_xfer_dlist("", &dlist, Some(&opt_args))
            .expect("Failed to prepare local descriptor list");

        let remote_handle: XferDlistHandle = local_agent.prepare_xfer_dlist(remote_agent.name().as_str(), &remote_dlist, Some(&opt_args))
            .expect("Failed to prepare local descriptor list");

        // Create transfer request using prepared handles with indices
        let local_indices = (0..DLIST_SIZE).step_by(2).map(|i| i as i32).collect::<Vec<i32>>();
        let remote_indices = (1..DLIST_SIZE).step_by(2).map(|i| i as i32).collect::<Vec<i32>>();
        let result = local_agent.make_xfer_req(
            XferOp::Write,
            &local_handle,
            &local_indices,
            &remote_handle,
            &remote_indices,
            Some(&opt_args)
        );

        assert!(
            result.is_ok(),
            "make_xfer_req failed with error: {:?}", result.err()
        );
    }
}

#[test]
fn test_make_xfer_req_invalid_indices() {
    const DLIST_SIZE: usize = 10;
    let (agent1, opt_args) = create_agent_with_backend("agent1").expect("Failed to create agent");
    let (agent2, opt_args_remote) = create_agent_with_backend("agent2").expect("Failed to create agent");

    let mut storage_list = create_storage_list(&agent1, &opt_args, DLIST_SIZE);
    let mut remote_storage_list = create_storage_list(&agent2, &opt_args_remote, DLIST_SIZE);

    {
        let local_dlist = create_dlist(&mut storage_list).expect("Failed to create descriptor list");
        let remote_dlist = create_dlist(&mut remote_storage_list).expect("Failed to create descriptor list");

        exchange_metadata(&agent1, &agent2).expect("Failed to exchange metadata");

        // Prepare descriptor list handles
        let local_handle = agent1.prepare_xfer_dlist("", &local_dlist, Some(&opt_args))
            .expect("Failed to prepare local descriptor list");
        let remote_handle = agent1.prepare_xfer_dlist(agent2.name().as_str(), &remote_dlist, Some(&opt_args))
            .expect("Failed to prepare remote descriptor list");

        // Test with out-of-bounds indices (should fail)
        let invalid_indices = [999i32];  // Index 999 doesn't exist
        let result = agent1.make_xfer_req(
            XferOp::Write,
            &local_handle,
            &invalid_indices,    // Out-of-bounds local index
            &remote_handle,
            &invalid_indices,    // Out-of-bounds remote index
            None
        );
        assert!(result.is_err_and(|e| matches!(e, NixlError::BackendError)), "Expected InvalidParam for out-of-bounds indices");
    }
}

// Tests for get_local_partial_md API
#[test]
fn test_get_local_partial_md_success() {
    let (agent, opt_args) = create_agent_with_backend("test_agent")
        .expect("Failed to setup agent with backend");
    let _storage_list = create_storage_list(&agent, &opt_args, 10);
    // Create a registration descriptor list
    let mut reg_descs = RegDescList::new(MemType::Dram)
        .expect("Failed to create registration descriptor list");
    reg_descs.add_desc(0x1000, 0x100, 0)
        .expect("Failed to add descriptor");
    // Get local partial metadata
    let result = agent.get_local_partial_md(&reg_descs, Some(&opt_args));
    // Should succeed and return metadata
    match result {
        Ok(metadata) => {
            assert!(!metadata.is_empty(), "Metadata should not be empty");
            println!("Partial metadata size: {}", metadata.len());
        }
        Err(e) => {
            // May fail if no partial metadata exists yet, which is acceptable
            assert!(
                matches!(e, NixlError::BackendError) || matches!(e, NixlError::InvalidParam),
                "Expected BackendError or InvalidParam, got: {:?}", e
            );
        }
    }
}

#[test]
fn test_get_local_partial_md_empty_descs() {
    let (agent, _) = create_agent_with_backend("test_agent")
        .expect("Failed to setup agent with backend");
    // Create empty registration descriptor list
    let reg_descs = RegDescList::new(MemType::Dram)
        .expect("Failed to create registration descriptor list");
    // Try with empty descriptor list should succeed and return all available backends
    let result = agent.get_local_partial_md(&reg_descs, None);
    assert!(
        result.is_ok(),
        "get_local_partial_md should succeed with empty descriptor list"
    );
}

// Tests for send_local_partial_md API
#[test]
fn test_send_local_partial_md_success() {
    let (agent, opt_args) = create_agent_with_backend("test_agent")
        .expect("Failed to setup agent with backend");
    let (agent2, opt_args2) = create_agent_with_backend("test_agent2")
        .expect("Failed to setup agent with backend");
    let _storage_list = create_storage_list(&agent, &opt_args, 10);
    let _remote_storage_list = create_storage_list(&agent2, &opt_args2, 10);

    // Create a registration descriptor list
    let mut reg_descs = RegDescList::new(MemType::Dram)
        .expect("Failed to create registration descriptor list");
    reg_descs.add_storage_desc(&_storage_list[0]).expect("Failed to add storage descriptor");

    // Send local partial metadata
    let mut opt_args_temp = OptArgs::new().expect("Failed to create opt args");
    opt_args_temp.set_ip_addr("127.0.0.1").expect("Failed to set ip address");
    let result = agent.send_local_partial_md(&reg_descs, Some(&opt_args_temp));

    assert!(
        result.is_ok(),
        "send_local_partial_md should succeed"
    );
}

// Tests for query_xfer_backend API
#[test]
fn test_query_xfer_backend_success() {
    let (agent1, opt_args) = create_agent_with_backend("agent1").expect("Failed to create agent");
    let (agent2, opt_args_remote) = create_agent_with_backend("agent2").expect("Failed to create agent");
    // Create descriptor lists
    let mut storage_list = create_storage_list(&agent1, &opt_args, 1);
    let mut remote_storage_list = create_storage_list(&agent2, &opt_args_remote, 1);
    {
        let local_dlist = create_dlist(&mut storage_list).expect("Failed to create descriptor list");
        let remote_dlist = create_dlist(&mut remote_storage_list).expect("Failed to create descriptor list");
        exchange_metadata(&agent1, &agent2).expect("Failed to exchange metadata");
        // Create transfer request
        let xfer_req = agent1.create_xfer_req(
            XferOp::Write,
            &local_dlist,
            &remote_dlist,
            "agent2",
            None
        ).expect("Failed to create transfer request");
        // Query which backend will be used for this transfer
        let result: Result<Backend, NixlError> = agent1.query_xfer_backend(&xfer_req);
        assert!(result.is_ok(), "query_xfer_backend failed with error: {:?}", result.err());
        let backend = result.unwrap();
        println!("Transfer will use backend: {:?}", backend);
   }
}
#[test]
fn test_query_xfer_backend_invalid_request() {
    let (agent1, opt_args) = create_agent_with_backend("agent1").expect("Failed to create agent");
    let (agent2, opt_args_remote) = create_agent_with_backend("agent2").expect("Failed to create agent");
    // Create descriptor lists
    let mut storage_list = create_storage_list(&agent1, &opt_args, 1);
    let mut remote_storage_list = create_storage_list(&agent2, &opt_args_remote, 1);
    {
        let local_dlist = create_dlist(&mut storage_list).expect("Failed to create descriptor list");
        let remote_dlist = create_dlist(&mut remote_storage_list).expect("Failed to create descriptor list");
        // Create transfer request with non-existent remote agent (should fail or succeed)
        let xfer_req_result = agent1.create_xfer_req(
            XferOp::Write,
            &local_dlist,
            &remote_dlist,
            "non_existent_agent",
            None
        );
        assert!(xfer_req_result.is_err(), "Transfer request creation should fail for non-existent agent");
 }
}

// Tests for get_xfer_telemetry API
#[test]
fn test_get_xfer_telemetry_success() {
    env::set_var("NIXL_TELEMETRY_ENABLE", "1");

    let (agent1, opt_args) = create_agent_with_backend("agent1").expect("Failed to create agent");
    let (agent2, opt_args_remote) = create_agent_with_backend("agent2").expect("Failed to create agent");

    let mut storage_list = create_storage_list(&agent1, &opt_args, 1);
    let mut remote_storage_list = create_storage_list(&agent2, &opt_args_remote, 1);

    {
        let local_dlist = create_dlist(&mut storage_list).expect("Failed to create descriptor list");
        let remote_dlist = create_dlist(&mut remote_storage_list).expect("Failed to create descriptor list");

        exchange_metadata(&agent1, &agent2).expect("Failed to exchange metadata");

        let xfer_req = agent1.create_xfer_req(
            XferOp::Write,
            &local_dlist,
            &remote_dlist,
            "agent2",
            None
        ).expect("Failed to create transfer request");

        let result = agent1.post_xfer_req(&xfer_req, Some(&opt_args));
        assert!(result.is_ok(), "post_xfer_req failed with error: {:?}", result.err());

        // Wait for transfer to complete
        loop {
            match agent1.get_xfer_status(&xfer_req) {
                Ok(XferStatus::Success) => break,
                Ok(XferStatus::InProgress) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                Err(e) => panic!("Failed to get transfer status: {:?}", e),
            }
        }

        let telemetry_result = xfer_req.get_telemetry();
        assert!(telemetry_result.is_ok(), "get_xfer_telemetry failed with error: {:?}", telemetry_result.err());

        let telemetry = telemetry_result.unwrap();
        assert!(telemetry.start_time_us > 0, "Start time should be greater than 0");
        assert!(telemetry.total_bytes > 0, "Total bytes should be greater than 0");
        assert!(telemetry.desc_count > 0, "Descriptor count should be greater than 0");

        // Test convenience methods
        let start_time = telemetry.start_time();
        assert!(start_time.as_micros() == telemetry.start_time_us as u128);

        let post_duration = telemetry.post_duration();
        assert!(post_duration.as_micros() == telemetry.post_duration_us as u128);

        let xfer_duration = telemetry.xfer_duration();
        assert!(xfer_duration.as_micros() == telemetry.xfer_duration_us as u128);

        let total_duration = telemetry.total_duration();
        assert!(total_duration.as_micros() == (telemetry.post_duration_us + telemetry.xfer_duration_us) as u128);

        // Test transfer rate calculation
        let rate = telemetry.transfer_rate_bps();
        if telemetry.xfer_duration_us > 0 {
            assert!(rate > 0.0, "Transfer rate should be positive when transfer duration > 0");
        }

        println!("Telemetry data: {:?}", telemetry);
        println!("Transfer rate: {:.2} MB/s", rate / 1_000_000.0);
    }
}

#[test]
fn test_get_xfer_telemetry_from_request() {
    env::set_var("NIXL_TELEMETRY_ENABLE", "1");

    let (agent1, opt_args) = create_agent_with_backend("agent1").expect("Failed to create agent");
    let (agent2, opt_args_remote) = create_agent_with_backend("agent2").expect("Failed to create agent");

    let mut storage_list = create_storage_list(&agent1, &opt_args, 1);
    let mut remote_storage_list = create_storage_list(&agent2, &opt_args_remote, 1);

    {
        let local_dlist = create_dlist(&mut storage_list).expect("Failed to create descriptor list");
        let remote_dlist = create_dlist(&mut remote_storage_list).expect("Failed to create descriptor list");

        exchange_metadata(&agent1, &agent2).expect("Failed to exchange metadata");

        let xfer_req = agent1.create_xfer_req(
            XferOp::Write,
            &local_dlist,
            &remote_dlist,
            "agent2",
            None
        ).expect("Failed to create transfer request");

        let result = agent1.post_xfer_req(&xfer_req, Some(&opt_args));
        assert!(result.is_ok(), "post_xfer_req failed with error: {:?}", result.err());

        // Wait for transfer to complete
        loop {
            match agent1.get_xfer_status(&xfer_req) {
                Ok(XferStatus::Success) => break,
                Ok(XferStatus::InProgress) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                Err(e) => panic!("Failed to get transfer status: {:?}", e),
            }
        }

        let telemetry_result = xfer_req.get_telemetry();
        assert!(telemetry_result.is_ok(), "get_telemetry from request failed with error: {:?}", telemetry_result.err());

        let telemetry = telemetry_result.unwrap();
        assert!(telemetry.start_time_us > 0, "Start time should be greater than 0");
        assert!(telemetry.total_bytes > 0, "Total bytes should be greater than 0");
        assert!(telemetry.desc_count > 0, "Descriptor count should be greater than 0");

        println!("Telemetry data from request: {:?}", telemetry);
    }
}

#[test]
fn test_get_xfer_telemetry_without_telemetry_enabled() {
    env::remove_var("NIXL_TELEMETRY_ENABLE");

    let (agent1, opt_args) = create_agent_with_backend("agent1").expect("Failed to create agent");
    let (agent2, opt_args_remote) = create_agent_with_backend("agent2").expect("Failed to create agent");

    // Create descriptor lists
    let mut storage_list = create_storage_list(&agent1, &opt_args, 1);
    let mut remote_storage_list = create_storage_list(&agent2, &opt_args_remote, 1);

    {
        let local_dlist = create_dlist(&mut storage_list).expect("Failed to create descriptor list");
        let remote_dlist = create_dlist(&mut remote_storage_list).expect("Failed to create descriptor list");

        exchange_metadata(&agent1, &agent2).expect("Failed to exchange metadata");

        let xfer_req = agent1.create_xfer_req(
            XferOp::Write,
            &local_dlist,
            &remote_dlist,
            "agent2",
            None
        ).expect("Failed to create transfer request");

        let result = agent1.post_xfer_req(&xfer_req, Some(&opt_args));
        assert!(result.is_ok(), "post_xfer_req failed with error: {:?}", result.err());

        // Wait for transfer to complete
        loop {
            match agent1.get_xfer_status(&xfer_req) {
                Ok(XferStatus::Success) => break,
                Ok(XferStatus::InProgress) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                Err(e) => panic!("Failed to get transfer status: {:?}", e),
            }
        }

        // Try to get telemetry data - should fail with NoTelemetry
        let telemetry_result = xfer_req.get_telemetry();
        assert!(telemetry_result.is_err(), "get_xfer_telemetry should fail when telemetry is disabled");

        match telemetry_result.err().unwrap() {
            NixlError::NoTelemetry => {
                println!("Correctly received NoTelemetry error");
            }
            other => panic!("Expected NoTelemetry error, got: {:?}", other),
        }
    }
}

#[test]
fn test_get_xfer_telemetry_before_posting() {
    env::set_var("NIXL_TELEMETRY_ENABLE", "1");

    let (agent1, opt_args) = create_agent_with_backend("agent1").expect("Failed to create agent");
    let (agent2, opt_args_remote) = create_agent_with_backend("agent2").expect("Failed to create agent");

    // Create descriptor lists
    let mut storage_list = create_storage_list(&agent1, &opt_args, 1);
    let mut remote_storage_list = create_storage_list(&agent2, &opt_args_remote, 1);

    {
        let local_dlist = create_dlist(&mut storage_list).expect("Failed to create descriptor list");
        let remote_dlist = create_dlist(&mut remote_storage_list).expect("Failed to create descriptor list");

        exchange_metadata(&agent1, &agent2).expect("Failed to exchange metadata");

        // Create transfer request
        let xfer_req = agent1.create_xfer_req(
            XferOp::Write,
            &local_dlist,
            &remote_dlist,
            "agent2",
            None
        ).expect("Failed to create transfer request");

        // Try to get telemetry before posting the request - should fail
        let telemetry_result = xfer_req.get_telemetry();
        assert!(telemetry_result.is_err(), "get_xfer_telemetry should fail before transfer is posted");
        let error = telemetry_result.err().unwrap();
        match error {
            NixlError::NoTelemetry | NixlError::BackendError => {
                println!("Got expected error before posting: {:?}", error);
            }
            other => panic!("Expected NoTelemetry or BackendError, got: {:?}", other),
        }

        println!("Successfully tested telemetry before posting - got expected error");
    }
}

// Tests for equality operators on RegDescList and XferDescList
#[test]
fn test_descriptor_list_equality() {
    // Test RegDescList equality scenarios
    {
        // 1. Empty lists of same type should be equal
        let reg_list1 = RegDescList::new(MemType::Dram).unwrap();
        let reg_list2 = RegDescList::new(MemType::Dram).unwrap();
        assert_eq!(reg_list1, reg_list2);
        assert!(!(reg_list1 != reg_list2));

        // 2. Lists with different memory types should not be equal
        let reg_list_vram = RegDescList::new(MemType::Vram).unwrap();
        assert_ne!(reg_list1, reg_list_vram);
        assert!(!(reg_list1 == reg_list_vram));

        // 3. Lists with same descriptors should be equal
        let mut reg_list3 = RegDescList::new(MemType::Dram).unwrap();
        let mut reg_list4 = RegDescList::new(MemType::Dram).unwrap();

        reg_list3.add_desc(0x1000, 0x100, 0).unwrap();
        reg_list3.add_desc(0x2000, 0x200, 1).unwrap();

        reg_list4.add_desc(0x1000, 0x100, 0).unwrap();
        reg_list4.add_desc(0x2000, 0x200, 1).unwrap();

        assert_eq!(reg_list3, reg_list4);
        assert!(!(reg_list3 != reg_list4));

        // 4. Lists with different descriptors should not be equal
        let mut reg_list5 = RegDescList::new(MemType::Dram).unwrap();
        let mut reg_list6 = RegDescList::new(MemType::Dram).unwrap();

        reg_list5.add_desc(0x1000, 0x100, 0).unwrap();
        reg_list6.add_desc(0x2000, 0x200, 1).unwrap();

        assert_ne!(reg_list5, reg_list6);
        assert!(!(reg_list5 == reg_list6));

        // 5. Lists with different lengths should not be equal
        let mut reg_list7 = RegDescList::new(MemType::Dram).unwrap();
        let reg_list8 = RegDescList::new(MemType::Dram).unwrap();

        reg_list7.add_desc(0x1000, 0x100, 0).unwrap();

        assert_ne!(reg_list7, reg_list8);
        assert!(!(reg_list7 == reg_list8));

        // 6. Lists with same descriptors but different order should not be equal
        let mut reg_list9 = RegDescList::new(MemType::Dram).unwrap();
        let mut reg_list10 = RegDescList::new(MemType::Dram).unwrap();

        reg_list9.add_desc(0x1000, 0x100, 0).unwrap();
        reg_list9.add_desc(0x2000, 0x200, 1).unwrap();

        reg_list10.add_desc(0x2000, 0x200, 1).unwrap();
        reg_list10.add_desc(0x1000, 0x100, 0).unwrap();

        assert_ne!(reg_list9, reg_list10);
        assert!(!(reg_list9 == reg_list10));

        // 7. Lists with same descriptors but different metadata should not be equal
        let mut reg_list11 = RegDescList::new(MemType::Dram).unwrap();
        let mut reg_list12 = RegDescList::new(MemType::Dram).unwrap();

        reg_list11.add_desc_with_meta(0x1000, 0x100, 0, b"metadata1").unwrap();
        reg_list12.add_desc_with_meta(0x1000, 0x100, 0, b"metadata2").unwrap();

        assert_ne!(reg_list11, reg_list12);
        assert!(!(reg_list11 == reg_list12));
    }

    // Test XferDescList equality scenarios (same tests as RegDescList)
    {
        // 1. Empty lists of same type should be equal
        let xfer_list1 = XferDescList::new(MemType::Dram).unwrap();
        let xfer_list2 = XferDescList::new(MemType::Dram).unwrap();
        assert_eq!(xfer_list1, xfer_list2);
        assert!(!(xfer_list1 != xfer_list2));

        // 2. Lists with different memory types should not be equal
        let xfer_list_vram = XferDescList::new(MemType::Vram).unwrap();
        assert_ne!(xfer_list1, xfer_list_vram);
        assert!(!(xfer_list1 == xfer_list_vram));

        // 3. Lists with same descriptors should be equal
        let mut xfer_list3 = XferDescList::new(MemType::Dram).unwrap();
        let mut xfer_list4 = XferDescList::new(MemType::Dram).unwrap();

        xfer_list3.add_desc(0x1000, 0x100, 0).unwrap();
        xfer_list3.add_desc(0x2000, 0x200, 1).unwrap();

        xfer_list4.add_desc(0x1000, 0x100, 0).unwrap();
        xfer_list4.add_desc(0x2000, 0x200, 1).unwrap();

        assert_eq!(xfer_list3, xfer_list4);
        assert!(!(xfer_list3 != xfer_list4));

        // 4. Lists with different descriptors should not be equal
        let mut xfer_list5 = XferDescList::new(MemType::Dram).unwrap();
        let mut xfer_list6 = XferDescList::new(MemType::Dram).unwrap();

        xfer_list5.add_desc(0x1000, 0x100, 0).unwrap();
        xfer_list6.add_desc(0x2000, 0x200, 1).unwrap();

        assert_ne!(xfer_list5, xfer_list6);
        assert!(!(xfer_list5 == xfer_list6));

        // 5. Lists with different lengths should not be equal
        let mut xfer_list7 = XferDescList::new(MemType::Dram).unwrap();
        let xfer_list8 = XferDescList::new(MemType::Dram).unwrap();

        xfer_list7.add_desc(0x1000, 0x100, 0).unwrap();

        assert_ne!(xfer_list7, xfer_list8);
        assert!(!(xfer_list7 == xfer_list8));

        // 6. Lists with same descriptors but different order should not be equal
        let mut xfer_list9 = XferDescList::new(MemType::Dram).unwrap();
        let mut xfer_list10 = XferDescList::new(MemType::Dram).unwrap();

        xfer_list9.add_desc(0x1000, 0x100, 0).unwrap();
        xfer_list9.add_desc(0x2000, 0x200, 1).unwrap();

        xfer_list10.add_desc(0x2000, 0x200, 1).unwrap();
        xfer_list10.add_desc(0x1000, 0x100, 0).unwrap();

        assert_ne!(xfer_list9, xfer_list10);
        assert!(!(xfer_list9 == xfer_list10));
    }
}
