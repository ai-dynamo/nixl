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

#[cfg(test)]
mod tests {
    use crate::*;

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

    #[test]
    fn test_get_backend_params() {
        let agent = Agent::new("test_agent").unwrap();
        let plugins = agent.get_available_plugins().unwrap();
        assert!(plugins.is_empty().unwrap_or(false) == false);

        let plugin_name = plugins.get(0).unwrap();
        let (_mems, params) = agent.get_plugin_params(&plugin_name).unwrap();
        let backend = agent.create_backend(&plugin_name, &params).unwrap();

        // Get backend params after initialization
        let (backend_mems, backend_params) = agent.get_backend_params(&backend).unwrap();

        // Verify we can access the parameters
        let param_iter = backend_params.iter().unwrap();
        for param in param_iter {
            let param = param.unwrap();
            println!("Backend param: {} = {}", param.key, param.value);
        }

        // Verify we can access the memory types
        for mem_type in backend_mems.iter() {
            println!("Backend memory type: {:?}", mem_type.unwrap());
        }
    }

    #[test]
    fn test_xfer_dlist() {
        let mut dlist = XferDescList::new(MemType::Dram).unwrap();

        // Add some descriptors
        dlist.add_desc(0x1000, 0x100, 0).unwrap();
        dlist.add_desc(0x2000, 0x200, 1).unwrap();

        // Check length
        assert_eq!(dlist.len().unwrap(), 2);

        // Check overlaps
        assert!(!dlist.has_overlaps().unwrap());

        // Add overlapping descriptor
        dlist.add_desc(0x1050, 0x100, 0).unwrap();
        assert!(dlist.has_overlaps().unwrap());

        // Clear list
        dlist.clear().unwrap();
        assert_eq!(dlist.len().unwrap(), 0);

        // Resize list
        dlist.resize(5).unwrap();

        // add descriptors with overlaps
        dlist.add_desc(0x1000, 0x100, 0).unwrap();
        dlist.add_desc(0x1050, 0x100, 0).unwrap();
        assert!(dlist.has_overlaps().unwrap());
    }

    #[test]
    fn test_reg_dlist() {
        let mut dlist = RegDescList::new(MemType::Dram).unwrap();

        // Add some descriptors
        dlist.add_desc(0x1000, 0x100, 0).unwrap();
        dlist.add_desc(0x2000, 0x200, 1).unwrap();

        // Check length
        assert_eq!(dlist.len().unwrap(), 2);

        // Check overlaps
        assert!(!dlist.has_overlaps().unwrap());

        // Add overlapping descriptor
        dlist.add_desc(0x1050, 0x100, 0).unwrap();
        assert!(dlist.has_overlaps().unwrap());

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

        let status = agent1.post_xfer_req(&xfer_req, None).unwrap();
        assert!(status);

        println!("Waiting for local completions");

        loop {
            let status = agent1.get_xfer_status(&xfer_req).unwrap();

            if status == false {
                println!("Xfer req completed");
                break;
            } else {
                println!("Xfer req not completed");
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        let mut notifs = NotificationMap::new().unwrap();
        println!("Waiting for notifications");
        std::thread::sleep(std::time::Duration::from_millis(100));

        // // see if the values in storage2 are 0xbb
        // assert!(storage2.as_slice().iter().all(|&x| x == 0xbb));

        loop {
            agent2.get_notifications(&mut notifs, None).unwrap();
            if !notifs.is_empty().unwrap() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        println!("Got notifications");

        // Get first notification from first agent
        let agent_name = notifs.agents().next().unwrap().unwrap();
        let notif = notifs
            .get_notifications(agent_name)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(notif, b"notification");

        // Verify memory patterns
        assert!(storage1.as_slice().iter().all(|&x| x == 0xbb));
        assert!(storage2.as_slice().iter().all(|&x| x == 0xbb));
    }
}
