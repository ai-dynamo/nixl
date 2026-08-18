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


//! Helpers shared by the integration test binaries.
#![allow(dead_code)]

use nixl_sys::*;

pub fn create_agent_with_backend(name: &str) -> Result<(Agent, OptArgs), NixlError> {
    let agent = Agent::new(name).expect("Failed to create agent");
    let plugins = agent.get_available_plugins().expect("Failed to get available plugins");
    let plugin_name = find_plugin(&plugins, "UCX").expect("Failed to find plugin");
    let (_mems, params) = agent.get_plugin_params(&plugin_name).expect("Failed to get plugin params");
    agent.create_backend(&plugin_name, &params).expect("Failed to create backend");

    let mut opt_args = OptArgs::new().expect("Failed to create opt args");
    let _ = opt_args.add_backend(&agent.get_backend("UCX").unwrap());

    Ok((agent, opt_args))
}

pub fn exchange_metadata(agent1: &Agent, agent2: &Agent) -> Result<(), NixlError> {
    let metadata1 = agent1.get_local_md().expect("Failed to get local metadata");
    let metadata2 = agent2.get_local_md().expect("Failed to get local metadata");
    agent1.load_remote_md(&metadata2).expect("Failed to load remote metadata");
    agent2.load_remote_md(&metadata1).expect("Failed to load remote metadata");
    Ok(())
}

// Helper function to find a plugin by name
pub fn find_plugin(plugins: &StringList, name: &str) -> Result<String, NixlError> {
    plugins
        .iter()
        .filter_map(Result::ok)
        .find(|&plugin| plugin == name)
        .map(ToString::to_string)
        .or_else(|| plugins.get(0).ok().map(ToString::to_string))
        .ok_or(NixlError::InvalidParam)
}
