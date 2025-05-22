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

use std::env;
use std::path::PathBuf;

fn get_lib_path(nixl_root_path: &str, arch: &str) -> String {
    let os_info = os_info::get();
    match os_info.os_type() {
        os_info::Type::Redhat
        | os_info::Type::RedHatEnterprise
        | os_info::Type::CentOS
        | os_info::Type::Fedora => {
            format!("{}/lib64", nixl_root_path)
        }
        os_info::Type::Ubuntu | os_info::Type::Debian => {
            format!("{}/lib/{}-linux-gnu", nixl_root_path, arch)
        }
        os_info::Type::Arch => {
            format!("{}/lib", nixl_root_path)
        }
        _ => {
            // For unknown distributions, try to detect the path dynamically
            let possible_paths = [
                format!("{}/lib64", nixl_root_path),
                format!("{}/lib/{}-linux-gnu", nixl_root_path, arch),
                format!("{}/lib", nixl_root_path),
            ];
            // Print a warning about unknown distribution
            println!(
                "cargo:warning=Unknown Linux distribution: {}. Trying common library paths.",
                os_info
            );
            // Return the first path that exists
            for path in possible_paths.iter() {
                if std::path::Path::new(path).exists() {
                    return path.clone();
                }
            }
            // If no path exists, default to lib64 as it's most common
            format!("{}/lib64", nixl_root_path)
        }
    }
}

fn get_arch() -> String {
    let os_info = os_info::get();
    match os_info.architecture().unwrap_or("x86_64").to_string() {
        arch if arch == "x86_64" => "x86_64".to_string(),
        arch if arch == "aarch64" || arch == "arm64" => "aarch64".to_string(),
        other => panic!("Unsupported architecture: {}", other),
    }
}

fn main() {
    let nixl_root_path =
        env::var("NIXL_PREFIX").unwrap_or_else(|_| "/opt/nvidia/nvda_nixl".to_string());
    let nixl_include_path = format!("{}/include", nixl_root_path);

    let arch = get_arch();
    let nixl_lib_path = get_lib_path(&nixl_root_path, &arch);

    // Check if etcd is enabled via environment variable
    let etcd_enabled = env::var("HAVE_ETCD").map(|v| v != "0").unwrap_or(false);

    // Tell cargo to look for shared libraries in the specified directories
    println!("cargo:rustc-link-search={}", nixl_lib_path);

    // Build the C++ wrapper
    let mut cc_builder = cc::Build::new();
    cc_builder
        .cpp(true)
        .compiler("g++") // Ensure we're using the C++ compiler
        .file("wrapper.cpp")
        .flag("-std=c++17")
        .flag("-fPIC")
        .include(&nixl_include_path)
        // Change ABI flag if necessary to match your precompiled libraries:
        //    .flag("-D_GLIBCXX_USE_CXX11_ABI=0")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-unused-variable");

    // Define HAVE_ETCD if etcd is enabled
    if etcd_enabled {
        cc_builder.define("HAVE_ETCD", "1");
    }

    cc_builder.compile("wrapper");

    // Link against NIXL libraries in correct order
    // Only link against etcd-cpp-api if it's enabled
    if etcd_enabled {
        println!("cargo:rustc-link-lib=dylib=etcd-cpp-api");
    }
    println!("cargo:rustc-link-lib=dylib=stream");
    println!("cargo:rustc-link-lib=dylib=nixl_common");
    println!("cargo:rustc-link-lib=dylib=nixl");
    println!("cargo:rustc-link-lib=dylib=nixl_build");
    println!("cargo:rustc-link-lib=dylib=serdes");

    // Link against C++ standard library
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rustc-link-search=native={}", nixl_lib_path);
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rerun-if-env-changed=HAVE_ETCD");

    // Get the output path for bindings
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate bindings
    bindgen::Builder::default()
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
