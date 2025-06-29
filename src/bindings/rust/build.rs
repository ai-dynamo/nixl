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
use os_info;

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

fn get_nixl_libs() -> Option<Vec<pkg_config::Library>> {
    // Try to get all libraries, but return None if any fails
    match (
        pkg_config::probe_library("nixl"),
        pkg_config::probe_library("nixl_build"),
        pkg_config::probe_library("nixl_common"),
        pkg_config::probe_library("stream"),
        pkg_config::probe_library("serdes"),
        pkg_config::probe_library("ucx_utils"),
        pkg_config::probe_library("etcd-cpp-api"),
    ) {
        (Ok(nixl), Ok(nixl_build), Ok(nixl_common), Ok(stream), Ok(serdes), Ok(ucx_utils), Ok(etcd)) => {
            Some(vec![nixl, nixl_build, nixl_common, stream, serdes, ucx_utils, etcd])
        }
        _ => None,
    }
}

fn main() {
    let nixl_root_path =
        env::var("NIXL_PREFIX").unwrap_or_else(|_| "/opt/nvidia/nvda_nixl".to_string());

    // Print the NIXL_PREFIX for debugging
    println!("cargo:warning=Using NIXL_PREFIX: {}", nixl_root_path);

    let nixl_include_path = format!("{}/include", nixl_root_path);
    let nixl_include_paths = [
        &nixl_include_path,
        "../../api/cpp",
        "../../infra",
        "../../core",
        "/usr/include",
    ];

    let arch = get_arch();
    let nixl_lib_path = get_lib_path(&nixl_root_path, &arch);

    // Print the library path for debugging
    println!("cargo:warning=Using library path: {}", nixl_lib_path);

    // Add all possible library paths
    println!("cargo:rustc-link-search=native={}", nixl_lib_path);
    println!("cargo:rustc-link-search=native={}/lib", nixl_root_path);
    println!("cargo:rustc-link-search=native={}/lib64", nixl_root_path);
    println!("cargo:rustc-link-search=native={}/lib/x86_64-linux-gnu", nixl_root_path);

    // Try to use pkg-config if available
    if let Some(libs) = get_nixl_libs() {
        println!("cargo:warning=Using pkg-config paths");
        for lib in libs {
            for path in lib.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
        }
    } else {
        println!("cargo:warning=pkg-config not available, using manual library paths");
    }

    // Check if etcd is enabled via environment variable
    let etcd_enabled = env::var("HAVE_ETCD").map(|v| v != "0").unwrap_or(false);

    // Build the C++ wrapper
    let mut cc_builder = cc::Build::new();
    cc_builder
        .cpp(true)
        .compiler("g++")
        .file("wrapper.cpp")
        .flag("-std=c++17")
        .flag("-fPIC")
        .includes(nixl_include_paths)
        .flag("-D_GLIBCXX_USE_CXX17_ABI=1")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-unused-variable")
        .flag("-pthread")
        .flag("-Wl,-rpath")
        .flag("-Wl,--no-as-needed")
        .compile("wrapper");

    // Define HAVE_ETCD if etcd is enabled
    if etcd_enabled {
        cc_builder.define("HAVE_ETCD", "1");
    }

    // Get the output path for bindings
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate bindings with minimal configuration
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-std=c++17")
        .clang_arg(format!("-I{}", nixl_include_path))
        .clang_arg("-I../../api/cpp")
        .clang_arg("-I../../infra")
        .clang_arg("-I../../core")
        .clang_arg("-x")
        .clang_arg("c++");

    // Add system include paths if needed
    if let Ok(cpp_include) = env::var("CPLUS_INCLUDE_PATH") {
        for path in cpp_include.split(':') {
            builder = builder.clang_arg(format!("-I{}", path));
        }
    }

    // Link against required libraries
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=dylib=stream");
    println!("cargo:rustc-link-lib=dylib=serdes");
    println!("cargo:rustc-link-lib=dylib=ucx_utils");

    // Add NIXL libraries
    println!("cargo:rustc-link-lib=dylib=nixl");
    println!("cargo:rustc-link-lib=dylib=nixl_build");
    println!("cargo:rustc-link-lib=dylib=nixl_common");

    if etcd_enabled {
        println!("cargo:rustc-link-lib=dylib=etcd-cpp-api");
    }

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rustc-link-search=native={}", nixl_lib_path);
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rerun-if-env-changed=HAVE_ETCD");

    builder
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
