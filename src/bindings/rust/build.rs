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

fn main() {
    let nixl_root_path =
        env::var("NIXL_PREFIX").unwrap_or_else(|_| "/opt/nvidia/nvda_nixl".to_string());
    let nixl_include_path = format!("{}/include", nixl_root_path);
    let nixl_lib_path = format!("{}/lib/x86_64-linux-gnu", nixl_root_path);

    // Tell cargo to look for shared libraries in the specified directories
    println!("cargo:rustc-link-search={}", nixl_lib_path);

    // Build the C++ wrapper
    cc::Build::new()
        .cpp(true)
        .compiler("g++") // Ensure we're using the C++ compiler
        .file("wrapper.cpp")
        .flag("-std=c++17")
        .flag("-fPIC")
        .include(&nixl_include_path)
        // Change ABI flag if necessary to match your precompiled libraries:
        //    .flag("-D_GLIBCXX_USE_CXX11_ABI=0")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-unused-variable")
        .compile("wrapper");

    // Link against NIXL libraries in correct order
    println!("cargo:rustc-link-lib=dylib=nixl");
    println!("cargo:rustc-link-lib=dylib=nixl_build");
    println!("cargo:rustc-link-lib=dylib=serdes");

    // Link against C++ standard library
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");

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
