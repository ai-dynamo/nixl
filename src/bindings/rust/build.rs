use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("OUT_DIR = {:?}", out_dir);

    // Link against nixl libraries
    println!("cargo:rustc-link-search=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=nixl");
    println!("cargo:rustc-link-lib=dylib=nixl_build");
    println!("cargo:rustc-link-lib=dylib=serdes");

    // Build the C++ wrapper
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .warnings(true)
        .extra_warnings(true)
        .flag("-std=c++17")
        .flag("-fPIC")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-unused-variable")
        .include("../../api/cpp")  // Add path to nixl.h
        .include("../../core")     // Add path to other headers
        .include("../../infra")    // Add path to other headers
        .file("wrapper.cpp")
        .compile("wrapper");

    // Generate Rust bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-I../../api/cpp")  // Add path to nixl.h
        .clang_arg("-I../../core")     // Add path to other headers
        .clang_arg("-I../../infra")    // Add path to other headers
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
