---
title: Rust Bindings
description: Build NIXL Rust bindings from source using Meson or Cargo.
---

## Via Meson

Add the `-Drust=true` flag during Meson setup:

```bash
meson setup <name_of_build_dir> -Drust=true
cd <name_of_build_dir>
ninja
ninja install
```

## Manual Build

Build and install the NIXL C++ library before compiling the Rust crate:

```bash
meson setup build
ninja -C build
ninja -C build install
cargo build --manifest-path src/bindings/rust/Cargo.toml --release
```

## Test

```bash
cargo test
```

## Usage

Add NIXL to your `Cargo.toml`:

```toml
[dependencies]
nixl-sys = { path = "path/to/nixl/src/bindings/rust" }
```

<Note>
For backend-specific build instructions, see [NIXL Backends](/nixl/user-guide/backend-selection).
</Note>
