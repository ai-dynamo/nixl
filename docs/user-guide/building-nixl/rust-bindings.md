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

```bash
cargo build --release
```

Install the NIXL C++ library (required by the Rust bindings):

```bash
ninja install
```

## Test

```bash
cargo test
```

## Usage

Add NIXL to your `Cargo.toml`:

```toml
[dependencies]
nixl-sys = { path = "path/to/nixl/bindings/rust" }
```

<Note>
For backend-specific build instructions, see [NIXL Backends](/docs/user-guide/backend-selection).
</Note>
