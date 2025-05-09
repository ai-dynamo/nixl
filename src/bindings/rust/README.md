# Rust Bindings for NIXL

Rust bindings for the NVIDIA Inference Xfer Library (NIXL). These bindings provide a safe and idiomatic Rust interface to the NIXL C++ library.

## Prerequisites

### Install Rust and Cargo using [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Install the NIXL library on your system

Refer to the [NIXL README](https://github.com/ai-dynamo/nixl/blob/main/README.md) for instructions on how to install the NIXL library on your system.



## Building

The bindings can be built using Cargo, Rust's package manager:

```bash
# From the src/bindings/rust directory
cargo build
```

### Environment Variables

- `NIXL_PREFIX`: Path to the NIXL installation (default: `/opt/nvidia/nvda_nixl`)

## Documentation

The crate is documented using Rust's standard documentation system. You can generate and view the documentation with:

```bash
cargo doc
```


## Testing

The bindings include a comprehensive test suite that can be run with:

```bash
cargo test
```