# NVIDIA Inference Xfer Library (NIXL)

NIXL is an abstraction library to abstract memory of heterogeneous devices, i.e., CPU, GPU, storage, and enable most efficient and low-latency communication among them, integrated with distributed inference servers such as Triton. This library will target distributed inference communication patterns to effectively transfer the KV cache in disaggregated LLM serving platform.

# Code style

* Lower camel style: Data types, Classes, Class Members, Member functions
* Snake style: function arguments, local variables

# Prerequisites
### Packages ###
```
Ubuntu: sudo apt install build-essential cmake pkg-config
Fedora: sudo dnf install gcc-c++ cmake pkg-config
```

### Python ###
```
pip3 install meson ninja pybind11
```

### UCX install ###

UCX version 1.18.0 was used for nixl backend testing. This is available from the latest
tarball download

```
$ wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
$ tar xzf ucx-1.18.0.tar.gz
$ cd ucx-1.18.0
$ ./contrib/configure-release --prefix=<PATH_TO_INSTALL>/install
$ make -j8
$ make install
```

## Getting started
To compile and use this library use:
```
meson setup <name_of_build_dir>
cd <name_of_build_dir>
ninja
```
## Contributing
Commits:

Please follow the following format for your commits:

```
<component>: <title>

Detailed description of the fix.

Signed-off-by: <real name> <<email address>>
```

Please make commits with proper description and try to combine commits to smaller features
as possible. Similar to the Linux Kernel development, each author provides a signed-off-by:
at the bottom of the commit to certify that they contributed to the commit. If more than one
author(s) sheperd the contribution, add their own attestation to the bottom of the commit also.

Example commit can be as shown below.

```
commit 067e922af48c0d9b45da507b5800c3951076c4e9
Author: Jane Doe <jane@daos.io>
Date:   Thu Jan 23 14:26:00 2024 +0800

    NIXL-001 include: Add new APIs

    Add awesome new APIs to the NIXL

    Signed-off-by: Jane Doe <jane@nixl.io>
    Signed-off-by: John Smith <jsmith@corp.com>
```

## pybind11 Python Interface

The pybind11 bindings for the public facing NIXL API are available in src/pybind. These bindings implement the headers in the include directory, but not include/internal. Some sample tests for how to use the NIXL Python module are available in the python directory.

The Python module will be built by default, to make it easy to import you can include either the build or install directory to your PYTHONPATH, see:

```
export PYTHONPATH=$PYTHONPATH:<path to build>/src/pybind
### or
export PYTHONPATH=$PYTHONPATH:<path to install>/lib64/python3.9/site-packages/
```

## License
TBD
