<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NVIDIA Inference Xfer Library (NIXL)

NVIDIA Inference Xfer Library (NIXL) is targeted for accelerating point to point communications in AI inference frameworks such as Dynamo, while providing an abstraction over various types of memory (e.g., CPU and GPU) and storage (e.g., file, block and object store) through a modular plug-in architecture.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/nixl)](https://github.com/ai-dynamo/nixl/releases/latest)

## Prerequisites
### Ubuntu:

`$ sudo apt install build-essential cmake pkg-config`

### Fedora:

`$ sudo dnf install gcc-c++ cmake pkg-config`

### Python

`$ pip3 install meson ninja pybind11`

### UCX

NIXL was tested with UCX version 1.18.0.

```
$ wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
$ tar xzf ucx-1.18.0.tar.gz
$ cd ucx-1.18.0
$ ./contrib/configure-release --prefix=<PATH_TO_INSTALL>/install
$ make -j8
$ make install
```

## Getting started
### Build & install

```
$ meson setup <name_of_build_dir>
$ cd <name_of_build_dir>
$ ninja
$ ninja-install
```

### pybind11 Python Interface
The pybind11 bindings for the public facing NIXL API are available in src/bindings/python. These bindings implement the headers in the src/api/cpp directory.

The Python module library (.so) will be built by default. To make it easy to import you can include either the build or install directory to your PYTHONPATH, see:

`$ export PYTHONPATH=$PYTHONPATH:<path to build>/src/pybind`

or

` $export PYTHONPATH=$PYTHONPATH:<path to install>/lib64/python3.9/site-packages/`

The preferred way is to build it through meson-python, which will just let it be installed with pip. This can be done from the root nixl directory:

` $pip install .`

## Examples

* [C++ examples](https://github.com/ai-dynamo/nixl/tree/main/test/nixl)

* [Python examples](https://github.com/ai-dynamo/nixl/tree/main/test/python)