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

NVIDIA Inference Xfer Library (NIXL) is targeted for accelerating point to point communications in AI inference frameworks such as NVIDIA Dynamo, while providing an abstraction over various types of memory (e.g., CPU and GPU) and storage (e.g., file, block and object store) through a modular plug-in architecture.

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

[GDRCopy](https://github.com/NVIDIA/gdrcopy) is available on Github and is necessary for maximum performance, but UCX and NIXL will work without it.

```
$ wget https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
$ tar xzf ucx-1.18.0.tar.gz
$ cd ucx-1.18.0
$ ./autogen.sh
$ ./configure     					   \
    --enable-shared             	   \
    --disable-static            	   \
    --disable-doxygen-doc       	   \
    --enable-optimizations      	   \
    --enable-cma                	   \
    --enable-devel-headers      	   \
    --with-cuda=<cuda install>  	   \
    --with-verbs               	 	   \
    --with-dm                   	   \
    --with-gdrcopy=<gdrcopy install>   \
    --enable-mt
$ make -j
$ make -j install-strip
$ ldconfig
```

## Getting started
### Build & install

```
$ meson setup <name_of_build_dir>
$ cd <name_of_build_dir>
$ ninja
$ ninja install
```

### pybind11 Python Interface
The pybind11 bindings for the public facing NIXL API are available in src/bindings/python. These bindings implement the headers in the src/api/cpp directory.

The preferred way is to build it through meson-python, which will just let it be installed with pip. This can be done from the root nixl directory:

` $ pip install .`

### Building Docker container
To build the docker container, first clone the current repository. Also make sure you are able to pull docker images to your machine before attempting to build the container.

Run the following from the root folder of the cloned NIXL repository:
```
$ ./contrib/build-container.sh
```

By default, the container is built with Ubuntu 24.04. To build a container for Ubuntu 22.04 use the --os option as follows:
```
$ ./contrib/build-container.sh --os 22.04
```

To see all the options supported by the container use:
```
$ ./contrib/build-container.sh -h
```

The container also includes a prebuilt python wheel in /workspace/dist if required for installing/distributing. Also, the wheel can be built with a separate script (see below).

### Building the python wheel
The contrib folder also includes a script to build the python wheel with the UCX dependencies. Note, that UCX and other NIXL dependencies are required to be installed.
```
$ ./contrib/build-wheel.sh
```

## Examples

* [C++ examples](https://github.com/ai-dynamo/nixl/tree/main/examples/cpp)

* [Python examples](https://github.com/ai-dynamo/nixl/tree/main/examples/python)
