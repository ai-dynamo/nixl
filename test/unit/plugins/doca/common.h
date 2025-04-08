/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DOCA_UNIT_H
#define DOCA_UNIT_H

#include <iostream>
#include <string>
#include <algorithm>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <cstring>
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"

#define NUM_TRANSFERS 1
#define SIZE 1024

#if __cplusplus
extern "C" {
#endif

int launch_warmup_kernel(cudaStream_t stream, uintptr_t addr, size_t size);
int launch_target_wait_kernel(cudaStream_t stream, uintptr_t addr, size_t size);
int launch_initiator_send_kernel(cudaStream_t stream, uintptr_t addr, size_t size);

#if __cplusplus
}
#endif

#endif
