/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Amazon.com, Inc. and affiliates.
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

#ifndef __NEURON_H
#define __NEURON_H

#include <iostream>

/* Return the number of visible neuron cores, or -1 if neuron library
   is not available at runtime */
int
neuronCoreCount();

int
neuronMalloc(void **addr, size_t buffer_size, int devid = 0);
int
neuronFree(void *addr);

enum neuronMemcpyKind {
    neuronMemcpyHostToDevice,
    neuronMemcpyDeviceToHost,
};

int
neuronMemcpy(void *dest, const void *src, size_t count, neuronMemcpyKind kind);
int
neuronMemset(void *addr, int val, size_t count);

#define CHECK_NEURON_ERROR(result, message)                                                       \
    do {                                                                                          \
        if (result != 0) {                                                                        \
            std::cerr << "NEURON: " << message << " (Error code: " << result << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)


#endif
