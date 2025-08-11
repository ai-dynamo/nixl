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

#include "util.h"

#ifdef MARK_NSIGHT
#include <nvToolsExt.h>
#endif

void
NVTXRangePush(const char *name) {
#ifdef MARK_NSIGHT
    constexpr uint32_t colors[] = {
        0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff};
    constexpr int num_colors = sizeof(colors) / sizeof(uint32_t);

    static thread_local int color_id = 0;
    ++color_id;

    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = colors[color_id % num_colors];
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
#endif
}

void
NVTXRangePop() {
#ifdef MARK_NSIGHT
    nvtxRangePop();
#endif
}
