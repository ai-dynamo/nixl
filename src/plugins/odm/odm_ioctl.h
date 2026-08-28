/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * ODM dma-buf ioctls. Struct layouts and magic must match the ODM kernel driver
 * (mrvl_cxl_core_char_dev.h). The plugin exports GPU VRAM as a dma-buf and
 * hands the fd to the kernel, which imports it and drives the ODM DMA engine.
 */
#ifndef NIXL_SRC_PLUGINS_ODM_ODM_IOCTL_H
#define NIXL_SRC_PLUGINS_ODM_ODM_IOCTL_H

#include <stdint.h>
#include <sys/ioctl.h>

/* Magic and command numbers must match kernel mrvl_cxl_core_char_dev.h */
#define ODM_IOCTL_MAGIC 0xCE

/*
 * dma-buf transfer command (same layout as kernel struct
 * mrvl_dma_xfer_commands_fd). Zero-initialize, set fields, then ioctl.
 */
struct mrvl_dma_xfer_commands_fd {
    uint32_t dmabuf_fd;        /* exported GPU VRAM dma-buf file descriptor */
    uint64_t target_iova_addr; /* Iliad/Structera device-local IOVA */
    uint32_t tranfer_size;
    uint32_t tranfer_type;     /* ODM_XTYPE_INBOUND or ODM_XTYPE_OUTBOUND */
    uint16_t qid;
};

/* GPU dma-buf -> Iliad device memory (write into device). */
#define MRVL_CXL_DMA_WRITE_COMMAND_FD _IOWR(ODM_IOCTL_MAGIC, 10, struct mrvl_dma_xfer_commands_fd)
/* Iliad device memory -> GPU dma-buf (read from device). */
#define MRVL_CXL_DMA_READ_COMMAND_FD  _IOWR(ODM_IOCTL_MAGIC, 13, struct mrvl_dma_xfer_commands_fd)

#define ODM_XTYPE_OUTBOUND 0  /* device -> host/GPU */
#define ODM_XTYPE_INBOUND  1  /* host/GPU -> device */

/* ODM transfer direction (relative to the GPU). */
#define ODM_DIR_TO_GPU   0    /* Iliad device -> GPU VRAM (READ_FD)  */
#define ODM_DIR_FROM_GPU 1    /* GPU VRAM -> Iliad device (WRITE_FD) */

#endif /* NIXL_SRC_PLUGINS_ODM_ODM_IOCTL_H */
