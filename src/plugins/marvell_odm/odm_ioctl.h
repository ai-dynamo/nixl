/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Marvell Technology, Inc. All rights reserved.
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
 * ODM dma-buf ioctls. Struct layouts and magic must match the ODM kernel driver.
 * The plugin exports GPU VRAM as a dma-buf and
 * hands the fd to the kernel, which imports it and drives the ODM DMA engine.
 */
#ifndef NIXL_SRC_PLUGINS_MARVELL_ODM_ODM_IOCTL_H
#define NIXL_SRC_PLUGINS_MARVELL_ODM_ODM_IOCTL_H

#include <stdint.h>
#include <sys/ioctl.h>

/* Magic and command numbers must match the ODM kernel driver. */
#define ODM_IOCTL_MAGIC 0xCE

/*
 * dma-buf transfer command (same layout as kernel struct
 * mrvl_dma_xfer_commands_fd). Zero-initialize, set fields, then ioctl.
 */
struct mrvl_dma_xfer_commands_fd {
    uint32_t dmabuf_fd; /* exported GPU VRAM dma-buf file descriptor */
    uint64_t target_iova_addr; /* ODM device-local IOVA */
    uint32_t tranfer_size;
    uint32_t tranfer_type; /* ODM_XTYPE_INBOUND or ODM_XTYPE_OUTBOUND */
    uint16_t qid;
};

/* GPU dma-buf -> Structera device memory (write into device). */
#define MRVL_CXL_DMA_WRITE_COMMAND_FD _IOWR(ODM_IOCTL_MAGIC, 10, struct mrvl_dma_xfer_commands_fd)
/* Structera device memory -> GPU dma-buf (read from device). */
#define MRVL_CXL_DMA_READ_COMMAND_FD _IOWR(ODM_IOCTL_MAGIC, 13, struct mrvl_dma_xfer_commands_fd)

/*
 * io_uring spray command: kernel fans one dma-buf transfer across qid_start..qid_end.
 * Must match the ODM kernel driver.
 */
struct mrvl_dma_xfer_commands_fd_spray {
    uint32_t dmabuf_fd;
    uint64_t target_iova_addr;
    uint32_t tranfer_size;
    uint32_t tranfer_type;
    uint16_t qid_start;
    uint16_t qid_end;
};

#define ODM_URING_CMD_WRITE_FD 10
#define ODM_URING_CMD_READ_FD 13

#define ODM_XTYPE_OUTBOUND 0 /* device -> host/GPU */
#define ODM_XTYPE_INBOUND 1 /* host/GPU -> device */

/*
 * Host-VA and IOVA mailbox ioctls (same layout as the ODM kernel driver).
 * Used by nixlbench consistency checks and tests.
 */
struct mrvl_dma_xfer_commands {
    uint64_t host_va_addr;
    uint64_t target_iova_addr;
    uint32_t tranfer_size;
    uint32_t tranfer_type;
    uint16_t qid;
};

struct mrvl_dma_iova_commands {
    uint64_t target_iova_addr;
    uint64_t target_iova_size;
};

#define MRVL_CXL_DMA_READ_COMMAND _IOWR(ODM_IOCTL_MAGIC, 3, struct mrvl_dma_xfer_commands)
#define MRVL_CXL_DMA_WRITE_COMMAND _IOWR(ODM_IOCTL_MAGIC, 4, struct mrvl_dma_xfer_commands)
#define MRVL_CXL_GET_IOVA_COMMAND _IOWR(ODM_IOCTL_MAGIC, 5, struct mrvl_dma_iova_commands)
#define MRVL_CXL_FREE_IOVA_COMMAND _IOWR(ODM_IOCTL_MAGIC, 6, struct mrvl_dma_iova_commands)

/* ODM transfer direction (from the device's perspective). */
#define ODM_DIR_OUTBOUND 0 /* device -> host DRAM or GPU VRAM (READ / READ_FD) */
#define ODM_DIR_INBOUND 1 /* host DRAM or GPU VRAM -> device (WRITE / WRITE_FD) */

#endif /* NIXL_SRC_PLUGINS_MARVELL_ODM_ODM_IOCTL_H */
