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
#include "hf3fs_utils.h"
#include "hf3fs_log.h"
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include "common/nixl_log.h"


nixl_status_t hf3fsUtil::registerFileHandle(int fd, int *ret)
{
	int ret_val = hf3fs_reg_fd(fd, 0);
	if (ret_val > 0) {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND,
            absl::StrFormat("Error registering file descriptor %d, error: %d (errno: %d - %s)",
                            fd, ret_val, errno, nixl_strerror(errno)));
	}
	*ret = ret_val;
	return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::openHf3fsDriver()
{
    return NIXL_SUCCESS;
}

void hf3fsUtil::closeHf3fsDriver()
{
    // nothing to do
}

void hf3fsUtil::deregisterFileHandle(int fd)
{
    hf3fs_dereg_fd(fd);
}

nixl_status_t
hf3fsUtil::wrapIOV(struct hf3fs_iov *iov,
                   void *addr,
                   size_t size,
                   size_t block_size,
                   const uint8_t *id) {
    auto ret = hf3fs_iovwrap(iov, addr, id, this->mount_point.c_str(), size, block_size, -1);

    if (ret < 0) {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND,
            absl::StrFormat("Error wrapping memory into IOV, error: %d (errno: %d - %s)",
                           ret, errno, nixl_strerror(errno)));
    }

    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::createIOR(struct hf3fs_ior *ior, int num_ios, bool is_read)
{
    auto ret = hf3fs_iorcreate(ior, this->mount_point.c_str(), num_ios, is_read, 0, -1);
    if (ret < 0) {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND,
            absl::StrFormat("Error creating IOR, error: %d (errno: %d - %s)",
                           ret, errno, nixl_strerror(errno)));
    }

    return NIXL_SUCCESS;
}

nixl_status_t
hf3fsUtil::createIOV(struct hf3fs_iov *iov, size_t size, size_t block_size) {
    auto ret = hf3fs_iovcreate(iov, this->mount_point.c_str(), size, block_size, -1);
    if (ret < 0) {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND,
            absl::StrFormat("Error creating IOV, error: %d (errno: %d - %s)",
                           ret, errno, nixl_strerror(errno)));
    }

    return NIXL_SUCCESS;
}

void hf3fsUtil::destroyIOV(struct hf3fs_iov *iov)
{
    hf3fs_iovdestroy(iov);
}

nixl_status_t validateIO(struct hf3fs_ior *ior, struct hf3fs_iov *iov, void *addr, size_t fd_offset,
                         size_t size, int fd, bool is_read)
{
    if (ior == nullptr) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM, "Error: IOR is nullptr");
    }
    if (iov == nullptr) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM, "Error: IOV is nullptr");
    }
    if (addr == nullptr) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM, "Error: Address is nullptr");
    }

    // Check for valid size
    if (size == 0) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM, "Error: Size cannot be zero");
    }

    // Get the memory range of the IOV
    uint8_t *iov_base = iov->base;
    size_t iov_size = iov->size;

    // Check if [addr, addr + size) is within [iov_base, iov_base + iov_size)
    if ((uint8_t*)addr < iov_base || (uint8_t*)addr + size > iov_base + iov_size) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM,
            absl::StrFormat("Error: Memory range [%p, %p) is not within IOV range [%p, %p)",
                            addr, (void*)((uint8_t*)addr + size), iov_base,
                            (void*)(iov_base + iov_size)));
    }

    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::prepIO(struct hf3fs_ior *ior, struct hf3fs_iov *iov, void *addr,
                                size_t fd_offset, size_t size, int fd, bool is_read,
                                void *user_data)
{
    if (validateIO(ior, iov, addr, fd_offset, size, fd, is_read) != NIXL_SUCCESS) {
        HF3FS_LOG_RETURN(NIXL_ERR_INVALID_PARAM, "Error: Invalid IO parameters");
    }
    // Now call the prep_io function
    auto ret = hf3fs_prep_io(ior, iov, is_read, addr, fd, fd_offset, size, user_data);
    if (ret < 0) {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND,
            absl::StrFormat("Error: hf3fs prep io error: %d (errno: %d - %s)",
                           ret, errno, nixl_strerror(errno)));
    }

    return NIXL_SUCCESS;
}


nixl_status_t hf3fsUtil::postIOR(struct hf3fs_ior *ior)
{
    auto ret = hf3fs_submit_ios(ior);
    if (ret < 0) {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND,
            absl::StrFormat("hf3fs submit ios error: %d (errno: %d - %s)",
                           ret, errno, nixl_strerror(errno)));
    }

    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::destroyIOR(struct hf3fs_ior *ior)
{
    hf3fs_iordestroy(ior);
    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::waitForIOs(struct hf3fs_ior *ior, struct hf3fs_cqe *cqes, int num_cqes,
                                    int min_cqes, struct timespec *ts, int *num_completed)
{
    auto ret = hf3fs_wait_for_ios(ior, cqes, num_cqes, min_cqes, ts);
    if (ret < 0 && ret != -ETIMEDOUT && ret != -EAGAIN) {
        HF3FS_LOG_RETURN(NIXL_ERR_BACKEND,
            absl::StrFormat("Error waiting for IOs: %d (errno: %d - %s)",
                           ret, errno, nixl_strerror(errno)));
    }

    *num_completed = ret;
    return NIXL_SUCCESS;
}
