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
#include <iostream>
#include "hf3fs_utils.h"
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include "common/status.h"
#include "common/nixl_log.h"


nixl_status_t hf3fsUtil::registerFileHandle(int fd, int *ret)
{
	int ret_val = hf3fs_reg_fd(fd, 0);
	if (ret_val > 0) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, absl::StrFormat("Error registering file descriptor %d, error: %d (errno: %d - %s)", fd, ret_val, errno, strerror(errno)));
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

nixl_status_t hf3fsUtil::wrapIOV(struct hf3fs_iov *iov, void *addr, size_t size, size_t block_size)
{   
    // Create a dummy ID - you might want a more sophisticated approach
    uint8_t dummy_id[16] = {0};
    
    auto ret = hf3fs_iovwrap(iov, addr, dummy_id, this->mount_point.c_str(), size, block_size, -1);
    
    if (ret < 0) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, absl::StrFormat("Error wrapping memory into IOV, error: %d (errno: %d - %s)", ret, errno, strerror(errno)));
    }
    
    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::createIOR(struct hf3fs_ior *ior, int num_ios, bool is_read)
{

    auto ret = hf3fs_iorcreate(ior, this->mount_point.c_str(), num_ios, is_read, 0, -1);
    if (ret < 0) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, absl::StrFormat("Error creating IOR, error: %d (errno: %d - %s)", ret, errno, strerror(errno)));
    }

    return NIXL_SUCCESS;
}

nixl_status_t validateIO(struct hf3fs_ior *ior, struct hf3fs_iov *iov, void *addr, size_t fd_offset, size_t size, int fd, bool is_read)
{      
    if (ior == nullptr) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: IOR is nullptr");
    }
    if (iov == nullptr) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: IOV is nullptr");
    }
    if (addr == nullptr) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: Address is nullptr");
    }
    
    // Check for valid size
    if (size == 0) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: Size cannot be zero");
    }
    
    // Get the memory range of the IOV
    uint8_t *iov_base = iov->base;
    size_t iov_size = iov->size;
    
    // Check if [addr, addr + size) is within [iov_base, iov_base + iov_size)
    if ((uint8_t*)addr < iov_base || (uint8_t*)addr + size > iov_base + iov_size) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, absl::StrFormat("Error: Memory range [%p, %p) is not within IOV range [%p, %p)", addr, (void*)((uint8_t*)addr + size), iov_base, (void*)(iov_base + iov_size)));
        return NIXL_ERR_INVALID_PARAM;
    }

    return NIXL_SUCCESS;
}

// TODO: better name?
nixl_status_t hf3fsUtil::prepIO(struct hf3fs_ior *ior, struct hf3fs_iov *iov, void *addr, size_t fd_offset, size_t size, int fd, bool is_read)
{
    if (validateIO(ior, iov, addr, fd_offset, size, fd, is_read) != NIXL_SUCCESS) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_INVALID_PARAM, "Error: Invalid IO parameters");
    }
    // Now call the prep_io function
    auto ret = hf3fs_prep_io(ior, iov, is_read, addr, fd, fd_offset, size, nullptr);
    if (ret < 0) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, absl::StrFormat("Error: hf3fs prep io error: %d (errno: %d - %s)", ret, errno, strerror(errno)));
    }
    
    return NIXL_SUCCESS;
}


nixl_status_t hf3fsUtil::postIOR(struct hf3fs_ior *ior)
{
    auto ret = hf3fs_submit_ios(ior);
    if (ret < 0) {
        NIXL_LOG_AND_RETURN_IF_ERROR(NIXL_ERR_BACKEND, absl::StrFormat("hf3fs submit ios error: %d (errno: %d - %s)", ret, errno, strerror(errno)));
    }  

    return NIXL_SUCCESS;
}

// TODO: probably need to chabge
nixl_status_t hf3fsUtil::checkXfer(struct hf3fs_ior *ior)
{
    
    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::destroyIOR(struct hf3fs_ior *ior)
{
    hf3fs_iordestroy(ior);
    return NIXL_SUCCESS;
}