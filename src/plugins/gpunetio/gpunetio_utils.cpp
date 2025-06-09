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

#include "gpunetio_backend.h"
#include "serdes/serdes.h"
#include <arpa/inet.h>
#include <stdexcept>
#include <unistd.h>
#include "common/nixl_log.h"
#include <chrono>

// constexpr auto connection_delay = 500ms;
constexpr std::chrono::microseconds connection_delay(500000);

nixlDocaMmap::nixlDocaMmap(void *addr,
                uint32_t elem_num,
                size_t elem_size,
                struct doca_dev *dev)
{
    doca_error_t result;
    if (addr == nullptr || elem_num == 0 || elem_size == 0 || dev == nullptr)
        throw std::invalid_argument("Invalid input values");

    result = doca_mmap_create(&mmap);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_mmap_create");

    result = doca_mmap_set_permissions(mmap,
                                        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE |
                                        DOCA_ACCESS_FLAG_RDMA_WRITE |
                                        DOCA_ACCESS_FLAG_PCI_RELAXED_ORDERING);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_mmap_set_permissions");

    result = doca_mmap_set_memrange(mmap, (void *)addr, (size_t)elem_num * elem_size);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_mmap_set_memrange");

    result = doca_mmap_add_dev(mmap, dev);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_mmap_add_dev");

    result = doca_mmap_start(mmap);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_mmap_start");
};

nixlDocaMmap::nixlDocaMmap() {}

nixlDocaMmap::~nixlDocaMmap() {
    doca_mmap_destroy(mmap);
};

nixlDocaBarr::nixlDocaBarr(struct doca_mmap *mmap,
                    uint32_t elem_num,
                    size_t elem_size,
                    struct doca_gpu *gpu)
{
    doca_error_t result;
    if (mmap == nullptr || elem_num == 0 || elem_size == 0 || gpu == nullptr)
        throw std::invalid_argument("Invalid input values");

    result = doca_buf_arr_create (elem_num, &barr);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_buf_arr_create");

    result = doca_buf_arr_set_params (barr, mmap, elem_size, 0);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_buf_arr_set_params");

    result = doca_buf_arr_set_target_gpu (barr, gpu);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_buf_arr_set_target_gpu");

    result = doca_buf_arr_start (barr);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_buf_arr_start");

    result = doca_buf_arr_get_gpu_handle (barr, &barr_gpu);
    if (result != DOCA_SUCCESS)
        throw std::invalid_argument("doca_buf_arr_get_gpu_handle");
};

nixlDocaBarr::~nixlDocaBarr() {
    doca_buf_arr_destroy(barr);
};

void
nixlDocaEngineCheckCudaError (cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << cudaGetErrorString (result)
                  << ")" << std::endl;
        exit (EXIT_FAILURE);
    }
}

void
nixlDocaEngineCheckCuError (CUresult result, const char *message) {
    const char* pStr;
    cuGetErrorString (result, &pStr);
    if (result != CUDA_SUCCESS) {
        std::cerr << message << " (Error code: " << result << " - " << pStr
                  << ")" << std::endl;
        exit (EXIT_FAILURE);
    }
}

int
oob_connection_client_setup (const char *server_ip, int *oob_sock_fd) {
    struct sockaddr_in server_addr = {0};
    int oob_sock_fd_;

    /* Create socket */
    oob_sock_fd_ = socket (AF_INET, SOCK_STREAM, 0);
    if (oob_sock_fd_ < 0) {
        NIXL_ERROR << "Unable to create socket";
        return -1;
    }
    NIXL_INFO << "Socket created successfully";

    /* Set port and IP the same as server-side: */
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons (DOCA_RDMA_CM_LOCAL_PORT_SERVER);
    server_addr.sin_addr.s_addr = inet_addr (server_ip);

    /* Send connection request to server: */
    if (connect (oob_sock_fd_, (struct sockaddr *)&server_addr, sizeof (server_addr)) < 0) {
        close (oob_sock_fd_);
        NIXL_ERROR << "Unable to connect to server at " << server_ip;
        return -1;
    }
    NIXL_INFO << "Connected with server successfully";

    *oob_sock_fd = oob_sock_fd_;
    return 0;
}

void
oob_connection_client_close (int oob_sock_fd) {
    if (oob_sock_fd > 0) close (oob_sock_fd);
}

void
oob_connection_server_close (int oob_sock_fd) {
    if (oob_sock_fd > 0) {
        shutdown (oob_sock_fd, SHUT_RDWR);
        close (oob_sock_fd);
    }
}

doca_error_t
open_doca_device_with_ibdev_name (const uint8_t *value, size_t val_size, struct doca_dev **retval) {
    struct doca_devinfo **dev_list;
    uint32_t nb_devs;
    char buf[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
    char val_copy[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
    doca_error_t res;
    size_t i;

    /* Set default return value */
    *retval = nullptr;

    /* Setup */
    if (val_size > DOCA_DEVINFO_IBDEV_NAME_SIZE) {
        NIXL_ERROR << "Value size too large. Failed to locate device";
        return DOCA_ERROR_INVALID_VALUE;
    }
    memcpy (val_copy, value, val_size);

    res = doca_devinfo_create_list (&dev_list, &nb_devs);
    if (res != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to load doca devices list. Doca_error value";
        return res;
    }

    /* Search */
    for (i = 0; i < nb_devs; i++) {
        res = doca_devinfo_get_ibdev_name (dev_list[i], buf, DOCA_DEVINFO_IBDEV_NAME_SIZE);
        if (res == DOCA_SUCCESS && strncmp (buf, val_copy, val_size) == 0) {
            /* If any special capabilities are needed */
            /* if device can be opened */
            res = doca_dev_open (dev_list[i], retval);
            if (res == DOCA_SUCCESS) {
                doca_devinfo_destroy_list (dev_list);
                return res;
            }
        }
    }

    NIXL_ERROR << "Matching device not found";

    res = DOCA_ERROR_NOT_FOUND;

    doca_devinfo_destroy_list (dev_list);
    return res;
}

void *
threadProgressFunc (void *arg) {
    using namespace nixlTime;
    struct sockaddr_in client_addr = {0};
    unsigned int client_size = 0;
    int oob_sock_client;
    std::string remote_agent;

    nixlDocaEngine *eng = (nixlDocaEngine *)arg;

    eng->pthrActive = 1;

    while (ACCESS_ONCE (eng->pthrStop) == 0) {
        /* Accept an incoming connection: */
        client_size = sizeof (client_addr);
        oob_sock_client =
                accept (eng->oob_sock_server, (struct sockaddr *)&client_addr, &client_size);
        if (oob_sock_client < 0) {
            if (ACCESS_ONCE (eng->pthrStop) == 0)
                NIXL_ERROR << "Can't accept new socket connection " << oob_sock_client;
            close (eng->oob_sock_server);
            return nullptr;
        }

        NIXL_INFO << "Client connected at IP: " << inet_ntoa (client_addr.sin_addr)
                  << " and port: " << ntohs (client_addr.sin_port);

        cuCtxSetCurrent (eng->main_cuda_ctx);

        eng->recvRemoteAgentName(oob_sock_client, remote_agent);

        NIXL_DEBUG << "recvRemoteAgentName remoteAgent " << remote_agent << std::endl;

        eng->addRdmaQp (remote_agent);
        eng->nixlDocaInitNotif (remote_agent, eng->ddev, eng->gdevs[0].second);
        eng->connectServerRdmaQp (oob_sock_client, remote_agent);

        close (oob_sock_client);
        /* Wait for predefined number of */
        auto start = nixlTime::getUs();
        while ((start + connection_delay.count()) > nixlTime::getUs()) {
            std::this_thread::yield();
        }
    }

    return nullptr;
}
