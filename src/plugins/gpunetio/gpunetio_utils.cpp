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

#define ALIGN_SIZE(size, align) size = ((size + (align) - 1) / (align)) * (align);

void
nixlDocaEngineCheckCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        NIXL_ERROR << message << " (Error code: " << result << " - " << cudaGetErrorString(result)
                   << ")";
        exit(EXIT_FAILURE);
    }
}

void
nixlDocaEngineCheckCuError(CUresult result, const char *message) {
    const char *pStr;
    cuGetErrorString(result, &pStr);
    if (result != CUDA_SUCCESS) {
        NIXL_ERROR << message << " (Error code: " << result << " - " << pStr << ")";
        exit(EXIT_FAILURE);
    }
}

int
oob_connection_client_setup(const char *server_ip, int *oob_sock_fd) {
    struct sockaddr_in server_addr = {0};
    int oob_sock_fd_;

    /* Create socket */
    oob_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (oob_sock_fd_ < 0) {
        NIXL_ERROR << "Unable to create socket";
        return -1;
    }
    NIXL_INFO << "Socket created successfully";

    /* Set port and IP the same as server-side: */
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(DOCA_RDMA_CM_LOCAL_PORT_SERVER);
    server_addr.sin_addr.s_addr = inet_addr(server_ip);

    /* Send connection request to server: */
    if (connect(oob_sock_fd_, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        close(oob_sock_fd_);
        NIXL_ERROR << "Unable to connect to server at " << server_ip;
        return -1;
    }
    NIXL_INFO << "Connected with server successfully";

    *oob_sock_fd = oob_sock_fd_;
    return 0;
}

void
oob_connection_client_close(int oob_sock_fd) {
    if (oob_sock_fd > 0) close(oob_sock_fd);
}

void
oob_connection_server_close(int oob_sock_fd) {
    if (oob_sock_fd > 0) {
        shutdown(oob_sock_fd, SHUT_RDWR);
        close(oob_sock_fd);
    }
}

doca_verbs_context *
open_ib_device(char *name) {
    int nb_ibdevs = 0;
    struct ibv_device **ibdev_list = ibv_get_device_list(&nb_ibdevs);
    doca_verbs_context *context;

    if ((ibdev_list == NULL) || (nb_ibdevs == 0)) {
        NIXL_ERROR << "Failed to get RDMA devices list, ibdev_list null";
        return NULL;
    }

    for (int i = 0; i < nb_ibdevs; i++) {
        if (strncmp(ibv_get_device_name(ibdev_list[i]), name, strlen(name)) == 0) {
            struct ibv_device *dev_handle = ibdev_list[i];
            ibv_free_device_list(ibdev_list);

            if (doca_verbs_bridge_verbs_context_create(
                    dev_handle, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, &context) != DOCA_SUCCESS)
                return NULL;

            return context;
        }
    }

    ibv_free_device_list(ibdev_list);

    return NULL;
}

#define RC_QP_RST2INIT_REQ_ATTR_MASK                                         \
    (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | \
     DOCA_VERBS_QP_ATTR_ATOMIC_MODE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | \
     DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM)
#define INIT2INIT_REQ_ATTR_MASK (0)
#define RC_QP_INIT2RTR_REQ_ATTR_MASK                                                               \
    (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN | DOCA_VERBS_QP_ATTR_DEST_QP_NUM |  \
     DOCA_VERBS_QP_ATTR_PATH_MTU | DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER | \
     DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC)
#define RC_QP_RTR2RTS_REQ_ATTR_MASK                                                               \
    (DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN | DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | \
     DOCA_VERBS_QP_ATTR_RETRY_CNT | DOCA_VERBS_QP_ATTR_RNR_RETRY |                                \
     DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC)
#define RTS2RTS_REQ_ATTR_MASK (0)

doca_error_t
create_verbs_ah_attr(doca_verbs_context *verbs_context,
                     uint32_t gid_index,
                     enum doca_verbs_addr_type addr_type,
                     doca_verbs_ah_attr **verbs_ah_attr) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    doca_verbs_ah_attr *new_ah_attr = NULL;

    status = doca_verbs_ah_attr_create(verbs_context, &new_ah_attr);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create doca verbs ah attributes: %s", doca_error_get_descr(status);
        return status;
    }

    status = doca_verbs_ah_attr_set_addr_type(new_ah_attr, addr_type);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set address type: %s", doca_error_get_descr(status);
        goto destroy_verbs_ah;
    }

    status = doca_verbs_ah_attr_set_sgid_index(new_ah_attr, gid_index);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set sgid index: %s", doca_error_get_descr(status);
        goto destroy_verbs_ah;
    }

    if (addr_type == DOCA_VERBS_ADDR_TYPE_IPv4 || addr_type == DOCA_VERBS_ADDR_TYPE_IPv6) {
        status = doca_verbs_ah_attr_set_hop_limit(new_ah_attr, VERBS_TEST_HOP_LIMIT);
        if (status != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to set hop limit: %s", doca_error_get_descr(status);
            goto destroy_verbs_ah;
        }
    }

    *verbs_ah_attr = new_ah_attr;

    return DOCA_SUCCESS;

destroy_verbs_ah:
    tmp_status = doca_verbs_ah_attr_destroy(new_ah_attr);
    if (tmp_status != DOCA_SUCCESS)
        NIXL_ERROR << "Failed to destroy doca verbs AH: %s", doca_error_get_descr(tmp_status);

    return status;
}

doca_error_t
connect_verbs_qp(nixlDocaEngine *eng, doca_verbs_qp *qp, uint32_t rqpn, uint32_t remote_gid) {
    doca_error_t status = DOCA_SUCCESS, tmp_status = DOCA_SUCCESS;
    doca_verbs_qp_attr *verbs_qp_attr = NULL;

    status = doca_verbs_ah_attr_set_gid(eng->verbs_ah_attr, eng->remote_gid);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set remote gid " << doca_error_get_descr(status);
        return status;
    }

    // IB
    if (eng->port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        status = doca_verbs_ah_attr_set_dlid(eng->verbs_ah_attr, eng->dlid);
        if (status != DOCA_SUCCESS) {
            NIXL_ERROR << "Failed to set dlid";
            return status;
        }
    }

    status = doca_verbs_qp_attr_create(&verbs_qp_attr);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to create DOCA verbs QP attributes " << doca_error_get_descr(status);
        return status;
    }

    status = doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, DOCA_MTU_SIZE_4K_BYTES);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set path MTU " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set RQ PSN " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set SQ PSN " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_port_num(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set port number " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, 14);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set ACK timeout " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, 7);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set retry counter " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set RNR retry " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set minimum RNR timer " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set next state " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set allow remote write " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set allow remote read " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_atomic_mode(verbs_qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set atomic mode " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_max_rd_atomic(verbs_qp_attr, 16);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set max rd atomic " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_max_dest_rd_atomic(verbs_qp_attr, 16);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set max dest rd atomic " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, eng->verbs_ah_attr);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set address handle " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, rqpn);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set destination QP number " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_modify(qp, verbs_qp_attr, RC_QP_RST2INIT_REQ_ATTR_MASK);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to modify QP " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set next state " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_modify(qp, verbs_qp_attr, RC_QP_INIT2RTR_REQ_ATTR_MASK);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to modify QP " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to set next state " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_modify(qp, verbs_qp_attr, RC_QP_RTR2RTS_REQ_ATTR_MASK);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to modify QP " << doca_error_get_descr(status);
        goto destroy_verbs_qp_attr;
    }

    status = doca_verbs_qp_attr_destroy(verbs_qp_attr);
    if (status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to destroy DOCA verbs QP attributes: %s",
            doca_error_get_descr(status);
        return status;
    }

    NIXL_INFO << "QP has been successfully connected and ready to use";

    return DOCA_SUCCESS;

destroy_verbs_qp_attr:
    tmp_status = doca_verbs_qp_attr_destroy(verbs_qp_attr);
    if (tmp_status != DOCA_SUCCESS) {
        NIXL_ERROR << "Failed to destroy DOCA verbs QP attributes: %s",
            doca_error_get_descr(tmp_status);
    }

    return status;
}

void *
threadProgressFunc(void *arg) {
    using namespace nixlTime;
    struct sockaddr_in client_addr = {0};
    unsigned int client_size = 0;
    int oob_sock_client;
    std::string remote_agent;

    nixlDocaEngine *eng = (nixlDocaEngine *)arg;
    while (ACCESS_ONCE(*eng->pthrStop) == 0) {
        /* Accept an incoming connection: */
        client_size = sizeof(client_addr);
        NIXL_ERROR << "Waiting on socket server ";
        oob_sock_client =
            accept(eng->oob_sock_server, (struct sockaddr *)&client_addr, &client_size);
        if (oob_sock_client < 0) {
            NIXL_ERROR << "Can't accept new socket connection " << oob_sock_client;
            if (ACCESS_ONCE(*eng->pthrStop) == 0)
                NIXL_ERROR << "Can't accept new socket connection " << oob_sock_client;
            // close(eng->oob_sock_server);
            return nullptr;
        }

        if (ACCESS_ONCE(*eng->pthrStop) == 1) {
            NIXL_DEBUG << "Stopping thread " << oob_sock_client;
            return nullptr;
        }

        NIXL_INFO << "Server: client connected at IP: " << inet_ntoa(client_addr.sin_addr)
                  << " and port: " << ntohs(client_addr.sin_port) << std::endl;

        // cuCtxSetCurrent(eng->main_cuda_ctx);

        eng->recvRemoteAgentName(oob_sock_client, remote_agent);

        eng->addRdmaQp(remote_agent);
        eng->nixlDocaInitNotif(remote_agent, eng->ddev, eng->gdevs[0].second);
        eng->connectServerRdmaQp(oob_sock_client, remote_agent);
        close(oob_sock_client);

        /* Wait for predefined number of */
        // auto start = nixlTime::getUs();
        // while ((start + connection_delay.count()) > nixlTime::getUs()) {
        //     std::this_thread::yield();
        // }
    }

    return nullptr;
}

static int
sockaddr_sizeof(const struct sockaddr *addr, size_t *size_p) {
    switch (addr->sa_family) {
    case AF_INET:
        *size_p = sizeof(struct sockaddr_in);
        return 0;
    case AF_INET6:
        *size_p = sizeof(struct sockaddr_in6);
        return 0;
    default:
        NIXL_ERROR << "unknown address family: " << addr->sa_family;
        return -1;
    }
}

static int
netif_flags_is_active(unsigned int flags) {
    return (flags & IFF_UP) && (flags & IFF_RUNNING);
}

int
netif_get_addr(const char *if_name,
               sa_family_t af,
               struct sockaddr *saddr,
               struct sockaddr *netmask) {
    int status = 0;
    struct ifaddrs *ifa;
    struct ifaddrs *ifaddrs;
    const struct sockaddr_in6 *saddr6;
    size_t addrlen;

    if (getifaddrs(&ifaddrs)) {
        NIXL_ERROR << "getifaddrs error";
        return -1;
    }

    for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
        if ((if_name != NULL) && (0 != strcmp(if_name, ifa->ifa_name))) {
            continue;
        }

        if ((ifa->ifa_addr == NULL) ||
            ((ifa->ifa_addr->sa_family != AF_INET) && (ifa->ifa_addr->sa_family != AF_INET6))) {
            continue;
        }

        if (!netif_flags_is_active(ifa->ifa_flags)) {
            continue;
        }

        if (ifa->ifa_addr->sa_family == AF_INET6) {
            saddr6 = (const struct sockaddr_in6 *)ifa->ifa_addr;
            if (IN6_IS_ADDR_LINKLOCAL(&saddr6->sin6_addr)) {
                continue;
            }
        }

        if ((af == AF_UNSPEC) || (ifa->ifa_addr->sa_family == af)) {
            status = sockaddr_sizeof(ifa->ifa_addr, &addrlen);
            if (status != 0) {
                goto out_free_ifaddr;
            }

            if (saddr != NULL) {
                memcpy(saddr, ifa->ifa_addr, addrlen);
            }

            if (netmask != NULL) {
                memcpy(netmask, ifa->ifa_netmask, addrlen);
            }

            status = 0;
            break;
        }
    }

out_free_ifaddr:
    freeifaddrs(ifaddrs);
    return status;
}
