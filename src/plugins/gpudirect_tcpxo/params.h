/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2026 Google LLC
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

#ifndef GPUDIRECT_TCPXO_PARAMS_H_
#define GPUDIRECT_TCPXO_PARAMS_H_

#include <cstdint>

#include <string>

#include "absl/strings/string_view.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/dxs-client.h"
#else
#include "rxdm_dxs_stub.h"
#endif
#include "tcpxo_common.h"

namespace tcpxo {

/*
 * Timeout threshold to finish the data transfer request from isend/irecv. Timeout enforces when the
 * Test() shows in progress data transport after the timeout threshold. The plugin won't timeout if
 * the NCCL invokes Test() after the timeout threshold and the data transport has finished. If set
 * to 0, or FastrakPluginDisableTimekeeping is set, then no timeout is enforced.
 */
inline constexpr char kFastrakDataTransferTimeoutParamName[] = "FASTRAK_DATA_TRANSFER_TIMEOUT_MS";
/*
 * Initial threshold after which a warning message will be printed if a request is still pending.
 * Each event causes this timeout to get multiplied by 2 to avoid polluting the log.
 */
inline constexpr char kFastrakDataTransferSlownessParamName[] = "FASTRAK_DATA_TRANSFER_SLOWNESS_MS";
/*
 * Timeout threshold for the connect call of DXS Client listens. If set to 0, or
 * FastrakPluginDisableTimekeeping is set, then no timeout is enforced.
 */
inline constexpr char kFastrakDxsListenTimeoutParamName[] = "FASTRAK_DXS_LISTEN_TIMEOUT_MS";
/*
 * Timeout threshold for the connect call of GPU<->GPU connections. If set to 0, or
 * FastrakPluginDisableTimekeeping is set, then no timeout is enforced.
 */
inline constexpr char kFastrakPluginConnectTimeoutParamName[] = "FASTRAK_PLUGIN_CONNECT_TIMEOUT_MS";
/*
 * Timeout threshold for the accept call of GPU<->GPU connections. If set to 0, or
 * FastrakPluginDisableTimekeeping is set, then no timeout is enforced.
 */
inline constexpr char kFastrakPluginAcceptTimeoutParamName[] = "FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS";
/* Number of flows per NCCL-level connection. */
inline constexpr char kFastrakNumFlowsPerDxsConnectionParamName[] = "FASTRAK_NUM_FLOWS";
/* The network device to use for the control channel. */
inline constexpr char kFastrakNumControlChannelWorkersParamName[] =
    "FASTRAK_NUM_CONTROL_CHANNEL_WORKERS";
/* The network device to use for the control channel. */
inline constexpr char kFastrakCtrlDevParamName[] = "FASTRAK_CTRL_DEV";
/* How often to send heartbeats to peers. */
inline constexpr char kFastrakHeartbeatSendPeriodParamName[] = "FASTRAK_HEARTBEAT_SEND_PERIOD_MS";
/* How long to wait for a heartbeat before disconnecting a peer. */
inline constexpr char kFastrakHeartbeatTimeoutParamName[] = "FASTRAK_HEARTBEAT_TIMEOUT_MS";
/*
 * Maximum time (in seconds) to wait for RxDM to come online during init. If set to 0, then wait for
 * 1 hour until RxDM is ready.
 */
inline constexpr char kFastrakRxdmInitTimeoutParamName[] = "FASTRAK_RXDM_INIT_TIMEOUT_SEC";
/* Only use lo interface for loopback tests if set to be true */
inline constexpr char kFastrakLoopbackOnly[] = "FASTRAK_LOOPBACK_ONLY";
/* Network interface names to be used */
inline constexpr char kFastrakIfname[] = "FASTRAK_IFNAME";
/* Ensure FasTrak operates purely in an IPv4 or IPv6 env */
inline constexpr char kFastrakSocketFamily[] = "FASTRAK_SOCKET_FAMILY";
/* Restrict communication to specific interface(s) */
inline constexpr char kFastrakSocketIfname[] = "FASTRAK_SOCKET_IFNAME";
/*
 * Optionally specifies connection information for NCCL boostrap network, used by ranks to find each
 * other during communicator initialization. Can be used to manually set the rendezvous point,
 * typically an IP address and port, that the root rank (rank 0) will use to listen for connections
 * from other ranks
 */
inline constexpr char kFastrakCommId[] = "FASTRAK_COMM_ID";
/*
 * If enabled, requests LLCM for FasTrak control path communication. Enabled by default.
 */
inline constexpr char kFastrakUseLlcm[] = "FASTRAK_USE_LLCM";
/*
 * Specify the search path for LLCM PCIe devices. Typically this would be sysfs, but in situations
 * where sysfs is mounted read-only, these devices may be exposed in a different location.
 */
inline constexpr char kFastrakLlcmDeviceDirectory[] = "FASTRAK_LLCM_DEVICE_DIRECTORY";
/*
 * If disabled and Snap is used currently, Send sockets created by DXS clients will not be closed by
 * DXS when they are done. Disabled by default.
 */
inline constexpr char kFastrakCloseSendOnDone[] = "FASTRAK_CLOSE_SEND_ON_DONE";
/*
 * Number of GPUs that will be used per node.
 * Set to kMaxGpuDevices (8) as default for A3 Mega machines.
 */
inline constexpr char kNumGpusPerNode[] = "NUM_GPUS_PER_NODE";

struct IntegerParamDef {
    const absl::string_view env_var_name;
    const int64_t default_value;
    const int64_t min_value;
    const int64_t max_value;
    // We expect all our integer values to be non-negative, but we will process negative numbers
    uint64_t value;
};

struct StringParamDef {
    const absl::string_view env_var_name;
    const absl::string_view default_value;
    std::string value;
};

struct BoolParamDef {
    const absl::string_view env_var_name;
    const bool default_value;
    bool value;
};

struct Params {
    IntegerParamDef fastrak_data_transfer_timeout_ms;
    IntegerParamDef fastrak_data_transfer_slowness_ms;
    IntegerParamDef fastrak_dxs_listen_timeout_ms;
    IntegerParamDef fastrak_plugin_connect_timeout_ms;
    IntegerParamDef fastrak_plugin_accept_timeout_ms;
    IntegerParamDef fastrak_num_flows_per_dxs_connection;
    IntegerParamDef fastrak_num_control_channel_workers;
    IntegerParamDef fastrak_heartbeat_send_period_ms;
    IntegerParamDef fastrak_heartbeat_timeout_ms;
    StringParamDef fastrak_ctrl_dev;
    IntegerParamDef fastrak_rxdm_init_timeout_sec;
    BoolParamDef fastrak_loopback_only;
    StringParamDef fastrak_ifname;
    StringParamDef fastrak_socket_family;
    StringParamDef fastrak_socket_ifname;
    StringParamDef fastrak_comm_id;
    BoolParamDef fastrak_use_llcm;
    StringParamDef fastrak_llcm_device_directory;
    BoolParamDef fastrak_close_send_on_done;
    IntegerParamDef fastrak_num_gpus_per_node;
};

inline Params
GetUnsetParams() {
    return Params{
        .fastrak_data_transfer_timeout_ms = {kFastrakDataTransferTimeoutParamName,
                                             2 * 60 * 60 * 1000, // 2 hours
                                             0,
                                             24 * 60 * 60 * 1000}, // 1 day
        .fastrak_data_transfer_slowness_ms = {kFastrakDataTransferSlownessParamName,
                                              5 * 60 * 1000, // 5 minutes
                                              1,
                                              24 * 60 * 60 * 1000}, // 1 day
        .fastrak_dxs_listen_timeout_ms = {kFastrakDxsListenTimeoutParamName,
                                          1000, // 1 second
                                          10, // 10 ms
                                          2 * 1000}, // 2 seconds
        .fastrak_plugin_connect_timeout_ms = {kFastrakPluginConnectTimeoutParamName,
                                              5 * 60 * 1000, // 5 minutes
                                              0,
                                              24 * 60 * 60 * 1000}, // 1 day
        .fastrak_plugin_accept_timeout_ms = {kFastrakPluginAcceptTimeoutParamName,
                                             15 * 60 * 1000, // 15 minutes
                                             0,
                                             24 * 60 * 60 * 1000}, // 1 day
        .fastrak_num_flows_per_dxs_connection = {kFastrakNumFlowsPerDxsConnectionParamName,
                                                 2,
                                                 1,
                                                 kFastrakMaxNumFlowsPerDxsConn},
        .fastrak_num_control_channel_workers =
            {kFastrakNumControlChannelWorkersParamName,
             8,
             1,
             16}, // number of cores we leave free in version vectors
        .fastrak_heartbeat_send_period_ms = {kFastrakHeartbeatSendPeriodParamName,
                                             15000, // 15 seconds
                                             100,
                                             15 * 60 * 1000}, // 15 minutes
        .fastrak_heartbeat_timeout_ms = {kFastrakHeartbeatTimeoutParamName,
                                         60 * 1000, // 1 minute
                                         400,
                                         2 * 60 * 60 * 1000}, // 2 hours
        .fastrak_ctrl_dev = {kFastrakCtrlDevParamName, "eth0"},
        .fastrak_rxdm_init_timeout_sec = {kFastrakRxdmInitTimeoutParamName,
                                          30, // 30 seconds
                                          0,
                                          60 * 60}, // 1 hour
        .fastrak_loopback_only = {kFastrakLoopbackOnly, false},
        .fastrak_ifname = {kFastrakIfname, ""},
        .fastrak_socket_family = {kFastrakSocketFamily, ""},
        .fastrak_socket_ifname = {kFastrakSocketIfname, ""},
        .fastrak_comm_id = {kFastrakCommId, ""},
        .fastrak_use_llcm = {kFastrakUseLlcm, true},
        .fastrak_llcm_device_directory = {kFastrakLlcmDeviceDirectory, dxs::kLlcmDeviceDirectory},
        .fastrak_close_send_on_done = {kFastrakCloseSendOnDone, false},
        .fastrak_num_gpus_per_node = {kNumGpusPerNode, kMaxGpuDevices, 1, kMaxGpuDevices},
    };
}

} // namespace tcpxo

#endif // GPUDIRECT_TCPXO_PARAMS_H_
