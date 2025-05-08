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

#include <fcntl.h>

#include <iostream>

#include "agent_data.h"
#include "common/nixl_log.h"
#include "common/nixl_time.h"
#include "common/str_tools.h"
#include "nixl.h"
#if HAVE_ETCD
#include <etcd/Client.hpp>
#endif  // HAVE_ETCD

int connectToIP(std::string ip_addr, int port)
{
    struct sockaddr_in listenerAddr;
    listenerAddr.sin_port = htons(port);
    listenerAddr.sin_family = AF_INET;

    int ret_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (ret_fd == -1) { return -1; }

    if (inet_pton(AF_INET, ip_addr.c_str(), &listenerAddr.sin_addr) <= 0) {
        close(ret_fd);
        return -1;
    }

    // make connect block for now to avoid ambiguity in send right after
    int orig_flags = fcntl(ret_fd, F_GETFL, 0);
    int new_flags = orig_flags ^ O_NONBLOCK;

    fcntl(ret_fd, F_SETFL, new_flags);

    if (connect(ret_fd, (struct sockaddr*)&listenerAddr, sizeof(listenerAddr)) < 0) {
        perror("async connect");
        close(ret_fd);
        return -1;
    }

    // make nonblocking again
    fcntl(ret_fd, F_SETFL, orig_flags);

    return ret_fd;
}

size_t sendCommMessage(int fd, std::string msg)
{
    size_t bytes;
    bytes = send(fd, msg.c_str(), msg.size(), 0);
    if (bytes < 0) { std::cerr << "Cannot send on socket to fd " << fd << std::endl; }
    return bytes;
}

ssize_t recvCommMessage(int fd, std::string &msg)
{
    char    buffer[16384];
    ssize_t one_recv_bytes = 0;
    ssize_t recv_bytes = 0;
    msg = std::string("");

    do {
        one_recv_bytes = recv(fd, buffer, sizeof(buffer), 0);
        if (one_recv_bytes == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return recv_bytes;
            std::cerr << "Cannot recv on socket fd " << fd << std::endl;
            return one_recv_bytes;
        }
        msg.append(buffer, one_recv_bytes);
        recv_bytes += one_recv_bytes;
    } while (one_recv_bytes > 0);

    return recv_bytes;
}

#if HAVE_ETCD
// Helper function to create etcd key
static std::string makeEtcdKey(const std::string &agent_name, const std::string &namespace_prefix,
        const std::string &metadata_type)
{
    std::stringstream ss;
    ss << namespace_prefix << "/" << agent_name << "/" << metadata_type;
    return ss.str();
}

// Store metadata in etcd
static nixl_status_t storeMetadataInEtcd(const std::string &agent_name,
        const std::string &namespace_prefix, std::unique_ptr<etcd::Client> &client,
        const std::string &metadata_type, const nixl_blob_t &metadata)
{
    // Check if etcd client is available
    if (!client) {
        NIXL_ERROR << "ETCD client not available" << std::endl;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        // Create key for metadata
        std::string metadata_key = makeEtcdKey(agent_name, namespace_prefix, metadata_type);

        // Store metadata in etcd
        etcd::Response response = client->put(metadata_key, metadata).get();

        if (response.is_ok()) {
            NIXL_DEBUG << "Successfully stored " << metadata_type
                       << " in etcd with key: " << metadata_key << std::endl;
            return NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "Failed to store " << metadata_type
                       << " in etcd: " << response.error_message() << std::endl;
            return NIXL_ERR_BACKEND;
        }
    } catch (const std::exception &e) {
        NIXL_ERROR << "Error sending " << metadata_type << " to etcd: " << e.what() << std::endl;
        return NIXL_ERR_BACKEND;
    }
}

// Remove metadata from etcd
static nixl_status_t removeMetadataFromEtcd(const std::string &agent_name,
        const std::string &namespace_prefix, std::unique_ptr<etcd::Client> &client,
        const std::string &metadata_type)
{
    // Check if etcd client is available
    if (!client) {
        NIXL_ERROR << "ETCD client not available" << std::endl;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        // Create key for metadata
        std::string metadata_key = makeEtcdKey(agent_name, namespace_prefix, metadata_type);

        // Remove metadata from etcd
        etcd::Response response = client->rm(metadata_key).get();

        if (response.is_ok()) {
            NIXL_DEBUG << "Successfully removed " << metadata_type
                       << " from etcd for agent: " << agent_name << std::endl;
            return NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "Warning: Failed to remove " << metadata_type
                       << " from etcd: " << response.error_message() << std::endl;
            return NIXL_ERR_BACKEND;
        }
    } catch (const std::exception &e) {
        NIXL_ERROR << "Error removing " << metadata_type << " from etcd: " << e.what() << std::endl;
        return NIXL_ERR_BACKEND;
    }
}

// Fetch metadata from etcd
static nixl_status_t fetchMetadataFromEtcd(const std::string &agent_name,
        const std::string &namespace_prefix, std::unique_ptr<etcd::Client> &client,
        const std::string &metadata_type, nixl_blob_t &metadata)
{
    // Check if etcd client is available
    if (!client) {
        NIXL_ERROR << "ETCD client not available" << std::endl;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        // Create key for agent's metadata
        std::string metadata_key = makeEtcdKey(agent_name, namespace_prefix, metadata_type);

        // First check if the key is marked as invalid
        std::string    invalid_key = metadata_key + ".invalid";
        etcd::Response invalid_check = client->get(invalid_key).get();

        if (invalid_check.is_ok()) {
            // Key is marked as invalid, delete both keys
            NIXL_INFO << "Key " << metadata_key << " is marked as invalid, removing";
            removeMetadataFromEtcd(agent_name, namespace_prefix, client, "metadata.invalid");
            removeMetadataFromEtcd(agent_name, namespace_prefix, client, metadata_type);
            return NIXL_ERR_INVALID_PARAM;
        }

        // Fetch metadata from etcd
        etcd::Response response = client->get(metadata_key).get();

        if (response.is_ok()) {
            metadata = response.value().as_string();
            NIXL_DEBUG << "Successfully fetched " << metadata_type << " for agent: " << agent_name
                       << std::endl;
            return NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "Failed to fetch " << metadata_type
                       << " from etcd: " << response.error_message() << std::endl;
            return NIXL_ERR_NOT_FOUND;
        }
    } catch (const std::exception &e) {
        NIXL_ERROR << "Error fetching " << metadata_type << " from etcd: " << e.what() << std::endl;
        return NIXL_ERR_UNKNOWN;
    }
}

// Create etcd client with specified endpoints or from environment variable
static std::unique_ptr<etcd::Client> createEtcdClient(std::string etcd_endpoints)
{
    try {
        // Sanity check
        if (etcd_endpoints.size() == 0) { throw std::runtime_error("No etcd endpoints provided"); }

        // Create and return new etcd client
        return std::make_unique<etcd::Client>(etcd_endpoints);
    } catch (const std::exception &e) {
        NIXL_ERROR << "Error creating etcd client: " << e.what();
        return nullptr;
    }
}
#endif  // HAVE_ETCD

void nixlAgentData::commWorker(nixlAgent* myAgent)
{
#if HAVE_ETCD
    auto        etcdclient = std::unique_ptr<etcd::Client>(nullptr);
    std::string etcd_endpoints;
    std::string namespace_prefix;

    // useEtcd is set in nixlAgent constructor and is true if NIXL_ETCD_ENDPOINTS is set
    if (useEtcd) {
        etcd_endpoints = std::string(std::getenv("NIXL_ETCD_ENDPOINTS"));
        etcdclient = createEtcdClient(etcd_endpoints);

        NIXL_DEBUG << "Created etcd client to " << etcd_endpoints;

        if (std::getenv("NIXL_ETCD_NAMESPACE")) {
            namespace_prefix = std::string(std::getenv("NIXL_ETCD_NAMESPACE"));
        } else {
            namespace_prefix = NIXL_ETCD_NAMESPACE_DEFAULT;
        }
        NIXL_DEBUG << "Using etcd namespace for agents: " << namespace_prefix;
    }
#endif  // HAVE_ETCD

    while (!(commThreadStop)) {
        std::vector<nixl_comm_req_t> work_queue;

        // first, accept new connections
        int new_fd = 0;

        while (new_fd != -1 && config.useListenThread && !useEtcd) {
            new_fd = listener->acceptClient();
            nixl_socket_peer_t accepted_client;

            if (new_fd != -1) {
                // need to convert fd to IP address and add to client map
                sockaddr_in client_address;
                socklen_t   client_addrlen = sizeof(client_address);
                if (getpeername(new_fd, (sockaddr*)&client_address, &client_addrlen) == 0) {
                    char client_ip[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &client_address.sin_addr, client_ip, INET_ADDRSTRLEN);
                    accepted_client.first = std::string(client_ip);
                    accepted_client.second = client_address.sin_port;
                } else {
                    throw std::runtime_error("getpeername failed");
                }
                remoteSockets[accepted_client] = new_fd;

                // make new socket nonblocking
                int new_flags = fcntl(new_fd, F_GETFL, 0) | O_NONBLOCK;

                if (fcntl(new_fd, F_SETFL, new_flags) == -1)
                    throw std::runtime_error("fcntl accept");
            }
        }

        // second, do agent commands
        getCommWork(work_queue);

        for (nixl_comm_req_t request : work_queue) {
            nixl_comm_t req_command = std::get<0>(request);
            std::string req_ip = std::get<1>(request);
            int         req_port = std::get<2>(request);
            std::string my_MD = std::get<3>(request);

            nixl_socket_peer_t req_sock = std::make_pair(req_ip, req_port);

            // use remote IP for socket lookup
            auto client = remoteSockets.find(req_sock);
            int  client_fd;

            switch (req_command) {
                case SOCK_SEND: {
                    // not connected
                    if (client == remoteSockets.end()) {
                        int new_client = connectToIP(req_ip, req_port);
                        if (new_client == -1) {
                            NIXL_ERROR << "Listener thread could not connect to IP " << req_ip
                                       << " and port " << req_port << std::endl;
                            break;
                        }
                        remoteSockets[req_sock] = new_client;
                        client_fd = new_client;
                    } else {
                        client_fd = client->second;
                    }

                    sendCommMessage(client_fd, std::string("NIXLCOMM:LOAD" + my_MD));
                    break;
                }
                case SOCK_FETCH: {
                    if (client == remoteSockets.end()) {
                        int new_client = connectToIP(req_ip, req_port);
                        if (new_client == -1) {
                            NIXL_ERROR << "Listener thread could not connect to IP " << req_ip;
                            break;
                        }
                        remoteSockets[req_sock] = new_client;
                        client_fd = new_client;
                    } else
                        client_fd = client->second;

                    sendCommMessage(client_fd, std::string("NIXLCOMM:SEND"));
                    break;
                }
                case SOCK_INVAL: {
                    if (client == remoteSockets.end()) {
                        // improper usage
                        throw std::runtime_error("invalidate on closed socket\n");
                    }
                    client_fd = client->second;
                    sendCommMessage(client_fd, std::string("NIXLCOMM:INVL") + name);
                    break;
                }
#if HAVE_ETCD
                // ETCD operations using existing methods
                case ETCD_SEND: {
                    if (!useEtcd) { throw std::runtime_error("ETCD is not enabled"); }

                    // Parse request parameters
                    std::string metadata_type = "metadata";

                    if (req_ip.find("/partial_metadata") != std::string::npos) {
                        metadata_type = "partial_metadata";
                    }

                    // Use local storeMetadataInEtcd function
                    nixl_status_t ret = storeMetadataInEtcd(
                            name, namespace_prefix, etcdclient, metadata_type, my_MD);
                    if (ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "Failed to store metadata in etcd: " << ret << std::endl;
                    }
                    break;
                }
                case ETCD_FETCH: {
                    if (!useEtcd) { throw std::runtime_error("ETCD is not enabled"); }

                    // First try a direct get
                    nixl_blob_t   remote_metadata;
                    nixl_status_t ret = fetchMetadataFromEtcd(
                            req_ip, namespace_prefix, etcdclient, "metadata", remote_metadata);

                    if (ret == NIXL_SUCCESS) {
                        // Load the metadata
                        std::string remote_agent;
                        ret = myAgent->loadRemoteMD(remote_metadata, remote_agent);
                        if (ret == NIXL_SUCCESS) {
                            NIXL_DEBUG << "Successfully loaded metadata "
                                       << " for agent: " << req_ip << std::endl;
                        } else {
                            NIXL_ERROR << "Failed to load remote metadata: " << ret << std::endl;
                        }
                    } else if (ret == NIXL_ERR_INVALID_PARAM) {
                        NIXL_DEBUG << "Metadata was invalidated for agent: " << req_ip << std::endl;
                    } else {
                        // Key not found, set up a watch
                        NIXL_DEBUG << "Metadata not found, setting up watch for agent: " << req_ip
                                   << std::endl;

                        try {
                            // Create key for agent's metadata
                            std::string metadata_key =
                                    makeEtcdKey(req_ip, namespace_prefix, "metadata");

                            // Get current index to watch from
                            etcd::Response response = etcdclient->get(metadata_key).get();
                            int64_t        watch_index = response.index();
                            // Set up watch
                            etcd::Response watch_response =
                                    etcdclient->watch(metadata_key, watch_index).get();

                            if (watch_response.is_ok()) {
                                std::string remote_md = watch_response.value().as_string();
                                std::string remote_agent;
                                ret = myAgent->loadRemoteMD(remote_md, remote_agent);
                                if (ret != NIXL_SUCCESS) {
                                    NIXL_ERROR
                                            << "Failed to load remote metadata from watch: " << ret
                                            << std::endl;
                                } else {
                                    NIXL_DEBUG
                                            << "Successfully loaded metadata from watch for agent: "
                                            << req_ip << std::endl;
                                }
                            } else {
                                NIXL_ERROR << "Watch failed: " << watch_response.error_message()
                                           << std::endl;
                            }
                        } catch (const std::exception &e) {
                            NIXL_ERROR << "Error watching etcd: " << e.what() << std::endl;
                        }
                    }
                    break;
                }
                case ETCD_INVAL: {
                    if (!useEtcd) { throw std::runtime_error("ETCD is not enabled"); }

                    // The agent name comes in req_ip
                    try {
                        std::string agent = req_ip;

                        // Mark main metadata as invalid instead of removing it
                        std::string metadata_key = makeEtcdKey(agent, namespace_prefix, "metadata");
                        std::string invalid_key = metadata_key + ".invalid";

                        // Check if the metadata key exists first
                        etcd::Response check_response = etcdclient->get(metadata_key).get();
                        if (check_response.is_ok()) {
                            // Mark the key as invalid by creating an invalid marker
                            etcd::Response response1 =
                                    etcdclient->put(invalid_key, "invalid").get();
                            if (response1.is_ok()) {
                                NIXL_DEBUG << "Successfully marked metadata as invalid for agent: "
                                           << agent << std::endl;
                            } else {
                                NIXL_ERROR << "Warning: Failed to mark metadata as invalid: "
                                           << response1.error_message() << std::endl;
                            }
                        }

                        // Mark partial metadata as invalid
                        std::string partial_key =
                                makeEtcdKey(agent, namespace_prefix, "partial_metadata");
                        std::string partial_invalid_key = partial_key + ".invalid";

                        // Check if the partial metadata key exists first
                        etcd::Response check_partial = etcdclient->get(partial_key).get();
                        if (check_partial.is_ok()) {
                            // Mark the key as invalid
                            etcd::Response response2 =
                                    etcdclient->put(partial_invalid_key, "invalid").get();
                            if (response2.is_ok()) {
                                NIXL_DEBUG << "Successfully marked partial metadata as invalid for "
                                              "agent: "
                                           << agent << std::endl;
                            } else {
                                NIXL_ERROR
                                        << "Warning: Failed to mark partial metadata as invalid: "
                                        << response2.error_message() << std::endl;
                            }
                        }
                    } catch (const std::exception &e) {
                        NIXL_ERROR << "Error marking metadata as invalid: " << e.what()
                                   << std::endl;
                    }
                    break;
                }
#endif  // HAVE_ETCD
                default: {
                    throw std::runtime_error("Impossible command\n");
                    break;
                }
            }
        }

        // third, do remote commands
        auto socket_iter = remoteSockets.begin();
        while (socket_iter != remoteSockets.end()) {
            std::string              commands;
            std::vector<std::string> command_list;
            nixl_status_t            ret;

            ssize_t recv_bytes = recvCommMessage(socket_iter->second, commands);

            if (recv_bytes == 0 || recv_bytes == -1) {
                socket_iter++;
                continue;
            }

            command_list = str_split_substr(commands, "NIXLCOMM:");

            for (std::string command : command_list) {
                if (command.size() < 4) continue;

                // always just 4 chars:
                std::string header = command.substr(0, 4);

                if (header == "LOAD") {
                    std::string remote_md = command.substr(4);
                    std::string remote_agent;
                    ret = myAgent->loadRemoteMD(remote_md, remote_agent);
                    if (ret != NIXL_SUCCESS) {
                        throw std::runtime_error(
                                "loadRemoteMD in listener thread failed, critically failing\n");
                    }
                    // not sure what to do with remote_agent
                } else if (header == "SEND") {
                    nixl_blob_t my_MD;
                    myAgent->getLocalMD(my_MD);

                    sendCommMessage(socket_iter->second, std::string("NIXLCOMM:LOAD" + my_MD));
                } else if (header == "INVL") {
                    std::string remote_agent = command.substr(4);
                    myAgent->invalidateRemoteMD(remote_agent);
                    break;
                } else {
                    throw std::runtime_error("Received socket message with bad header" + header +
                            ", critically failing\n");
                }
            }

            socket_iter++;
        }

        nixlTime::us_t start = nixlTime::getUs();
        while ((start + config.lthrDelay) > nixlTime::getUs()) { std::this_thread::yield(); }
    }

    // Close remaining connections
    for (auto &[remote, fd] : remoteSockets) {
        shutdown(fd, SHUT_RDWR);
        close(fd);
    }
}

void nixlAgentData::enqueueCommWork(std::tuple<nixl_comm_t, std::string, int, std::string> request)
{
    std::lock_guard<std::mutex> lock(commLock);
    commQueue.push_back(request);
}

void nixlAgentData::getCommWork(std::vector<nixl_comm_req_t> &req_list)
{
    std::lock_guard<std::mutex> lock(commLock);
    req_list = commQueue;
    commQueue.clear();
}
