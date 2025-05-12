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
#include "nixl.h"
#include "common/nixl_time.h"
#include "common/str_tools.h"
#include "agent_data.h"
#include "common/nixl_log.h"
#if HAVE_ETCD
#include <etcd/Client.hpp>
#endif // HAVE_ETCD

const std::string default_metadata_label = "metadata";
const std::string default_partial_metadata_label = "partial_metadata";

namespace {

static const std::string invalid_label = "invalid";

int connectToIP(std::string ip_addr, int port) {

    struct sockaddr_in listenerAddr;
    listenerAddr.sin_port   = htons(port);
    listenerAddr.sin_family = AF_INET;

    int ret_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (ret_fd == -1) {
        return -1;
    }

    if (inet_pton(AF_INET, ip_addr.c_str(),
                  &listenerAddr.sin_addr) <= 0) {
        close(ret_fd);
        return -1;
    }

    //make connect block for now to avoid ambiguity in send right after
    int orig_flags = fcntl(ret_fd, F_GETFL, 0);
    int new_flags = orig_flags ^ O_NONBLOCK;

    fcntl(ret_fd, F_SETFL, new_flags);

    if (connect(ret_fd, (struct sockaddr*)&listenerAddr,
                    sizeof(listenerAddr)) < 0) {
        perror("async connect");
        close(ret_fd);
        return -1;
    }

    //make nonblocking again
    fcntl(ret_fd, F_SETFL, orig_flags);

    return ret_fd;
}

size_t sendCommMessage(int fd, std::string msg){
    size_t bytes;
    bytes = send(fd, msg.c_str(), msg.size(), 0);
    if(bytes < 0) {
        NIXL_ERROR << "Cannot send on socket to fd " << fd;
    }
    return bytes;
}

ssize_t recvCommMessage(int fd, std::string &msg){
    char buffer[16384];
    ssize_t one_recv_bytes = 0;
    ssize_t recv_bytes = 0;
    msg = std::string("");

    do {
        one_recv_bytes = recv(fd, buffer, sizeof(buffer), 0);
        if (one_recv_bytes == -1){
            if(errno == EAGAIN || errno == EWOULDBLOCK) return recv_bytes;
            NIXL_ERROR << "Cannot recv on socket fd " << fd;
            return one_recv_bytes;
        }
        msg.append(buffer, one_recv_bytes);
        recv_bytes += one_recv_bytes;
    } while(one_recv_bytes > 0);

    return recv_bytes;
}

#if HAVE_ETCD
// Helper function to create etcd key
static std::string makeEtcdKey(const std::string& agent_name,
                                const std::string& namespace_prefix,
                                const std::string& metadata_type) {
    std::stringstream ss;
    ss << namespace_prefix << "/" << agent_name << "/" << metadata_type;
    return ss.str();
}

// Store metadata in etcd
static nixl_status_t storeMetadataInEtcd(const std::string& agent_name,
                                   const std::string& namespace_prefix,
                                   std::unique_ptr<etcd::Client>& client,
                                   const std::string& metadata_type,
                                   const nixl_blob_t& metadata) {
    // Check if etcd client is available
    if (!client) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        // Create key for metadata
        std::string metadata_key = makeEtcdKey(agent_name, namespace_prefix, metadata_type);

        // Store metadata in etcd
        etcd::Response response = client->put(metadata_key, metadata).get();

        if (response.is_ok()) {
            NIXL_DEBUG << "Successfully stored " << metadata_type << " in etcd with key: " << metadata_key << " (rev " << response.value().modified_index() << ")";
            return NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "Failed to store " << metadata_type << " in etcd: " << response.error_message();
            return NIXL_ERR_BACKEND;
        }
    } catch (const std::exception& e) {
        NIXL_ERROR << "Error sending " << metadata_type << " to etcd: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

// Remove all agent's metadata from etcd
static nixl_status_t removeMetadataFromEtcd(const std::string& agent_name,
                                            const std::string& namespace_prefix,
                                            std::unique_ptr<etcd::Client>& client) {
    // Check if etcd client is available
    if (!client) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    try {
        // Create key for metadata with agent's prefix
        std::string agent_prefix = makeEtcdKey(agent_name, namespace_prefix, "");

        // Remove all keys for the agent from etcd
        etcd::Response response = client->rmdir(agent_prefix, true).get();

        if (response.is_ok()) {
            NIXL_DEBUG << "Successfully removed " << response.values().size()
                       << " etcd keys for agent: " << agent_name;
            return NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "Warning: Failed to remove etcd keys for agent: "
                       << agent_name << " : " << response.error_message();
            return NIXL_ERR_BACKEND;
        }
    } catch (const std::exception& e) {
        NIXL_ERROR << "Exception removing etcd keys for agent: " << agent_name << " : " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

// Fetch metadata from etcd
static nixl_status_t fetchMetadataFromEtcd(const std::string& agent_name,
                                     const std::string& namespace_prefix,
                                     std::unique_ptr<etcd::Client>& client,
                                     const std::string& metadata_type,
                                     nixl_blob_t& metadata) {
    // Check if etcd client is available
    if (!client) {
        NIXL_ERROR << "ETCD client not available";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // Create key for agent's metadata
    std::string metadata_key = makeEtcdKey(agent_name, namespace_prefix, metadata_type);
    try {
        // First check if the key is marked as invalid
        std::string invalid_key = makeEtcdKey(agent_name, namespace_prefix, invalid_label);
        etcd::Response invalid_check = client->get(invalid_key).get();

        if (invalid_check.is_ok()) {
            // Key is marked as invalid, delete both keys
            NIXL_INFO << "Agent " << agent_name << " is marked as invalid, removing all keys";
            removeMetadataFromEtcd(agent_name, namespace_prefix, client);
            return NIXL_ERR_INVALID_PARAM;
        }

        // Fetch metadata from etcd
        etcd::Response response = client->get(metadata_key).get();

        if (response.is_ok()) {
            metadata = response.value().as_string();
            NIXL_DEBUG << "Successfully fetched key: " << metadata_key << " (rev " << response.value().modified_index() << ")";
            return NIXL_SUCCESS;
        } else {
            NIXL_ERROR << "Failed to fetch key: " << metadata_key << " from etcd: " << response.error_message();
            return NIXL_ERR_NOT_FOUND;
        }
    } catch (const std::exception& e) {
        NIXL_ERROR << "Error fetching key: " << metadata_key << " from etcd: " << e.what();
        return NIXL_ERR_UNKNOWN;
    }
}

// Create etcd client with specified endpoints or from environment variable
static std::unique_ptr<etcd::Client> createEtcdClient(std::string etcd_endpoints) {
    try {
        // Sanity check
        if (etcd_endpoints.size() == 0) {
            throw std::runtime_error("No etcd endpoints provided");
        }

        // Create and return new etcd client
        return std::make_unique<etcd::Client>(etcd_endpoints);
    } catch (const std::exception& e) {
        NIXL_ERROR << "Error creating etcd client: " << e.what();
        return nullptr;
    }
}
#endif // HAVE_ETCD

} // unnamed namespace

void nixlAgentData::commWorker(nixlAgent* myAgent){

#if HAVE_ETCD
        auto etcdclient = std::unique_ptr<etcd::Client>(nullptr);
        std::string etcd_endpoints;
        std::string namespace_prefix;

        // useEtcd is set in nixlAgent constructor and is true if NIXL_ETCD_ENDPOINTS is set
        if(useEtcd) {
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
#endif // HAVE_ETCD

    while(!(commThreadStop)) {
        std::vector<nixl_comm_req_t> work_queue;

        // first, accept new connections
        int new_fd = 0;

        while(new_fd != -1 && config.useListenThread && !useEtcd) {
            new_fd = listener->acceptClient();
            nixl_socket_peer_t accepted_client;

            if(new_fd != -1){
                // need to convert fd to IP address and add to client map
                sockaddr_in client_address;
                socklen_t client_addrlen = sizeof(client_address);
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

        for(const auto &request: work_queue) {

            // TODO: req_ip and req_port are relevant only for SOCK_*, need different request structure for ETCD_*
            const auto &[req_command, req_ip, req_port, my_MD] = request;

            nixl_socket_peer_t req_sock = std::make_pair(req_ip, req_port);

            // use remote IP for socket lookup
            const auto client = remoteSockets.find(req_sock);
            int client_fd;

            switch(req_command) {
                case SOCK_SEND:
                {
                    // not connected
                    if(client == remoteSockets.end()) {
                        int new_client = connectToIP(req_ip, req_port);
                        if(new_client == -1) {
                            NIXL_ERROR << "Listener thread could not connect to IP " << req_ip << " and port " << req_port;
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
                case SOCK_FETCH:
                {
                    if(client == remoteSockets.end()) {
                        int new_client = connectToIP(req_ip, req_port);
                        if(new_client == -1) {
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
                case SOCK_INVAL:
                {
                    if(client == remoteSockets.end()) {
                        // improper usage
                        throw std::runtime_error("invalidate on closed socket\n");
                    }
                    client_fd = client->second;
                    sendCommMessage(client_fd, std::string("NIXLCOMM:INVL") + name);
                    break;
                }
#if HAVE_ETCD
                // ETCD operations using existing methods
                case ETCD_SEND:
                {
                    if (!useEtcd) {
                        throw std::runtime_error("ETCD is not enabled");
                    }

                    // Parse request parameters
                    const std::string &metadata_label = req_ip;

                    // Use local storeMetadataInEtcd function
                    nixl_status_t ret = storeMetadataInEtcd(name, namespace_prefix, etcdclient, metadata_label, my_MD);
                    if (ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "Failed to store metadata in etcd: " << ret;
                    }
                    break;
                }
                case ETCD_FETCH:
                {
                    if (!useEtcd) {
                        throw std::runtime_error("ETCD is not enabled");
                    }

                    const std::string &metadata_label = req_ip;
                    const std::string &remote_agent = my_MD;

                    // First try a direct get
                    nixl_blob_t remote_metadata;
                    nixl_status_t ret = fetchMetadataFromEtcd(remote_agent, namespace_prefix, etcdclient, metadata_label, remote_metadata);

                    if (ret == NIXL_SUCCESS) {
                        // Load the metadata
                        std::string remote_agent_from_md;
                        ret = myAgent->loadRemoteMD(remote_metadata, remote_agent_from_md);
                        if (ret == NIXL_SUCCESS) {
                            if (remote_agent_from_md != remote_agent) {
                                NIXL_ERROR << "Metadata mismatch for agent: " << remote_agent << " from md: " << remote_agent_from_md;
                                break;
                            }
                            NIXL_DEBUG << "Successfully loaded metadata for agent: " << remote_agent;
                        } else {
                            NIXL_ERROR << "Failed to load remote metadata: " << ret;
                        }
                    } else if (ret == NIXL_ERR_INVALID_PARAM) {
                        NIXL_DEBUG << "Metadata was invalidated for agent: " << remote_agent;
                    } else {
                        // Key not found, set up a watch
                        NIXL_DEBUG << "Metadata not found, setting up watch for agent: " << remote_agent;

                        try {
                            // Create key for agent's metadata
                            std::string metadata_key = makeEtcdKey(remote_agent, namespace_prefix, "metadata");

                            // Get current index to watch from
                            etcd::Response response = etcdclient->get(metadata_key).get();
                            int64_t watch_index = response.index();
                            // Set up watch
                            etcd::Response watch_response = etcdclient->watch(metadata_key, watch_index).get();

                            if (watch_response.is_ok()) {
                                std::string remote_md = watch_response.value().as_string();
                                std::string remote_agent_from_md;
                                ret = myAgent->loadRemoteMD(remote_md, remote_agent_from_md);
                                if (ret != NIXL_SUCCESS) {
                                    NIXL_ERROR << "Failed to load remote metadata from watch: " << ret;
                                    if (remote_agent_from_md != remote_agent) {
                                        NIXL_ERROR << "Metadata mismatch for agent: " << remote_agent << " from md: " << remote_agent_from_md;
                                        break;
                                    }
                                } else {
                                    NIXL_DEBUG << "Successfully loaded metadata from watch for agent: " << remote_agent;
                                }
                            } else {
                                NIXL_ERROR << "Watch failed: " << watch_response.error_message();
                            }
                        } catch (const std::exception& e) {
                            NIXL_ERROR << "Error watching etcd: " << e.what();
                        }
                    }
                    break;
                }
                case ETCD_INVAL:
                {
                    if (!useEtcd) {
                        throw std::runtime_error("ETCD is not enabled");
                    }

                    try {
                        // Mark agent's metadata as invalid instead of removing it
                        std::string agent_prefix = makeEtcdKey(name, namespace_prefix, "");
                        std::string invalid_key = makeEtcdKey(name, namespace_prefix, invalid_label);

                        // Check if the agent has any keys in etcd first. 1 key is enough to confirm.
                        etcd::Response check_response = etcdclient->keys(agent_prefix, 1).get();
                        if (check_response.is_ok() && check_response.keys().size() > 0) {
                            // Mark the agent's metadata as invalid by creating an invalid marker
                            etcd::Response response1 = etcdclient->put(invalid_key, "invalid").get();
                            if (response1.is_ok()) {
                                NIXL_DEBUG << "Successfully marked metadata as invalid for agent: " << name;
                            } else {
                                NIXL_ERROR << "Warning: Failed to mark metadata as invalid for agent: "
                                           << name << " : " << response1.error_message();
                            }
                        } else {
                            NIXL_DEBUG << "Agent " << name << " has no keys in etcd, skipping invalidation";
                        }
                    } catch (const std::exception& e) {
                        NIXL_ERROR << "Error marking metadata as invalid for agent: "
                                   << name << " : " << e.what();
                    }
                    break;
                }
#endif // HAVE_ETCD
                default:
                {
                    throw std::runtime_error("Impossible command\n");
                    break;
                }
            }
        }

        // third, do remote commands
        auto socket_iter = remoteSockets.begin();
        while (socket_iter != remoteSockets.end()) {
            std::string commands;
            std::vector<std::string> command_list;
            nixl_status_t ret;

            ssize_t recv_bytes = recvCommMessage(socket_iter->second, commands);

            if(recv_bytes == 0 || recv_bytes == -1) {
                socket_iter++;
                continue;
            }

            command_list = str_split_substr(commands, "NIXLCOMM:");

            for(const auto &command : command_list) {

                if(command.size() < 4) continue;

                // always just 4 chars:
                std::string header = command.substr(0, 4);

                if(header == "LOAD") {
                    std::string remote_md = command.substr(4);
                    std::string remote_agent;
                    ret = myAgent->loadRemoteMD(remote_md, remote_agent);
                    if(ret != NIXL_SUCCESS) {
                        throw std::runtime_error("loadRemoteMD in listener thread failed, critically failing\n");
                    }
                    // not sure what to do with remote_agent
                } else if(header == "SEND") {
                    nixl_blob_t my_MD;
                    myAgent->getLocalMD(my_MD);

                    sendCommMessage(socket_iter->second, std::string("NIXLCOMM:LOAD" + my_MD));
                } else if(header == "INVL") {
                    std::string remote_agent = command.substr(4);
                    myAgent->invalidateRemoteMD(remote_agent);
                    break;
                } else {
                    throw std::runtime_error("Received socket message with bad header" + header + ", critically failing\n");
                }
            }

            socket_iter++;
        }

        nixlTime::us_t start = nixlTime::getUs();
        while( (start + config.lthrDelay) > nixlTime::getUs()) {
            std::this_thread::yield();
        }
    }
}

void nixlAgentData::enqueueCommWork(nixl_comm_req_t request){
    std::lock_guard<std::mutex> lock(commLock);
    commQueue.push_back(std::move(request));
}

void nixlAgentData::getCommWork(std::vector<nixl_comm_req_t> &req_list){
    std::lock_guard<std::mutex> lock(commLock);
    req_list = std::move(commQueue);
    commQueue.clear();
}
