/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "nixl.h"
#include "common/nixl_time.h"
#include "agent_data.h"
#include "common/nixl_log.h"
#if HAVE_ETCD
#include <etcd/SyncClient.hpp>
#include <etcd/Watcher.hpp>
#include <future>
#endif // HAVE_ETCD
#include <absl/strings/str_format.h>
#include <absl/strings/str_split.h>
#include <poll.h>

const std::string default_metadata_label = "metadata";

namespace {

static const std::string invalid_label = "invalid";

int connectToIP(std::string ip_addr, int port) {

    struct sockaddr_in listenerAddr;
    listenerAddr.sin_port   = htons(port);
    listenerAddr.sin_family = AF_INET;

    if (inet_pton(AF_INET, ip_addr.c_str(), &listenerAddr.sin_addr) <= 0) {
        NIXL_ERROR << "inet_pton failed for ip_addr: " << ip_addr;
        return -1;
    }

    // Create a non-blocking socket
    int ret_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (ret_fd == -1) {
        NIXL_ERROR << "socket creation failed for ip_addr: " << ip_addr << " and port: " << port;
        return -1;
    }

    // Connect will return immediately with EINPROGRESS
    int ret = connect(ret_fd, (struct sockaddr*)&listenerAddr, sizeof(listenerAddr));
    if (ret < 0 && errno != EINPROGRESS) {
        close(ret_fd);
        return -1;
    }

    // Use poll to wait for connection with timeout
    struct pollfd pfd;
    pfd.fd = ret_fd;
    pfd.events = POLLOUT;
    pfd.revents = 0;

    ret = poll(&pfd, 1, 1000); // 1000ms timeout
    if (ret <= 0) {
        if (ret < 0) {
            NIXL_PERROR << "poll failed for ip_addr: " << ip_addr << " and port: " << port;
        } else {
            NIXL_ERROR << "poll timed out for ip_addr: " << ip_addr << " and port: " << port;
        }
        close(ret_fd);
        return -1;
    }

    if (!(pfd.revents & POLLOUT)) {
        NIXL_ERROR << "poll returned but socket not ready for write for ip_addr: " << ip_addr
                   << " and port: " << port;
        close(ret_fd);
        return -1;
    }

    // Check if connection was successful
    int error = 0;
    socklen_t len = sizeof(error);
    if (getsockopt(ret_fd, SOL_SOCKET, SO_ERROR, &error, &len) < 0 || error != 0) {
        NIXL_PERROR << "getsockopt failed for ip_addr: " << ip_addr << " and port: " << port;
        close(ret_fd);
        return -1;
    }

    return ret_fd;
}

void
sendCommMessage(int fd, const std::string& msg) {
    size_t size = msg.size();
    constexpr size_t iov_size = 2;
    struct iovec iov[iov_size] = {
        {&size, sizeof(size)},
        {const_cast<char*>(msg.data()), msg.size()}
    };

    for (size_t i = 0, offset = 0, sent = 0; i < iov_size;) {
        auto bytes = send(fd,
                          static_cast<char *>(iov[i].iov_base) + offset,
                          iov[i].iov_len - offset,
                          MSG_NOSIGNAL);
        if (bytes < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }

            throw std::runtime_error(
                absl::StrFormat("sendCommMessage(fd=%d, msg=%s) %zu/%zu bytes failed, errno=%d",
                                fd,
                                msg.c_str(),
                                sent,
                                size + sizeof(size),
                                errno));
        }

        offset += bytes;
        sent += bytes;
        if (offset == iov[i].iov_len) {
            offset = 0;
            ++i;
        }
    }
}

bool
recvCommMessageType(int fd, void *data, size_t size, bool force = false) {
    for (size_t received = 0; received < size;) {
        auto bytes = recv(fd, static_cast<char *>(data) + received, size - received, 0);
        if (bytes > 0) {
            received += bytes;
            continue;
        }
        if (bytes == 0 && received == 0 && !force) {
            return false;
        }

        if (bytes < 0) {
            if (errno == EINTR) {
                continue;
            }

            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (!force && received == 0) {
                    return false; // nothing to read yet
                }

                continue;
            }
        }

        throw std::runtime_error(
                absl::StrFormat("recvCommMessage(fd=%d) %zu/%zu bytes failed ret=%d errno=%d",
                                fd,
                                received,
                                size,
                                bytes,
                                errno));
    }

    return true;
}

bool
recvCommMessage(int fd, std::string &msg) {
    size_t size;
    if (!recvCommMessageType(fd, &size, sizeof(size))) {
        return false;
    }

    msg.resize(size);
    return recvCommMessageType(fd, msg.data(), size, true);
}

#if HAVE_ETCD
class nixlEtcdClient {
private:
    std::unique_ptr<etcd::SyncClient> etcd;
    std::string namespace_prefix;
    std::vector<std::string> invalidated_agents;
    std::mutex invalidated_agents_mutex;
    std::unordered_map<std::string, std::unique_ptr<etcd::Watcher>,
                        std::hash<std::string>, strEqual> agentWatchers;
    std::chrono::microseconds watchTimeout_;

    // Helper function to create etcd key
    std::string makeKey(const std::string& agent_name,
                        const std::string& metadata_type) {
        std::stringstream ss;
        ss << namespace_prefix << "/" << agent_name << "/" << metadata_type;
        return ss.str();
    }

public:
    nixlEtcdClient(const std::string &my_agent_name,
                   const std::chrono::microseconds &timeout = std::chrono::microseconds(5000000))
        : watchTimeout_(timeout) {
        const char* etcd_endpoints = std::getenv("NIXL_ETCD_ENDPOINTS");
        if (!etcd_endpoints || strlen(etcd_endpoints) == 0) {
            throw std::runtime_error("No etcd endpoints provided");
        }

        try {
            etcd = std::make_unique<etcd::SyncClient>(etcd_endpoints);
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Error creating etcd client: " << e.what();
            return;
        }
        NIXL_DEBUG << "Created etcd client to endpoints: " << etcd_endpoints;

        const char* etcd_namespace = std::getenv("NIXL_ETCD_NAMESPACE");
        namespace_prefix = etcd_namespace ? etcd_namespace : NIXL_ETCD_NAMESPACE_DEFAULT;

        NIXL_DEBUG << "Using etcd namespace for agents: " << namespace_prefix;

        std::string agent_prefix = makeKey(my_agent_name, "");
        etcd::Response response = etcd->put(agent_prefix, "");
        if (!response.is_ok()) {
            throw std::runtime_error("Failed to store agent " + my_agent_name +
                                     " prefix key in etcd: " + response.error_message());
        }
    }

    // Store metadata in etcd
    nixl_status_t storeMetadataInEtcd(const std::string& agent_name,
                                      const std::string& metadata_type,
                                      const nixl_blob_t& metadata) {
        if (!etcd) {
            NIXL_ERROR << "ETCD client not available";
            return NIXL_ERR_NOT_SUPPORTED;
        }

        try {
            std::string metadata_key = makeKey(agent_name, metadata_type);
            etcd::Response response = etcd->put(metadata_key, metadata);

            if (response.is_ok()) {
                NIXL_DEBUG << "Successfully stored " << metadata_type
                           << " in etcd with key: " << metadata_key << " (rev "
                           << response.value().modified_index() << ")";
                return NIXL_SUCCESS;
            } else {
                NIXL_ERROR << "Failed to store " << metadata_type << " in etcd: " << response.error_message();
                return NIXL_ERR_BACKEND;
            }
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Error sending " << metadata_type << " to etcd: " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    // Remove all agent's metadata from etcd
    nixl_status_t removeMetadataFromEtcd(const std::string& agent_name) {
        if (!etcd) {
            NIXL_ERROR << "ETCD client not available";
            return NIXL_ERR_NOT_SUPPORTED;
        }

        try {
            std::string agent_prefix = makeKey(agent_name, "");
            etcd::Response response = etcd->rmdir(agent_prefix, true);

            if (response.is_ok()) {
                NIXL_DEBUG << "Successfully removed " << response.values().size()
                           << " etcd keys for agent: " << agent_name;
                return NIXL_SUCCESS;
            } else {
                NIXL_ERROR << "Warning: Failed to remove etcd keys for agent: "
                           << agent_name << " : " << response.error_message();
                return NIXL_ERR_BACKEND;
            }
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Exception removing etcd keys for agent: " << agent_name << " : " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    // Fetch metadata from etcd
    nixl_status_t fetchMetadataFromEtcd(const std::string& agent_name,
                                        const std::string& metadata_type,
                                        nixl_blob_t& metadata) {
        if (!etcd) {
            NIXL_ERROR << "ETCD client not available";
            return NIXL_ERR_NOT_SUPPORTED;
        }

        std::string metadata_key = makeKey(agent_name, metadata_type);
        try {
            etcd::Response response = etcd->get(metadata_key);

            if (response.is_ok()) {
                metadata = response.value().as_string();
                NIXL_DEBUG << "Successfully fetched key: " << metadata_key
                           << " (rev " << response.value().modified_index() << ")";
                return NIXL_SUCCESS;
            } else {
                NIXL_INFO << "Failed to fetch key: " << metadata_key
                          << " from etcd: " << response.error_message();
                return NIXL_ERR_NOT_FOUND;
            }
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Error fetching key: " << metadata_key << " from etcd: " << e.what();
            return NIXL_ERR_UNKNOWN;
        }
    }

    nixl_status_t waitForMetadataFromEtcd(const std::string& metadata_key,
                                          nixl_blob_t& remote_metadata) {
        try {

            // Get current index to watch from
            etcd::Response response = etcd->get(metadata_key);
            int64_t watch_index = response.index();
            std::promise<nixl_status_t> ret_prom;
            auto future = ret_prom.get_future();
            std::atomic<bool> promise_set{false};

            // This lambda assumes lifetime only inside this method
            auto watcher_callback = [&](etcd::Response response) -> void {
                if (promise_set.exchange(true)) {
                    NIXL_DEBUG << "Ignoring subsequent watch event for key: " << metadata_key;
                    return;
                }

                if (!response.is_ok()) {
                    NIXL_ERROR << "Watch failed for key: " << metadata_key << " : "
                               << response.error_message();
                    ret_prom.set_value(NIXL_ERR_BACKEND);
                    return;
                }
                if (response.action() == "delete") {
                    NIXL_ERROR << "Watch response: metadata key deleted: " << metadata_key;
                    ret_prom.set_value(NIXL_ERR_INVALID_PARAM);
                    return;
                }
                remote_metadata = response.value().as_string();
                NIXL_DEBUG << "Watch response: metadata key fetched: " << metadata_key;
                ret_prom.set_value(NIXL_SUCCESS);
            };

            auto watcher = etcd::Watcher(*etcd, metadata_key, watch_index, watcher_callback);

            auto status = future.wait_for(watchTimeout_);
            if (status == std::future_status::timeout) {
                NIXL_ERROR << "Watch timed out for key: " << metadata_key;
                return NIXL_ERR_BACKEND;
            }
            watcher.Cancel();
            return future.get();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "Error watching etcd for key: " << metadata_key << " : " << e.what();
            return NIXL_ERR_BACKEND;
        }
    }

    // Fetch metadata from etcd or wait for it to be available
    nixl_status_t fetchOrWaitForMetadataFromEtcd(const std::string& remote_agent,
                                                 const std::string& metadata_label,
                                                 nixl_blob_t& remote_metadata) {
        nixl_status_t ret = fetchMetadataFromEtcd(remote_agent, metadata_label, remote_metadata);
        if (ret == NIXL_SUCCESS) {
            return NIXL_SUCCESS;
        }

        std::string metadata_key = makeKey(remote_agent, metadata_label);
        NIXL_DEBUG << "Metadata not found, setting up watch for: " << metadata_key;

        return waitForMetadataFromEtcd(metadata_key, remote_metadata);
    }

    // Setup a watcher for an agent's metadata invalidation if it doesn't already exist
    void setupAgentWatcher(const std::string &agent_name) {
        if (agentWatchers.find(agent_name) != agentWatchers.end()) {
            return;
        }

        // DELETE events are enqueued to be deleted in commWorker (can't be done inside the Watcher callback)
        auto process_response = [this, agent_name](etcd::Response response) -> void {
            if (!response.is_ok()) {
                NIXL_ERROR << "Watcher failed to watch agent " << agent_name
                           << " from etcd: " << response.error_message();
                return;
            }
            NIXL_DEBUG << "Watcher received " << response.events().size() << " events from etcd";
            if (response.events().size() != 1) {
                NIXL_ERROR << "Watcher agent " << agent_name << " received unexpected number of events from etcd: "
                           << response.events().size();
                return;
            }
            const auto &event = response.events()[0];
            if (event.event_type() == etcd::Event::EventType::DELETE_) {
                NIXL_DEBUG << "Watcher DELETE: " << event.kv().key()
                           << " (rev " << event.kv().modified_index() << ")";
                std::lock_guard<std::mutex> lock(invalidated_agents_mutex);
                invalidated_agents.push_back(agent_name);
            } else {
                NIXL_ERROR << "Watcher for " << event.kv().key()
                           << " received unexpected event from etcd: "
                           << static_cast<int>(event.event_type());
            }
        };

        std::string agent_prefix = makeKey(agent_name, "");
        agentWatchers[agent_name] = std::make_unique<etcd::Watcher>(*etcd, agent_prefix, process_response);
    }

    // Process invalidated agents from watchers
    void processInvalidatedAgents(nixlAgent* my_agent) {
        std::vector<std::string> tmp_invalidated_agents;
        {
            std::lock_guard<std::mutex> lock(invalidated_agents_mutex);
            tmp_invalidated_agents = std::move(invalidated_agents);
        }
        for (const auto &agent : tmp_invalidated_agents) {
            NIXL_DEBUG << "Invalidated agent: " << agent;
            agentWatchers.erase(agent);
            nixl_status_t ret = my_agent->invalidateRemoteMD(agent);
            if (ret != NIXL_SUCCESS)
                NIXL_ERROR << "Failed to invalidate remote metadata for agent: " << agent << ": " << ret;
            else
                NIXL_DEBUG << "Successfully invalidated remote metadata for agent: " << agent;
        }
    }
};
#endif // HAVE_ETCD

} // unnamed namespace

void
nixlAgentData::commWorker(nixlAgent &myAgent) noexcept {
    try {
        commWorkerInternal(&myAgent);
    }
    catch (...) {
        commThreadException_ = std::current_exception();
    }
}

void
nixlAgentData::commWorkerInternal(nixlAgent *myAgent) {
#if HAVE_ETCD
    std::unique_ptr<nixlEtcdClient> etcdClient = nullptr;
    // useEtcd is set in nixlAgent constructor and is true if NIXL_ETCD_ENDPOINTS is set
    if(useEtcd) {
        etcdClient = std::make_unique<nixlEtcdClient>(name, config.etcdWatchTimeout);
    }
#endif // HAVE_ETCD

    while(!(commThreadStop)) {
        std::vector<nixl_comm_req_t> work_queue;

        // first, accept new connections
        int new_fd = 0;

        while(new_fd != -1 && config.useListenThread) {
            new_fd = listener->acceptClient();
            nixl_socket_peer_t accepted_client;

            if(new_fd != -1){
                // need to convert fd to IP address and add to client map
                sockaddr_in client_address;
                socklen_t client_addrlen = sizeof(client_address);
                if (getpeername(new_fd, (sockaddr*)&client_address, &client_addrlen) == 0) {
                    char client_ip[INET_ADDRSTRLEN];
                    if (inet_ntop(AF_INET, &client_address.sin_addr, client_ip, INET_ADDRSTRLEN) ==
                        nullptr) {
                        NIXL_PERROR << "inet_ntop failed for client address";
                        close(new_fd);
                        throw std::runtime_error("inet_ntop failed for client address");
                    }
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
            auto client = remoteSockets.find(req_sock);

            // not connected
            if (req_command < SOCK_MAX) {
                if (client == remoteSockets.end()) {
                    int new_client = connectToIP(req_ip, req_port);
                    if (new_client == -1) {
                        NIXL_ERROR << "Listener thread could not connect to IP " << req_ip
                                   << " and port " << req_port;
                        continue;
                    }
                    remoteSockets[req_sock] = new_client;
                    client = remoteSockets.find(req_sock);
                }
            }

            bool needs_disconnect = false;
            switch(req_command) {
            case SOCK_SEND: {
                try {
                    sendCommMessage(client->second, "NIXLCOMM:LOAD" + my_MD);
                }
                catch (const std::runtime_error &e) {
                    NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
                    needs_disconnect = true;
                }
                break;
            }
            case SOCK_FETCH: {
                try {
                    sendCommMessage(client->second, "NIXLCOMM:SEND");
                }
                catch (const std::runtime_error &e) {
                    NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
                    needs_disconnect = true;
                }
                break;
            }
            case SOCK_INVAL: {
                try {
                    sendCommMessage(client->second, "NIXLCOMM:INVL" + name);
                }
                catch (const std::runtime_error &e) {
                    NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
                    needs_disconnect = true;
                }
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
                    nixl_status_t ret = etcdClient->storeMetadataInEtcd(name, metadata_label, my_MD);
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
                    nixl_status_t ret = etcdClient->fetchOrWaitForMetadataFromEtcd(remote_agent, metadata_label,
                                                                                   remote_metadata);
                    if (ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "Failed to fetch metadata from etcd: " << ret;
                        break;
                    }

                    std::string remote_agent_from_md;
                    ret = myAgent->loadRemoteMD(remote_metadata, remote_agent_from_md);
                    if (ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "Failed to load remote metadata: " << ret;
                        break;
                    } else if (remote_agent_from_md != remote_agent) {
                        NIXL_ERROR << "Metadata mismatch for agent: " << remote_agent
                                   << " from md: " << remote_agent_from_md;
                        break;
                    }
                    NIXL_DEBUG << "Successfully loaded metadata for agent: " << remote_agent;

                    etcdClient->setupAgentWatcher(remote_agent);
                    break;
                }
                case ETCD_INVAL:
                {
                    if (!useEtcd) {
                        throw std::runtime_error("ETCD is not enabled");
                    }

                    nixl_status_t ret = etcdClient->removeMetadataFromEtcd(name);
                    if (ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "Failed to invalidate metadata in etcd: " << ret;
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
            if (needs_disconnect) {
                close(client->second);
                client = remoteSockets.erase(client);
            }
        }

        // third, do remote commands
        auto socket_iter = remoteSockets.begin();
        while (socket_iter != remoteSockets.end()) {
            std::string commands;
            std::vector<std::string> command_list;
            nixl_status_t ret;
            bool disconnected = false;

            try {
                const bool received = recvCommMessage(socket_iter->second, commands);
                if (!received) {
                    // No message received, but without error condition.
                    // Skip to the next peer
                    socket_iter++;
                    continue;
                }
            }
            catch (const std::runtime_error &e) {
                NIXL_ERROR << "Failed to receive message from peer, disconnecting: " << e.what();
                close(socket_iter->second);
                socket_iter = remoteSockets.erase(socket_iter);
                continue;
            }

            command_list = absl::StrSplit(commands, "NIXLCOMM:");

            for(const auto &command : command_list) {

                if(command.size() < 4) continue;

                // always just 4 chars:
                std::string header = command.substr(0, 4);

                if(header == "LOAD") {
                    std::string remote_md = command.substr(4);
                    std::string remote_agent;
                    ret = myAgent->loadRemoteMD(remote_md, remote_agent);
                    if(ret != NIXL_SUCCESS) {
                        NIXL_ERROR << "loadRemoteMD in listener thread failed for md from peer "
                                   << socket_iter->first.first << ":" << socket_iter->first.second
                                   << " with error " << ret;
                        continue;
                    }
                    // not sure what to do with remote_agent
                } else if(header == "SEND") {
                    nixl_blob_t my_MD;
                    myAgent->getLocalMD(my_MD);

                    try {
                        sendCommMessage(socket_iter->second, std::string("NIXLCOMM:LOAD" + my_MD));
                    }
                    catch (const std::runtime_error &e) {
                        NIXL_ERROR << "Failed to send message to peer, disconnecting: " << e.what();
                        disconnected = true;
                        break;
                    }
                } else if(header == "INVL") {
                    std::string remote_agent = command.substr(4);
                    myAgent->invalidateRemoteMD(remote_agent);
                    break;
                } else {
                    NIXL_ERROR << "Received socket message with bad header" + header + " from peer "
                               << socket_iter->first.first << ":" << socket_iter->first.second;
                }
            }

            if (disconnected) {
                close(socket_iter->second);
                socket_iter = remoteSockets.erase(socket_iter);
            } else {
                socket_iter++;
            }
        }

#if HAVE_ETCD
        if (etcdClient) {
            etcdClient->processInvalidatedAgents(myAgent);
        }
#endif // HAVE_ETCD

        nixlTime::us_t start = nixlTime::getUs();
        while( (start + config.lthrDelay) > nixlTime::getUs()) {
            std::this_thread::yield();
        }
    }
}

void nixlAgentData::enqueueCommWork(nixl_comm_req_t request){
    if (agentShutdown) {
        NIXL_WARN << "Agent shutting down, unable to accept new requests";
        return;
    }
    std::lock_guard<std::mutex> lock(commLock);
    commQueue.push_back(std::move(request));
}

void nixlAgentData::getCommWork(std::vector<nixl_comm_req_t> &req_list){
    std::lock_guard<std::mutex> lock(commLock);
    req_list = std::move(commQueue);
    commQueue.clear();
}

nixl_status_t
nixlAgentData::loadConnInfo(const std::string &remote_name,
                            const nixl_backend_t &backend,
                            const nixl_blob_t &conn_info) {
    if (backendEngines.count(backend) == 0) {
        NIXL_DEBUG << "Agent " << name << " does not support a remote backend: " << backend;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    // No need to reload same conn info, error if it changed
    if ((remoteBackends.count(remote_name) != 0) &&
        (remoteBackends[remote_name].count(backend) != 0)) {
        if (remoteBackends[remote_name][backend] != conn_info) {
            return NIXL_ERR_NOT_ALLOWED;
        }

        return NIXL_SUCCESS;
    }

    nixlBackendEngine *eng = backendEngines[backend];
    if (!eng->supportsRemote()) {
        NIXL_DEBUG << backend << " does not support remote operations";
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const nixl_status_t ret = eng->loadRemoteConnInfo(remote_name, conn_info);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    remoteBackends[remote_name].emplace(backend, conn_info);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgentData::loadRemoteSections(const std::string &remote_name, nixlSerDes &sd) {
    if (remoteSections.count(remote_name) == 0) {
        remoteSections[remote_name] = new nixlRemoteSection(remote_name);
    }

    const nixl_status_t ret = remoteSections[remote_name]->loadRemoteData(&sd, backendEngines);
    // TODO: can be more graceful, if just the new MD blob was improper
    if (ret != NIXL_SUCCESS) {
        delete remoteSections[remote_name];
        remoteSections.erase(remote_name);
        remoteBackends.erase(remote_name);
        return ret;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlAgentData::invalidateRemoteData(const std::string &remote_name) {
    if (remote_name == name) {
        NIXL_ERROR << "Agent " << name << " cannot invalidate itself";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixl_status_t ret = NIXL_ERR_NOT_FOUND;
    auto it_section = remoteSections.find(remote_name);
    if (it_section != remoteSections.end()) {
        delete it_section->second;
        remoteSections.erase(it_section);
        ret = NIXL_SUCCESS;
    }

    auto it_backends = remoteBackends.find(remote_name);
    if (it_backends != remoteBackends.end()) {
        for (auto &it : it_backends->second) {
            backendEngines[it.first]->disconnect(remote_name);
        }

        remoteBackends.erase(it_backends);
        ret = NIXL_SUCCESS;
    }

    return ret;
}
