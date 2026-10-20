/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Standalone real-NIXL replay, no payload pack or private core API.
#include <cuda_runtime.h>
#include <nixl.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sched.h>
#include <unistd.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

extern "C" int
fillPayload(void *, void *, uint64_t);
extern "C" int
validatePayload(const void *,
                const void *,
                unsigned,
                uint64_t,
                const uint64_t *,
                unsigned,
                unsigned,
                unsigned *,
                unsigned *);
using Clock = std::chrono::steady_clock;
constexpr size_t L = 27, R = 64, W = 576, P = 4, S = 3;

void
check(nixl_status_t s) {
    if (s != NIXL_SUCCESS) {
        throw std::runtime_error("NIXL status " + std::to_string(s));
    }
}

void
gpu(int s) {
    if (s != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(cudaError_t(s)));
    }
}

void
require(bool value, const char *message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

double
us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

void
exact(int fd, void *buffer, size_t length, bool sending) {
    auto *p = static_cast<char *>(buffer);
    while (length) {
        const auto n = sending ? send(fd, p, length, MSG_NOSIGNAL) : recv(fd, p, length, 0);
        require(n > 0, "control socket closed/error");
        p += n;
        length -= size_t(n);
    }
}

void
put(int fd, std::string data) {
    uint32_t size = htonl(uint32_t(data.size()));
    exact(fd, &size, 4, true);
    exact(fd, data.data(), data.size(), true);
}

std::string
get(int fd) {
    uint32_t size;
    exact(fd, &size, 4, false);
    size = ntohl(size);
    require(size <= 16 * 1024 * 1024, "oversized control frame");
    std::string data(size, '\0');
    exact(fd, data.data(), size, false);
    return data;
}

struct Agent {
    std::unique_ptr<nixlAgent> api;
    nixl_reg_dlist_t reg{VRAM_SEG};
    void *data = nullptr, *marker = nullptr;
    unsigned *errors = nullptr;
};

int
main(int argc, char **argv) {
    // ROLE BACKEND HOST CONTROL_PORT OOB_BASE NIC IFACE GID ITERATIONS WARMUP
    if (argc < 11 || argc > 13) {
        return 2;
    }
    try {
        const bool sender = std::string(argv[1]) == "sender";
        require(sender || std::string(argv[1]) == "receiver", "invalid role");
        const std::string backend = argv[2];
        require(backend == "UCX" || backend == "GPUNETIO", "invalid backend");
        const bool strided = argc >= 12 && std::string(argv[11]) == "strided";
        require(argc == 11 || strided, "invalid mode");
        require(!strided || backend == "GPUNETIO", "strided mode requires GPUNETIO");
        const bool noNotif = argc == 13 && std::string(argv[12]) == "no-notif";
        const bool negativeInputs = argc == 13 && std::string(argv[12]) == "negative";
        require(argc < 13 || noNotif || negativeInputs, "invalid diagnostic");
        const size_t transferredRows =
            std::getenv("PD_REPLAY_ROWS") ? std::stoul(std::getenv("PD_REPLAY_ROWS")) : R;
        const size_t transferredLayers =
            std::getenv("PD_REPLAY_LAYERS") ? std::stoul(std::getenv("PD_REPLAY_LAYERS")) : L;
        require(transferredRows >= 1 && transferredRows <= R && transferredLayers >= 1 &&
                    transferredLayers <= L,
                "invalid workload geometry");
        const int count = std::stoi(argv[9]), warmup = std::stoi(argv[10]);
        require(count > 0 && count <= 1000000 && warmup >= 0 && warmup <= 1000,
                "invalid iteration count");
        const bool layerMrs = std::getenv("PD_REPLAY_LAYER_MRS") != nullptr;
        const int replays =
            std::getenv("PD_REPLAY_REPOSTS") ? std::stoi(std::getenv("PD_REPLAY_REPOSTS")) : 1;
        const int standalone = std::getenv("PD_REPLAY_STANDALONE") ?
            std::stoi(std::getenv("PD_REPLAY_STANDALONE")) :
            0;
        require(replays >= 1 && replays <= 8 && standalone >= 0 && standalone <= 4,
                "invalid qualification counts");
        std::cout << "WORKLOAD " << transferredLayers << ' ' << transferredRows << ' ' << W << ' '
                  << P << " reposts=" << replays << " standalone=" << standalone
                  << " layer_mrs=" << layerMrs << " cpu=" << sched_getcpu() << std::endl;
        gpu(cudaSetDevice(0));
        std::array<Agent, P> agents;
        const size_t bytes = L * S * R * W * (sender ? P : 1);
        for (size_t i = 0; i < (sender ? 1 : P); ++i) {
            auto &a = agents[i];
            gpu(cudaMalloc(&a.data, bytes));
            gpu(cudaMemset(a.data, 0xa5, bytes));
            gpu(cudaMalloc(&a.marker, (sender ? P : 1) * sizeof(uint64_t)));
            gpu(cudaMemset(a.marker, 0, (sender ? P : 1) * sizeof(uint64_t)));
            if (!sender) {
                gpu(cudaMalloc(&a.errors, sizeof(unsigned)));
                uint64_t empty[3]{};
                unsigned errors = 0;
                gpu(validatePayload(a.marker,
                                    a.data,
                                    unsigned(i),
                                    0,
                                    empty,
                                    transferredRows,
                                    transferredLayers,
                                    a.errors,
                                    &errors));
                require(errors == 0, "GPU validator positive control");
                gpu(cudaMemset(a.data, 0, 1));
                gpu(validatePayload(a.marker,
                                    a.data,
                                    unsigned(i),
                                    0,
                                    empty,
                                    transferredRows,
                                    transferredLayers,
                                    a.errors,
                                    &errors));
                require(errors == 1, "GPU validator corruption control");
                gpu(cudaMemset(a.data, 0xa5, 1));
            }
            nixlAgentConfig cfg;
            cfg.useProgThread = false;
            a.api = std::make_unique<nixlAgent>(
                std::string(sender ? "pd-source-" : "pd-target-") + std::to_string(i), cfg);
            nixl_mem_list_t types;
            nixl_b_params_t params;
            check(a.api->getPluginParams(backend, types, params));
            if (backend == "UCX") {
                params["num_threads"] = "0";
            } else {
                params["network_devices"] = argv[6];
                params["oob_interface"] = argv[7];
                params["gid_index"] = argv[8];
                params["gpu_devices"] = "0";
                params["oob_port"] = std::to_string(std::stoi(argv[5]) + int(i));
                if (const char *owners = std::getenv("PD_REPLAY_PROGRESS_OWNERS")) {
                    params["progress_owners"] = owners;
                }
                if (const char *ordered = std::getenv("PD_REPLAY_ORDERED_NOTIFICATIONS")) {
                    params["ordered_notifications"] = ordered;
                }
            }
            nixlBackendH *handle = nullptr;
            check(a.api->createBackend(backend, params, handle));
            const size_t regions = layerMrs ? L : 1;
            for (size_t region = 0; region < regions; ++region) {
                a.reg.addDesc(nixlBlobDesc(
                    uintptr_t(a.data) + region * (bytes / regions), bytes / regions, 0));
            }
            a.reg.addDesc(
                nixlBlobDesc(uintptr_t(a.marker), (sender ? P : 1) * sizeof(uint64_t), 0));
            check(a.api->registerMem(a.reg));
        }
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        require(fd >= 0, "socket");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(uint16_t(std::stoi(argv[4])));
        if (sender) {
            require(inet_pton(AF_INET, argv[3], &addr.sin_addr) == 1, "invalid host");
            require(connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "connect");
        } else {
            int yes = 1;
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
            addr.sin_addr.s_addr = INADDR_ANY;
            require(bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0, "bind");
            require(listen(fd, 1) == 0, "listen");
            std::cout << "READY" << std::endl;
            const int accepted = accept(fd, nullptr, nullptr);
            close(fd);
            fd = accepted;
            require(fd >= 0, "accept");
        }
        timeval timeout{30, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        int noDelay = 1;
        require(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &noDelay, sizeof(noDelay)) == 0,
                "TCP_NODELAY");
        std::array<std::string, P> peers;
        std::array<uintptr_t, P> remoteData{}, remoteMarker{};
        if (sender) {
            nixl_blob_t md;
            check(agents[0].api->getLocalMD(md));
            put(fd, md);
            for (size_t i = 0; i < P; ++i) {
                auto blob = get(fd);
                check(agents[0].api->loadRemoteMD(blob, peers[i]));
                remoteData[i] = std::stoull(get(fd));
                remoteMarker[i] = std::stoull(get(fd));
            }
        } else {
            const auto incoming = get(fd);
            for (auto &a : agents) {
                std::string peer;
                check(a.api->loadRemoteMD(incoming, peer));
                nixl_blob_t md;
                check(a.api->getLocalMD(md));
                put(fd, md);
                put(fd, std::to_string(uintptr_t(a.data)));
                put(fd, std::to_string(uintptr_t(a.marker)));
            }
        }
        // Record actual mappings for this process, not merely files present on disk.
        std::ifstream maps("/proc/self/maps");
        for (std::string line; std::getline(maps, line);) {
            if (line.find("libnixl") != std::string::npos ||
                line.find("libplugin_") != std::string::npos ||
                line.find("libucp") != std::string::npos ||
                line.find("libdoca") != std::string::npos) {
                std::cout << "MAP " << line << '\n';
            }
        }
        std::cout << std::flush;
        std::array<std::array<uint64_t, S>, P> epochs{};
        std::array<nixl_xfer_dlist_t, P> local = {nixl_xfer_dlist_t(VRAM_SEG),
                                                  nixl_xfer_dlist_t(VRAM_SEG),
                                                  nixl_xfer_dlist_t(VRAM_SEG),
                                                  nixl_xfer_dlist_t(VRAM_SEG)};
        auto remote = local;
        const size_t payloadDescriptors =
            strided ? transferredLayers : transferredLayers * transferredRows;
        if (sender) {
            for (size_t peer = 0; peer < P; ++peer) {
                local[peer].resize(payloadDescriptors + 1);
                remote[peer].resize(payloadDescriptors + 1);
                for (size_t row = 0; row < payloadDescriptors + 1; ++row) {
                    local[peer][row].len = remote[peer][row].len =
                        row == payloadDescriptors ? 8 : (strided ? transferredRows * W : W);
                    local[peer][row].devId = remote[peer][row].devId = 0;
                }
                local[peer][payloadDescriptors].addr = uintptr_t(agents[0].marker) + peer * 8;
                remote[peer][payloadDescriptors].addr = remoteMarker[peer];
            }
        }
        for (int n = 0; n < count + warmup; ++n) {
            const uint64_t epoch = uint64_t(n) + 1;
            if (sender) {
                gpu(fillPayload(agents[0].data, agents[0].marker, epoch));
                require(get(fd) == "ready", "ready epoch");
                const auto begin = Clock::now();
                for (size_t peer = 0; peer < P; ++peer) {
                    for (size_t layer = 0; layer < transferredLayers; ++layer) {
                        for (size_t row = 0; row < (strided ? 1 : transferredRows); ++row) {
                            const auto src =
                                ((layer * S * R + size_t(n) % S * R + row) * P + peer) * W;
                            const auto dst = (layer * S * R + (size_t(n) + peer) % S * R + row) * W;
                            const size_t index = strided ? layer : layer * transferredRows + row;
                            local[peer][index].addr = uintptr_t(agents[0].data) + src;
                            remote[peer][index].addr = remoteData[peer] + dst;
                        }
                    }
                }
                const auto planned = Clock::now();
                if (negativeInputs && n == 0) {
                    auto reject = [&](std::string label,
                                      nixl_xfer_dlist_t ls,
                                      nixl_xfer_dlist_t rs,
                                      std::string plan,
                                      nixl_xfer_op_t op,
                                      nixl_status_t expected) {
                        nixl_opt_args_t options;
                        options.customParam = std::move(plan);
                        nixlXferReqH *request = nullptr;
                        const auto status =
                            agents[0].api->createXferReq(op, ls, rs, peers[0], request, &options);
                        if (status == NIXL_SUCCESS) {
                            check(agents[0].api->releaseXferReq(request));
                        }
                        require(status == expected && request == nullptr,
                                "negative input status/handle mismatch");
                        std::cout << "NEGATIVE_PASS " << label << ' ' << int(status) << std::endl;
                    };
                    const std::string valid =
                        "gpunetio_strided_write_v1=" + std::to_string(transferredRows) + ",576,4,1";
                    reject("zero-rows",
                           local[0],
                           remote[0],
                           "gpunetio_strided_write_v1=0,576,4,1",
                           NIXL_WRITE,
                           NIXL_ERR_INVALID_PARAM);
                    reject("parser-overflow",
                           local[0],
                           remote[0],
                           "gpunetio_strided_write_v1=64,576,4,4294967296",
                           NIXL_WRITE,
                           NIXL_ERR_INVALID_PARAM);
                    reject("read-unsupported",
                           local[0],
                           remote[0],
                           valid,
                           NIXL_READ,
                           NIXL_ERR_INVALID_PARAM);
                    auto badSource = local[0];
                    badSource[0].addr = uintptr_t(agents[0].data) + bytes - transferredRows * W;
                    reject("expanded-source-span",
                           badSource,
                           remote[0],
                           valid,
                           NIXL_WRITE,
                           NIXL_ERR_INVALID_PARAM);
                    auto badTarget = remote[0];
                    badTarget[0].addr = remoteData[0] + L * S * R * W - transferredRows * W;
                    reject("expanded-target-span",
                           local[0],
                           badTarget,
                           "gpunetio_strided_write_v1=" + std::to_string(transferredRows) +
                               ",576,4,4",
                           NIXL_WRITE,
                           NIXL_ERR_INVALID_PARAM);
                    badSource = local[0];
                    badTarget = remote[0];
                    --badSource[0].len;
                    --badTarget[0].len;
                    reject("logical-length",
                           badSource,
                           badTarget,
                           valid,
                           NIXL_WRITE,
                           NIXL_ERR_INVALID_PARAM);
                }
                std::array<nixlXferReqH *, P> requests{};
                for (size_t peer = 0; peer < P; ++peer) {
                    nixl_opt_args_t options;
                    if (!noNotif) {
                        options.notif =
                            "epoch-" + std::to_string(epoch) + "-peer-" + std::to_string(peer);
                    }
                    if (strided) {
                        options.customParam =
                            "gpunetio_strided_write_v1=" + std::to_string(transferredRows) +
                            ",576,4,1";
                    }
                    check(agents[0].api->createXferReq(NIXL_WRITE,
                                                       local[peer],
                                                       remote[peer],
                                                       peers[peer],
                                                       requests[peer],
                                                       &options));
                }
                const auto prepared = Clock::now();
                Clock::time_point posted;
                for (int replay = 0; replay < replays; ++replay) {
                    for (auto request : requests) {
                        const auto status = agents[0].api->postXferReq(request);
                        require(status == NIXL_SUCCESS || status == NIXL_IN_PROG, "post failed");
                    }
                    posted = Clock::now();
                    for (int round = 0; round < standalone; ++round) {
                        for (size_t peer = 0; peer < P; ++peer) {
                            check(agents[0].api->genNotif(peers[peer],
                                                          "standalone-" + std::to_string(epoch) +
                                                              "-round-" + std::to_string(round) +
                                                              "-peer-" + std::to_string(peer)));
                        }
                    }
                    for (auto request : requests) {
                        auto status = agents[0].api->getXferStatus(request);
                        while (status == NIXL_IN_PROG &&
                               Clock::now() - begin < std::chrono::seconds(15)) {
                            status = agents[0].api->getXferStatus(request);
                        }
                        check(status);
                    }
                }
                const auto done = Clock::now();
                put(fd, "done");
                require(get(fd) == "validated", "receiver validation");
                const auto visible = Clock::now();
                for (auto request : requests) {
                    check(agents[0].api->releaseXferReq(request));
                }
                if (n >= warmup) {
                    std::cout << "SAMPLE " << n - warmup << ' ' << us(begin, planned) << ' '
                              << us(planned, prepared) << ' ' << us(prepared, posted) << ' '
                              << us(posted, done) << ' ' << us(begin, done) << ' '
                              << us(begin, visible) << '\n';
                }
            } else {
                put(fd, "ready");
                std::array<std::multiset<std::string>, P> expected;
                for (size_t peer = 0; peer < P; ++peer) {
                    for (int replay = 0; replay < replays; ++replay) {
                        expected[peer].insert("epoch-" + std::to_string(epoch) + "-peer-" +
                                              std::to_string(peer));
                        for (int round = 0; round < standalone; ++round) {
                            expected[peer].insert("standalone-" + std::to_string(epoch) +
                                                  "-round-" + std::to_string(round) + "-peer-" +
                                                  std::to_string(peer));
                        }
                    }
                }
                auto begin = Clock::now();
                pollfd control{fd, POLLIN, 0};
                auto waiting = [&] {
                    if (noNotif) {
                        return poll(&control, 1, 0) == 0;
                    }
                    return !std::all_of(expected.begin(), expected.end(), [](const auto &messages) {
                        return messages.empty();
                    });
                };
                while (waiting()) {
                    require(Clock::now() - begin < std::chrono::seconds(15),
                            "notification timeout");
                    for (size_t peer = 0; peer < P; ++peer) {
                        nixl_notifs_t messages;
                        check(agents[peer].api->getNotifs(messages));
                        for (const auto &[name, values] : messages) {
                            for (const auto &value : values) {
                                (void)name;
                                auto entry = expected[peer].find(value);
                                require(!noNotif && entry != expected[peer].end(),
                                        "unexpected notification");
                                expected[peer].erase(entry);
                            }
                        }
                    }
                }
                require(get(fd) == "done", "sender done");
                for (size_t peer = 0; peer < P; ++peer) {
                    epochs[peer][(size_t(n) + peer) % S] = epoch;
                    unsigned errors = 0;
                    gpu(validatePayload(agents[peer].marker,
                                        agents[peer].data,
                                        unsigned(peer),
                                        epoch,
                                        epochs[peer].data(),
                                        transferredRows,
                                        transferredLayers,
                                        agents[peer].errors,
                                        &errors));
                    require(errors == 0, "GPU marker/payload/window validation");
                }
                put(fd, "validated");
            }
        }
        // Both sides finish progress before either side destroys remote metadata.
        put(fd, "finished");
        require(get(fd) == "finished", "finish handshake");
        close(fd);
        // Stop independent persistent engines together before context-wide unregistration.
        std::vector<std::thread> shutdown;
        for (size_t i = 0; i < (sender ? 1 : P); ++i) {
            auto &a = agents[i];
            check(a.api->deregisterMem(a.reg));
        }
        for (size_t i = 0; i < (sender ? 1 : P); ++i) {
            shutdown.emplace_back([&, i] {
                gpu(cudaSetDevice(0));
                agents[i].api.reset();
            });
        }
        for (auto &thread : shutdown) {
            thread.join();
        }
        for (size_t i = 0; i < (sender ? 1 : P); ++i) {
            auto &a = agents[i];
            gpu(cudaFree(a.data));
            gpu(cudaFree(a.marker));
            if (a.errors) {
                gpu(cudaFree(a.errors));
            }
        }
        std::cout << "REPLAY_PASS " << count + warmup << ' ' << backend << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "REPLAY_FAIL " << e.what() << std::endl;
        return 1;
    }
}
