/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <random>
#include <type_traits>
#include <vector>
#include "absl/cleanup/cleanup.h"
#include "compression_backend.h"
#include "delta_backend.h"
#include "nixl_service.h"
#include "nixl_service_data.h"
#include "staging_backend.h"
#include "unit/services/marshals/mock_marshal_backend.h"
#include <cuda_runtime.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace gtest {
namespace services {

    // Infra-free: does not use the agent fixture, so it needs neither etcd nor a CUDA device.
    TEST(ServiceFingerprint, RoundTrip) {
        marshalLayoutFingerprint fingerprint;
        fingerprint.mode = 3;
        fingerprint.algo = static_cast<uint32_t>(nixl_marshal_compress_algo_t::ANS_DELTA);
        fingerprint.chunkedPayloadSize = 128 * 1024 * 1024;
        fingerprint.chunkSize = 128 * 1024;
        fingerprint.wireDataCapacity = 130 * 1024 * 1024;
        fingerprint.memType = static_cast<uint32_t>(VRAM_SEG);

        const nixl_blob_t bytes = serializeFingerprint(fingerprint);
        const marshalLayoutFingerprint deserialized = deserializeFingerprint(bytes);
        EXPECT_EQ(deserialized, fingerprint);
        EXPECT_FALSE(deserialized != fingerprint);

        // A truncated buffer must be rejected rather than read out of bounds. The result is
        // intentionally discarded here (cast to void) since only the thrown exception matters;
        // deserializeFingerprint() is [[nodiscard]] for every other (non-throwing) call site.
        const nixl_blob_t truncated = bytes.substr(0, bytes.size() - 1);
        EXPECT_THROW((void)deserializeFingerprint(truncated), std::runtime_error);
    }

    TEST(ServiceAgentData, NullBackendRejected) {
        EXPECT_THROW(
            { nixlServiceAgentData(nixlMarshalStagingConfig{}, 128 * 1024, false, nullptr); },
            std::invalid_argument);
        EXPECT_THROW(
            { nixlServiceAgentData(nixlMarshalDirectConfig{}, 128 * 1024, false, nullptr); },
            std::invalid_argument);
    }

    TEST(ServiceCompressionOptions, DefaultDataTypeIsFloat16) {
        const nixlMarshalCompressOptArgs options;
        EXPECT_EQ(options.dataType, nixl_marshal_compress_data_type_t::FLOAT16);
    }

    TEST(ServiceReadPayloads, RReqCompressionDataTypeValidation) {
        EXPECT_TRUE(isValidCompressionDataType(nixl_marshal_compress_data_type_t::CHAR));
        EXPECT_TRUE(isValidCompressionDataType(nixl_marshal_compress_data_type_t::UCHAR));
        EXPECT_TRUE(isValidCompressionDataType(nixl_marshal_compress_data_type_t::FLOAT16));
        EXPECT_TRUE(isValidCompressionDataType(nixl_marshal_compress_data_type_t::FLOAT8_E4M3));
        EXPECT_FALSE(
            isValidCompressionDataType(nixl_marshal_compress_data_type_t::NUM_COMPRESS_DATA_TYPES));
        EXPECT_FALSE(
            isValidCompressionDataType(static_cast<nixl_marshal_compress_data_type_t>(UINT8_MAX)));
    }

    // Infra-free round-trip tests for the new READ protocol payloads (RREQ/RPOSTED/RRSLOT/
    // RABORT/RABORT_ACK/RNAK). None of these are produced or consumed by any code path yet;
    // these tests only verify each payload's own serialize()/deserialize-ctor fidelity.

    TEST(ServiceReadPayloads, RReqRoundTrip) {
        marshalLayoutFingerprint fingerprint;
        fingerprint.mode = 3;
        fingerprint.algo = static_cast<uint32_t>(nixl_marshal_compress_algo_t::ANS);
        fingerprint.chunkedPayloadSize = 128 * 1024 * 1024;
        fingerprint.chunkSize = 128 * 1024;
        fingerprint.wireDataCapacity = 130 * 1024 * 1024;
        fingerprint.memType = static_cast<uint32_t>(VRAM_SEG);

        // Embedded null byte, to prove the length-prefixed encoding doesn't rely on
        // NUL-termination. "fake-serialized-dlist" (21) + '\0' (1) + "with-null" (9) = 31 bytes.
        const std::string serialized_src_list("fake-serialized-dlist\0with-null", 31);
        const std::array<nixlBasicDesc, slots_per_xfer> recv_slots{nixlBasicDesc(0x3000, 4096, 0),
                                                                   nixlBasicDesc(0x4000, 4096, 0)};
        const std::array<nixlMarshal::mem_space_t, slots_per_xfer> recv_mem_spaces{
            nixlMarshal::mem_space_t::DEVICE, nixlMarshal::mem_space_t::DEVICE};

        {
            // Without delta opt-args.
            rReqPayload payload(7, serialized_src_list, recv_slots, recv_mem_spaces, fingerprint);
            payload.dataType = nixl_marshal_compress_data_type_t::FLOAT8_E4M3;
            rReqPayload deserialized(payload.serialize());
            EXPECT_EQ(deserialized.xferId, 7u);
            EXPECT_EQ(deserialized.serializedSrcList, serialized_src_list);
            EXPECT_EQ(deserialized.recvSlotDescriptors, recv_slots);
            EXPECT_EQ(deserialized.recvMemSpaces, recv_mem_spaces);
            EXPECT_EQ(deserialized.fingerprint, fingerprint);
            EXPECT_EQ(deserialized.dataType, nixl_marshal_compress_data_type_t::FLOAT8_E4M3);
            EXPECT_FALSE(deserialized.deltaOptArgs.has_value());
        }
        {
            // With delta opt-args, set after construction (mirrors rtsNotifPayload's pattern).
            rReqPayload payload(8, serialized_src_list, recv_slots, recv_mem_spaces, fingerprint);
            payload.deltaOptArgs = nixlMarshalDeltaSenderRefArgs{
                reinterpret_cast<std::byte *>(0xABCD0000), VRAM_SEG, 4};
            rReqPayload deserialized(payload.serialize());
            EXPECT_EQ(deserialized.xferId, 8u);
            ASSERT_TRUE(deserialized.deltaOptArgs.has_value());
            EXPECT_EQ(deserialized.deltaOptArgs->ref, payload.deltaOptArgs->ref);
            EXPECT_EQ(deserialized.deltaOptArgs->memType, payload.deltaOptArgs->memType);
            EXPECT_EQ(deserialized.deltaOptArgs->elementSize, payload.deltaOptArgs->elementSize);
        }
    }

    TEST(ServiceReadPayloads, RPostedRoundTrip) {
        constexpr size_t wire_size = 4096;
        constexpr auto backend = nixl_marshal_mode_t::COMPRESSION;
        rPostedPayload payload(11, 1, 6144, wire_size, 0x10000, "compress-metadata", backend);
        rPostedPayload deserialized(payload.serialize());
        EXPECT_EQ(deserialized.xferId, 11u);
        EXPECT_EQ(deserialized.slotIndex, 1u);
        EXPECT_EQ(deserialized.originalSize, 6144u);
        EXPECT_EQ(deserialized.wireSize, wire_size);
        EXPECT_EQ(deserialized.dstByteOffset, static_cast<size_t>(0x10000));
        EXPECT_EQ(deserialized.metadata, "compress-metadata");
        EXPECT_EQ(deserialized.backend, backend);
    }

    TEST(ServiceReadPayloads, RRSlotRoundTrip) {
        rrSlotPayload payload(3, 0, 42);
        rrSlotPayload deserialized(payload.serialize());
        EXPECT_EQ(deserialized.xferId, 3u);
        EXPECT_EQ(deserialized.slotIndex, 0u);
        EXPECT_EQ(deserialized.slotGeneration, 42u);
    }

    TEST(ServiceReadPayloads, RAbortRoundTrip) {
        rAbortPayload payload(5);
        rAbortPayload deserialized(payload.serialize());
        EXPECT_EQ(deserialized.xferId, 5u);
    }

    TEST(ServiceReadPayloads, RAbortAckRoundTrip) {
        rAbortAckPayload payload(6);
        rAbortAckPayload deserialized(payload.serialize());
        EXPECT_EQ(deserialized.xferId, 6u);
    }

    TEST(ServiceReadPayloads, RNakRoundTrip) {
        rNakPayload payload(9, static_cast<int32_t>(NIXL_ERR_NOT_SUPPORTED));
        rNakPayload deserialized(payload.serialize());
        EXPECT_EQ(deserialized.xferId, 9u);
        EXPECT_EQ(deserialized.errorCode, static_cast<int32_t>(NIXL_ERR_NOT_SUPPORTED));
    }

    namespace {

        bool
        isPortOpen(const char *ip, uint16_t port) {
            int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                return false;
            }

            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            if (::inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
                ::close(fd);
                return false;
            }

            bool ok = (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);
            ::close(fd);
            return ok;
        }

        template<typename MarshalConfig>
        MarshalConfig
        makeMarshalConfig(nixl_marshal_compress_algo_t /*algo*/) {
            return MarshalConfig{};
        }

        template<>
        nixlMarshalCompressConfig
        makeMarshalConfig<nixlMarshalCompressConfig>(nixl_marshal_compress_algo_t algo) {
            return nixlMarshalCompressConfig{algo};
        }

        // The service fixes the chunked payload size at 128 MB, which these tests cannot
        // afford in device memory - and ReadPeerRejectsFingerprintMismatch needs two agents
        // that disagree on it - so they build agents through the protected prepare() seam.
        class testServiceAgent : public nixlServiceAgent {
        public:
            testServiceAgent(const std::string &name,
                             nixlServiceAgentConfig cfg,
                             size_t chunked_payload_size)
                : nixlServiceAgent(name, prepare(std::move(cfg), chunked_payload_size)) {}

            testServiceAgent(const std::string &name,
                             nixlServiceAgentConfig cfg,
                             size_t chunked_payload_size,
                             std::shared_ptr<nixlMarshal::backend> backend)
                : nixlServiceAgent(
                      name,
                      prepare(std::move(cfg), chunked_payload_size, std::move(backend))) {}
        };

        template<typename MarshalConfig>
        nixl_marshal_opt_args_t
        makeMarshalOptArgsForConfig();

        template<>
        nixl_marshal_opt_args_t
        makeMarshalOptArgsForConfig<nixlMarshalStagingConfig>() {
            return nixlMarshalStagingOptArgs{};
        }

        template<>
        nixl_marshal_opt_args_t
        makeMarshalOptArgsForConfig<nixlMarshalCompressConfig>() {
            return nixlMarshalCompressOptArgs{};
        }

        template<>
        nixl_marshal_opt_args_t
        makeMarshalOptArgsForConfig<nixlMarshalDeltaConfig>() {
            return nixlMarshalDeltaOptArgs{};
        }

    } // namespace

    template<typename MarshalConfig,
             size_t slotPerServiceMemT,
             size_t chunkSizeT,
             nixl_marshal_compress_algo_t compressAlgo = nixl_marshal_compress_algo_t::ANS,
             size_t memorySizeT = 1024 * chunkSizeT>
    class nixlServiceTestF : public ::testing::Test {
    public:
        using marshal_config_t = MarshalConfig;

        // Public so TYPED_TEST can use them as TypeParam:: constants (TestFixture is the
        // empty nixlServiceTest wrapper; inherited static constexpr values are not constant
        // expressions when named through that intermediate class).
        static constexpr size_t memSize = memorySizeT;
        static constexpr size_t chunkSize = chunkSizeT;

    protected:
        std::unique_ptr<nixlServiceAgent> agent_0_;
        std::unique_ptr<nixlServiceAgent> agent_1_;
        bool etcd_valid_ = false;
        bool cuda_valid_ = false;
        int device_id_ = 0;
        // True for plain delta, and for compression configured with the ANS_DELTA algo - the
        // two cases that need a delta reference buffer on each side.
        static constexpr bool usesDelta = std::is_same_v<MarshalConfig, nixlMarshalDeltaConfig> ||
            (std::is_same_v<MarshalConfig, nixlMarshalCompressConfig> &&
             compressAlgo == nixl_marshal_compress_algo_t::ANS_DELTA);

        static size_t
        serviceMemSize() {
            static_assert(slotPerServiceMemT % MarshalBackendSizing::slots_per_transfer == 0);
            constexpr auto num_slot_groups =
                slotPerServiceMemT / MarshalBackendSizing::slots_per_transfer;
            static_assert(num_slot_groups > 1);
            constexpr auto max_concurrent_transfers = num_slot_groups - 1;
            return nixlService::recommendServiceMemSize(
                makeMarshalConfig<MarshalConfig>(compressAlgo),
                max_concurrent_transfers,
                chunkSize);
        }

        void *mem_agent_0_ = nullptr;
        void *svc_mem_agent_0_ = nullptr;
        void *mem_agent_1_ = nullptr;
        void *svc_mem_agent_1_ = nullptr;
        void *delta_ref_agent_0_ = nullptr;
        void *delta_ref_agent_1_ = nullptr;
        std::vector<unsigned char> src_data_;
        std::vector<unsigned char> dst_data_;
        std::shared_ptr<nixlMarshal::backend> injectedBackend_;

        void
        SetUp() override {
            if constexpr (usesDelta) {
                GTEST_SKIP() << "DeltaBackend/CompressionBackend ans_delta not implemented "
                                "(delta mode disabled)";
            }
            auto env_etcd_endpoints_set = (std::getenv("NIXL_ETCD_ENDPOINTS") != nullptr);
            auto etcd_running = isPortOpen("127.0.0.1", 2379);
            if (env_etcd_endpoints_set && !etcd_running) {
                GTEST_FAIL()
                    << "NIXL_ETCD_ENDPOINTS is set, but etcd is not running, skipping test";
            }
            nixlServiceAgentConfig cfg;
            cfg.mode = makeMarshalConfig<MarshalConfig>(compressAlgo);
            cfg.useProgThread = true;
            cfg.captureTelemetry = true;
            const auto make_agent = [&](const char *name) {
                return injectedBackend_ ?
                    std::make_unique<testServiceAgent>(name, cfg, chunkSize, injectedBackend_) :
                    std::make_unique<testServiceAgent>(name, cfg, chunkSize);
            };
            agent_0_ = make_agent("agent_0");
            agent_1_ = make_agent("agent_1");
            etcd_valid_ = env_etcd_endpoints_set && etcd_running;
            if (!etcd_valid_) {
                return;
            }
            nixlBackendH *backend_handle_0 = nullptr;
            nixlBackendH *backend_handle_1 = nullptr;
            ASSERT_EQ(agent_0_->createBackend("UCX", {}, backend_handle_0), NIXL_SUCCESS);
            ASSERT_EQ(agent_1_->createBackend("UCX", {}, backend_handle_1), NIXL_SUCCESS);
            ASSERT_NE(backend_handle_0, nullptr);
            ASSERT_NE(backend_handle_1, nullptr);

            int device_count = 0;
            ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
            if (device_count == 0) {
                GTEST_FAIL() << "No CUDA device available";
            }
            ASSERT_EQ(cudaGetDevice(&device_id_), cudaSuccess);

            const auto service_mem_size = serviceMemSize();
            ASSERT_EQ(cudaMalloc(&mem_agent_0_, memSize), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&svc_mem_agent_0_, service_mem_size), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&mem_agent_1_, memSize), cudaSuccess);
            ASSERT_EQ(cudaMalloc(&svc_mem_agent_1_, service_mem_size), cudaSuccess);
            if constexpr (usesDelta) {
                ASSERT_EQ(cudaMalloc(&delta_ref_agent_0_, memSize), cudaSuccess);
                ASSERT_EQ(cudaMalloc(&delta_ref_agent_1_, memSize), cudaSuccess);
                ASSERT_EQ(cudaMemset(delta_ref_agent_0_, 0, memSize), cudaSuccess);
                ASSERT_EQ(cudaMemset(delta_ref_agent_1_, 0, memSize), cudaSuccess);
            }

            src_data_.resize(memSize);
            dst_data_.assign(memSize, 0);
            std::fill(src_data_.begin(), src_data_.end(), 0);
            ASSERT_EQ(cudaMemset(mem_agent_0_, 0, memSize), cudaSuccess);
            ASSERT_EQ(cudaMemset(mem_agent_1_, 0, memSize), cudaSuccess);

            nixl_reg_dlist_t mem_descs_agent_0(VRAM_SEG);
            mem_descs_agent_0.addDesc(
                nixlBlobDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
            nixl_reg_dlist_t svc_descs_agent_0(VRAM_SEG);
            svc_descs_agent_0.addDesc(nixlBlobDesc(
                reinterpret_cast<uintptr_t>(svc_mem_agent_0_), service_mem_size, device_id_));

            nixl_reg_dlist_t mem_descs_agent_1(VRAM_SEG);
            mem_descs_agent_1.addDesc(
                nixlBlobDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));
            nixl_reg_dlist_t svc_descs_agent_1(VRAM_SEG);
            svc_descs_agent_1.addDesc(nixlBlobDesc(
                reinterpret_cast<uintptr_t>(svc_mem_agent_1_), service_mem_size, device_id_));

            ASSERT_EQ(agent_0_->registerMem(mem_descs_agent_0), NIXL_SUCCESS);
            ASSERT_EQ(agent_0_->registerServiceMem(svc_descs_agent_0), NIXL_SUCCESS);
            ASSERT_EQ(agent_1_->registerMem(mem_descs_agent_1), NIXL_SUCCESS);
            ASSERT_EQ(agent_1_->registerServiceMem(svc_descs_agent_1), NIXL_SUCCESS);

            nixl_blob_t md_agent_0;
            nixl_blob_t md_agent_1;
            ASSERT_EQ(agent_0_->getLocalMD(md_agent_0), NIXL_SUCCESS);
            ASSERT_EQ(agent_1_->getLocalMD(md_agent_1), NIXL_SUCCESS);

            std::string remote_name_0;
            std::string remote_name_1;
            ASSERT_EQ(agent_1_->loadRemoteMD(md_agent_0, remote_name_1), NIXL_SUCCESS);
            ASSERT_EQ(remote_name_1, "agent_0");
            ASSERT_EQ(agent_0_->loadRemoteMD(md_agent_1, remote_name_0), NIXL_SUCCESS);
            ASSERT_EQ(remote_name_0, "agent_1");

            cuda_valid_ = true;
        }

        void
        TearDown() override {
            // Ensure progress threads stop before freeing registered GPU memory.
            agent_0_.reset();
            agent_1_.reset();

            if (delta_ref_agent_1_ != nullptr) {
                (void)cudaFree(delta_ref_agent_1_);
                delta_ref_agent_1_ = nullptr;
            }
            if (delta_ref_agent_0_ != nullptr) {
                (void)cudaFree(delta_ref_agent_0_);
                delta_ref_agent_0_ = nullptr;
            }
            if (svc_mem_agent_1_ != nullptr) {
                (void)cudaFree(svc_mem_agent_1_);
                svc_mem_agent_1_ = nullptr;
            }
            if (mem_agent_1_ != nullptr) {
                (void)cudaFree(mem_agent_1_);
                mem_agent_1_ = nullptr;
            }
            if (svc_mem_agent_0_ != nullptr) {
                (void)cudaFree(svc_mem_agent_0_);
                svc_mem_agent_0_ = nullptr;
            }
            if (mem_agent_0_ != nullptr) {
                (void)cudaFree(mem_agent_0_);
                mem_agent_0_ = nullptr;
            }
        }

        nixl_marshal_opt_args_t
        makeMarshalOptArgs() const {
            return makeMarshalOptArgs(delta_ref_agent_0_, delta_ref_agent_1_);
        }

        nixl_marshal_opt_args_t
        makeMarshalOptArgs(void *sender_ref, void *receiver_ref) const {
            auto args = makeMarshalOptArgsForConfig<MarshalConfig>();
            if constexpr (usesDelta) {
                nixlMarshalDeltaOptArgs delta;
                delta.senderRef = reinterpret_cast<std::byte *>(sender_ref);
                delta.receiverRef = reinterpret_cast<std::byte *>(receiver_ref);
                delta.senderMemType = VRAM_SEG;
                delta.receiverMemType = VRAM_SEG;
                delta.elementSize = 1;
                if constexpr (std::is_same_v<MarshalConfig, nixlMarshalDeltaConfig>) {
                    std::get<nixlMarshalDeltaOptArgs>(args) = delta;
                } else {
                    // nixlMarshalCompressConfig with the ANS_DELTA algo.
                    std::get<nixlMarshalCompressOptArgs>(args).delta = delta;
                }
            }
            return args;
        }

        nixl_status_t
        reserveRemainingXferCapacity(nixl_xfer_op_t operation,
                                     const nixl_xfer_dlist_t &local_descs,
                                     const nixl_xfer_dlist_t &remote_descs,
                                     const nixl_service_opt_args_t *extra_params,
                                     std::vector<nixlServiceXferReqH *> &reserved_handles) {
            // A malformed slot layout must not turn this diagnostic loop into an unbounded
            // allocation. The fixtures currently have at most two transfer groups, so this
            // leaves ample headroom while still failing deterministically on a regression.
            constexpr size_t max_capacity_reservations = 64;
            for (size_t reservation = 0; reservation < max_capacity_reservations; ++reservation) {
                nixlServiceXferReqH *request = nullptr;
                const nixl_status_t status = agent_0_->createXferReq(
                    operation, local_descs, remote_descs, "agent_1", request, extra_params);
                if (status != NIXL_SUCCESS) {
                    return request == nullptr ? status : NIXL_ERR_UNKNOWN;
                }
                if (request == nullptr) {
                    return NIXL_ERR_UNKNOWN;
                }
                reserved_handles.push_back(request);
            }
            return NIXL_ERR_UNKNOWN;
        }

        static constexpr auto testTimeout = std::chrono::seconds(30);

        static nixl_status_t
        progressUntilTerminal(nixlServiceAgent &owner,
                              nixlServiceAgent &peer,
                              nixlServiceXferReqH *request) {
            const auto deadline = std::chrono::steady_clock::now() + testTimeout;
            while (std::chrono::steady_clock::now() < deadline) {
                const nixl_status_t status = owner.getXferStatus(request);
                if (status != NIXL_IN_PROG) {
                    return status;
                }

                nixl_notifs_t owner_notifs;
                nixl_status_t progress_status = owner.getNotifs(owner_notifs);
                if (progress_status != NIXL_SUCCESS) {
                    return progress_status;
                }

                const nixl_status_t status_after_owner_progress = owner.getXferStatus(request);
                if (status_after_owner_progress != NIXL_IN_PROG) {
                    return status_after_owner_progress;
                }

                nixl_notifs_t peer_notifs;
                progress_status = peer.getNotifs(peer_notifs);
                if (progress_status != NIXL_SUCCESS) {
                    return progress_status;
                }
                std::this_thread::yield();
            }
            return NIXL_IN_PROG;
        }

        static nixl_status_t
        dispatchServiceNotificationsInOneProgressCall(nixlServiceAgent &sender,
                                                      const std::string &receiver_name,
                                                      nixlServiceAgent &receiver,
                                                      const std::string &sentinel) {
            // A normal notification sent after the service controls is a transport FIFO barrier.
            // Once the base API observes it, every preceding service callback has populated
            // serviceNotifQueue_, but the service override has not dispatched that queue yet.
            nixl_status_t status = sender.genNotif(receiver_name, sentinel);
            if (status != NIXL_SUCCESS) {
                return status;
            }

            const auto deadline = std::chrono::steady_clock::now() + testTimeout;
            while (std::chrono::steady_clock::now() < deadline) {
                nixl_notifs_t notifs;
                status = static_cast<nixlAgent &>(receiver).getNotifs(notifs);
                if (status != NIXL_SUCCESS) {
                    return status;
                }
                const bool received_sentinel =
                    std::any_of(notifs.begin(), notifs.end(), [&](const auto &agent_notifs) {
                        const auto &messages = agent_notifs.second;
                        return std::find(messages.begin(), messages.end(), sentinel) !=
                            messages.end();
                    });
                if (received_sentinel) {
                    nixl_notifs_t service_notifs;
                    return receiver.getNotifs(service_notifs);
                }
                std::this_thread::yield();
            }
            return NIXL_IN_PROG;
        }

        // Fills the first fill_size bytes of buf with a non-constant pattern, cheaply, so the
        // byte-for-byte comparisons below can actually detect corruption/truncation/misplacement
        // - unlike filling every byte via std::uniform_int_distribution, which for these
        // multi-hundred-MB buffers costs multiple seconds of pure host CPU time, dwarfing the
        // real transfer this fixture means to exercise. A single tile is generated the slow way,
        // then replicated with memcpy; the tile size is a prime deliberately not a multiple of
        // any fixture's chunk_size, so corresponding bytes across chunk boundaries still differ.
        // Two buffers filled with different seeds are guaranteed distinct patterns, unlike two
        // buffers drawn from the same rng stream, for tests that need that (e.g. to tell a
        // WRITE's and a READ's data apart if they were ever cross-wired).
        static void
        fillWithRepeatingRandomPattern(std::vector<unsigned char> &buf,
                                       size_t fill_size,
                                       uint32_t seed = 42) {
            constexpr size_t tile_size_max = 1'000'003;
            std::mt19937 rng(seed);
            std::uniform_int_distribution<int> dist(0, 4);
            const size_t tile_size = std::min(tile_size_max, fill_size);
            for (size_t i = 0; i < tile_size; ++i) {
                buf[i] = static_cast<unsigned char>(dist(rng));
            }
            for (size_t off = tile_size; off < fill_size; off += tile_size) {
                std::memcpy(buf.data() + off, buf.data(), std::min(tile_size, fill_size - off));
            }
        }

        // End-to-end marshalled READ, shared by every mode this fixture is instantiated for:
        // agent_1_ is the peer being read from (P), its buffer is prefilled and never written
        // to; agent_0_ is the initiator (I), its buffer starts zeroed and should end up holding
        // a copy of agent_1_'s data.
        void
        runReadValidTest() {
            if (!etcd_valid_) {
                GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
            }
            if (!cuda_valid_) {
                GTEST_SKIP() << "No CUDA device available";
            }

            fillWithRepeatingRandomPattern(src_data_, memSize);
            ASSERT_EQ(cudaMemcpy(mem_agent_1_, src_data_.data(), memSize, cudaMemcpyHostToDevice),
                      cudaSuccess);

            nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG); // agent_0_'s (I's) destination.
            local_xfer_descs.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
            nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG); // agent_1_'s (P's) source.
            remote_xfer_descs.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));

            nixlServiceXferReqH *req_hndl = nullptr;
            nixl_service_opt_args_t extra_params{};
            // sender_ref/receiver_ref are swapped relative to the no-arg makeMarshalOptArgs()'s
            // WRITE-oriented default: for a READ, agent_1_ (P) is the one encoding and agent_0_
            // (I) is the one decoding. A no-op for any mode that doesn't use delta references.
            extra_params.marshalOptArgs =
                makeMarshalOptArgs(delta_ref_agent_1_, delta_ref_agent_0_);
            extra_params.notif = "read_done";
            ASSERT_EQ(agent_0_->createXferReq(NIXL_READ,
                                              local_xfer_descs,
                                              remote_xfer_descs,
                                              "agent_1",
                                              req_hndl,
                                              &extra_params),
                      NIXL_SUCCESS);
            ASSERT_NE(req_hndl, nullptr);
            ASSERT_EQ(agent_0_->getXferStatus(req_hndl), NIXL_ERR_NOT_POSTED);
            ASSERT_EQ(agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

            nixl_notifs_t notifs;
            const auto transfer_deadline = std::chrono::steady_clock::now() + testTimeout;
            while (agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
                ASSERT_LT(std::chrono::steady_clock::now(), transfer_deadline)
                    << "marshalled READ did not complete within the time bound";
                ASSERT_EQ(agent_0_->getNotifs(notifs), NIXL_SUCCESS);
                ASSERT_EQ(notifs.size(), 0);
                // The notification (once delivered) always targets the peer being read from,
                // not the initiator itself - see tryCompleteReadReceive.
                ASSERT_EQ(agent_1_->getNotifs(notifs), NIXL_SUCCESS);
                ASSERT_LE(notifs.size(), 1);
            }
            ASSERT_EQ(agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);

            ASSERT_EQ(cudaMemcpy(dst_data_.data(), mem_agent_0_, memSize, cudaMemcpyDeviceToHost),
                      cudaSuccess);
            ASSERT_EQ(dst_data_, src_data_);

            const auto notif_deadline = std::chrono::steady_clock::now() + testTimeout;
            while (notifs.size() == 0) {
                ASSERT_LT(std::chrono::steady_clock::now(), notif_deadline)
                    << "marshalled READ completion notification was not delivered";
                ASSERT_EQ(agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            }
            ASSERT_EQ(notifs.size(), 1);
            ASSERT_EQ(notifs["agent_0"].size(), 1);
            ASSERT_EQ(notifs["agent_0"].front(), "read_done");

            ASSERT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
        }

        // Posts a single xfer with a single descriptor and reports its telemetry. Both buffers
        // are expected to already hold the payload, so it works for either direction
        void
        postSingleDescXferAndGetTelemetry(nixl_xfer_op_t operation,
                                          const nixl_marshal_opt_args_t &marshal_opt_args,
                                          nixl_xfer_telem_t &telemetry) {
            nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
            local_xfer_descs.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
            nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
            remote_xfer_descs.addDesc(
                nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));

            nixl_service_opt_args_t extra_params{{}, marshal_opt_args};
            nixlServiceXferReqH *req_hndl = nullptr;
            ASSERT_EQ(agent_0_->createXferReq(operation,
                                              local_xfer_descs,
                                              remote_xfer_descs,
                                              "agent_1",
                                              req_hndl,
                                              &extra_params),
                      NIXL_SUCCESS);
            ASSERT_NE(req_hndl, nullptr);
            constexpr auto desc_count = 1;
            constexpr auto time_margin = std::chrono::milliseconds(1);
            const auto xfer_start_t = std::chrono::steady_clock::now();
            const auto post_status = agent_0_->postXferReq(req_hndl, &extra_params);
            ASSERT_TRUE(post_status == NIXL_SUCCESS || post_status == NIXL_IN_PROG);
            nixl_notifs_t notifs;
            while (agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
                ASSERT_EQ(agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            }
            ASSERT_EQ(agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);
            ASSERT_EQ(agent_0_->getXferTelemetry(req_hndl, telemetry), NIXL_SUCCESS);
            EXPECT_EQ(telemetry.descCount, desc_count);
            EXPECT_GT(telemetry.postDuration, chrono_period_us_t(0));
            EXPECT_GE(telemetry.xferDuration, telemetry.postDuration);
            const auto elapsed = std::chrono::steady_clock::now() - xfer_start_t;
            EXPECT_LT(telemetry.xferDuration, elapsed + time_margin);
            ASSERT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
        }
    };

    template<typename Fixture> class nixlServiceTest : public Fixture {};

    constexpr uint32_t nixlServiceTest_max_concurrent_transfers = 1;
    constexpr uint32_t nixlServiceTest_slot_groups = nixlServiceTest_max_concurrent_transfers + 1;

    using nixl_service_staging_f =
        nixlServiceTestF<nixlMarshalStagingConfig,
                         MarshalBackendSizing::slots_per_transfer * nixlServiceTest_slot_groups,
                         128 * 1024>;
    using nixl_service_compress_f =
        nixlServiceTestF<nixlMarshalCompressConfig,
                         MarshalBackendSizing::slots_per_transfer * nixlServiceTest_slot_groups,
                         128 * 1024>;
    using nixl_service_delta_f =
        nixlServiceTestF<nixlMarshalDeltaConfig,
                         MarshalBackendSizing::slots_per_transfer * nixlServiceTest_slot_groups,
                         128 * 1024>;

    using nixl_service_test_types_t =
        ::testing::Types<nixl_service_staging_f, nixl_service_compress_f, nixl_service_delta_f>;

    TYPED_TEST_SUITE(nixlServiceTest, nixl_service_test_types_t);

    template<typename Fixture, nixl_xfer_op_t Op> struct nixlServiceTestWithOp : Fixture {
        static constexpr nixl_xfer_op_t operation = Op;
    };

    template<typename Combined> class nixlServiceBothDirectionsTest : public Combined {};

    using nixl_service_both_directions_types_t =
        ::testing::Types<nixlServiceTestWithOp<nixl_service_staging_f, NIXL_WRITE>,
                         nixlServiceTestWithOp<nixl_service_staging_f, NIXL_READ>,
                         nixlServiceTestWithOp<nixl_service_compress_f, NIXL_WRITE>,
                         nixlServiceTestWithOp<nixl_service_compress_f, NIXL_READ>,
                         nixlServiceTestWithOp<nixl_service_delta_f, NIXL_WRITE>,
                         nixlServiceTestWithOp<nixl_service_delta_f, NIXL_READ>>;

    TYPED_TEST_SUITE(nixlServiceBothDirectionsTest, nixl_service_both_directions_types_t);

    class nixlServiceOrderingTest
        : public nixlServiceTestF<nixlMarshalStagingConfig,
                                  MarshalBackendSizing::slots_per_transfer *
                                      nixlServiceTest_slot_groups,
                                  128 * 1024> {};

    template<typename Fixture> class nixlServiceBidirectionalTest : public Fixture {};

    constexpr uint32_t nixlServiceBidirectionalTest_max_concurrent_transfers = 2;
    constexpr uint32_t nixlServiceBidirectionalTest_slot_groups =
        nixlServiceBidirectionalTest_max_concurrent_transfers + 1;

    using nixl_service_bidirectional_test_types_t =
        ::testing::Types<nixlServiceTestF<nixlMarshalStagingConfig,
                                          MarshalBackendSizing::slots_per_transfer *
                                              nixlServiceBidirectionalTest_slot_groups,
                                          128 * 1024>,
                         nixlServiceTestF<nixlMarshalCompressConfig,
                                          MarshalBackendSizing::slots_per_transfer *
                                              nixlServiceBidirectionalTest_slot_groups,
                                          512 * 1024>,
                         nixlServiceTestF<nixlMarshalDeltaConfig,
                                          MarshalBackendSizing::slots_per_transfer *
                                              nixlServiceBidirectionalTest_slot_groups,
                                          512 * 1024>>;

    TYPED_TEST_SUITE(nixlServiceBidirectionalTest, nixl_service_bidirectional_test_types_t);

    // Compression configured with the ANS_DELTA algo is its own dedicated suite rather than a
    // fourth entry in nixl_service_test_types_t: every other TYPED_TEST bound to nixlServiceTest
    // (e.g. MixedSmallAndLargeDescriptors) would otherwise also run against it, and several
    // of those tests only account for plain nixlMarshalDeltaConfig's single-descriptor
    // restriction, not the equivalent restriction for compression+ANS_DELTA. ANS_DELTA needs
    // extra per-slot workspace beyond plain ANS (see algoWorkspaceOverhead) for the delta
    // kernel's full-payload staging copy; the backend recommendation accounts for it.
    template<typename Fixture> class nixlServiceAnsDeltaTest : public Fixture {};

    constexpr uint32_t nixlServiceAnsDeltaTest_max_concurrent_transfers = 1;
    constexpr uint32_t nixlServiceAnsDeltaTest_slot_groups =
        nixlServiceAnsDeltaTest_max_concurrent_transfers + 1;

    using nixl_service_ans_delta_test_types_t =
        ::testing::Types<nixlServiceTestF<nixlMarshalCompressConfig,
                                          MarshalBackendSizing::slots_per_transfer *
                                              nixlServiceAnsDeltaTest_slot_groups,
                                          512 * 1024,
                                          nixl_marshal_compress_algo_t::ANS_DELTA>>;

    TYPED_TEST_SUITE(nixlServiceAnsDeltaTest, nixl_service_ans_delta_test_types_t);

    constexpr uint32_t nixlServiceMockMarshalTest_max_concurrent_transfers = 1;
    constexpr uint32_t nixlServiceMockMarshalTest_slot_groups =
        nixlServiceMockMarshalTest_max_concurrent_transfers + 1;

    class nixlServiceMockMarshalTest
        : public nixlServiceTestF<nixlMarshalStagingConfig,
                                  MarshalBackendSizing::slots_per_transfer *
                                      nixlServiceMockMarshalTest_slot_groups,
                                  128 * 1024> {
    protected:
        using base_t = nixlServiceTestF<nixlMarshalStagingConfig,
                                        MarshalBackendSizing::slots_per_transfer *
                                            nixlServiceMockMarshalTest_slot_groups,
                                        128 * 1024>;

        void
        SetUp() override {
            mock_ = marshals::mockMarshalBackend::createBackend();
            injectedBackend_ = mock_;
            base_t::SetUp();
        }

        std::shared_ptr<marshals::mockMarshalBackend> mock_;
    };

    struct asymmetricTestParam {
        nixl_mem_t targetType;
        size_t transferSize;
    };

    constexpr uint32_t nixlServiceAsymmetricModeTest_max_concurrent_transfers = 1;
    constexpr uint32_t nixlServiceAsymmetricModeTest_slot_groups =
        nixlServiceAsymmetricModeTest_max_concurrent_transfers + 1;

    class nixlServiceAsymmetricModeTest
        : public nixlServiceTestF<nixlMarshalCompressConfig,
                                  MarshalBackendSizing::slots_per_transfer *
                                      nixlServiceAsymmetricModeTest_slot_groups,
                                  default_chunked_payload_size,
                                  nixl_marshal_compress_algo_t::ANS,
                                  4 * default_chunked_payload_size>,
          public ::testing::WithParamInterface<asymmetricTestParam> {
        using base_t = nixlServiceTestF<nixlMarshalCompressConfig,
                                        MarshalBackendSizing::slots_per_transfer *
                                            nixlServiceAsymmetricModeTest_slot_groups,
                                        default_chunked_payload_size,
                                        nixl_marshal_compress_algo_t::ANS,
                                        4 * default_chunked_payload_size>;

    protected:
        static_assert(memSize <= max_descriptor_size);
        static_assert(memSize % default_chunked_payload_size == 0);
        size_t transferSize_ = 0;

        std::vector<unsigned char> dramTarget_;
        std::unique_ptr<nixl_reg_dlist_t> targetRegDescs_;
        nixlBackendH *targetBackend_ = nullptr;
        int targetFd_ = -1;
        std::string targetPath_;
        bool targetRegistered_ = false;

        void
        SetUp() override {
            transferSize_ = GetParam().transferSize;
            base_t::SetUp();
            if (!etcd_valid_ || !cuda_valid_) {
                return;
            }

            const auto target_type = GetParam().targetType;
            if (target_type == VRAM_SEG) {
                return;
            }

            targetRegDescs_ = std::make_unique<nixl_reg_dlist_t>(target_type);
            if (target_type == DRAM_SEG) {
                dramTarget_.resize(transferSize_);
                targetRegDescs_->addDesc(nixlBlobDesc(
                    reinterpret_cast<uintptr_t>(dramTarget_.data()), dramTarget_.size(), 0));
                ASSERT_EQ(agent_1_->registerMem(*targetRegDescs_), NIXL_SUCCESS);
                targetRegistered_ = true;

                nixl_blob_t remote_md;
                ASSERT_EQ(agent_1_->getLocalMD(remote_md), NIXL_SUCCESS);
                std::string remote_name;
                ASSERT_EQ(agent_0_->loadRemoteMD(remote_md, remote_name), NIXL_SUCCESS);
                ASSERT_EQ(remote_name, "agent_1");
                return;
            }

            ASSERT_EQ(target_type, FILE_SEG);
            nixlServiceAgentConfig cfg;
            cfg.mode = nixlMarshalCompressConfig{};
            cfg.useProgThread = true;
            cfg.captureTelemetry = true;
            // GDS_MT does not support special notification injection.
            cfg.disableServiceNotifCallbacks = true;
            agent_0_ = std::make_unique<testServiceAgent>("ssd_agent", cfg, chunkSize);
            ASSERT_EQ(agent_0_->createBackend("GDS_MT", {}, targetBackend_), NIXL_SUCCESS);
            ASSERT_NE(targetBackend_, nullptr);

            nixl_opt_args_t backend_opts;
            backend_opts.backends = {targetBackend_};
            nixl_reg_dlist_t source_descs(VRAM_SEG);
            source_descs.addDesc(
                nixlBlobDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_, device_id_));
            ASSERT_EQ(agent_0_->registerMem(source_descs, &backend_opts), NIXL_SUCCESS);
            nixl_reg_dlist_t service_descs(VRAM_SEG);
            service_descs.addDesc(nixlBlobDesc(
                reinterpret_cast<uintptr_t>(svc_mem_agent_0_), serviceMemSize(), device_id_));
            ASSERT_EQ(agent_0_->registerServiceMem(service_descs), NIXL_SUCCESS);

            const char *target_dir = std::getenv("NIXL_GDS_TEST_DIR");
            if (target_dir == nullptr || target_dir[0] == '\0') {
                GTEST_SKIP() << "NIXL_GDS_TEST_DIR is not set";
            }
            std::string path = std::string(target_dir) + "/nixl_service_asymmetric_XXXXXX";
            targetFd_ = mkstemp(path.data());
            ASSERT_GE(targetFd_, 0);
            targetPath_ = path;
            ASSERT_EQ(ftruncate(targetFd_, transferSize_), 0);
            targetRegDescs_->addDesc(nixlBlobDesc(0, transferSize_, 0, "rw,direct:" + targetPath_));
            ASSERT_EQ(agent_0_->registerMem(*targetRegDescs_, &backend_opts), NIXL_SUCCESS);
            targetRegistered_ = true;
        }

        void
        TearDown() override {
            // VRAM is owned by base_t; only derived DRAM/FILE registrations reach here.
            if (targetRegistered_) {
                nixl_opt_args_t backend_opts;
                if (targetBackend_ != nullptr) {
                    backend_opts.backends = {targetBackend_};
                }
                nixlAgent *target_agent =
                    GetParam().targetType == DRAM_SEG ? agent_1_.get() : agent_0_.get();
                if (target_agent != nullptr) {
                    (void)target_agent->deregisterMem(
                        *targetRegDescs_, targetBackend_ == nullptr ? nullptr : &backend_opts);
                }
            }
            if (targetFd_ >= 0) {
                (void)close(targetFd_);
            }
            if (!targetPath_.empty()) {
                (void)unlink(targetPath_.c_str());
            }
            base_t::TearDown();
        }

        nixl_xfer_dlist_t
        targetDescs() const {
            if (GetParam().targetType == VRAM_SEG) {
                nixl_xfer_dlist_t descs(VRAM_SEG);
                descs.addDesc(nixlBasicDesc(
                    reinterpret_cast<uintptr_t>(mem_agent_1_), transferSize_, device_id_));
                return descs;
            }
            return targetRegDescs_->trim();
        }

        std::string
        targetAgentName() const {
            return GetParam().targetType == FILE_SEG ? "ssd_agent" : "agent_1";
        }

        nixlServiceAgent &
        targetAgent() const {
            return GetParam().targetType == FILE_SEG ? *agent_0_ : *agent_1_;
        }

        nixl_service_opt_args_t
        makeExtraParams(nixl_marshal_phase_t phase) const {
            nixl_service_opt_args_t opts{};
            opts.marshalOptArgs = nixlMarshalCompressOptArgs{};
            opts.phase = phase;
            if (targetBackend_ != nullptr) {
                opts.backends = {targetBackend_};
            }
            return opts;
        }

        void
        writeTarget(const void *src, size_t size) {
            switch (GetParam().targetType) {
            case DRAM_SEG:
                std::memcpy(dramTarget_.data(), src, size);
                break;
            case VRAM_SEG:
                ASSERT_EQ(cudaMemcpy(mem_agent_1_, src, size, cudaMemcpyHostToDevice), cudaSuccess);
                break;
            case FILE_SEG:
                ASSERT_EQ(pwrite(targetFd_, src, size, 0), static_cast<ssize_t>(size));
                ASSERT_EQ(fsync(targetFd_), 0);
                break;
            default:
                FAIL() << "Unsupported target memory type";
            }
        }

        void
        readTarget(void *dst, size_t size) {
            switch (GetParam().targetType) {
            case DRAM_SEG:
                std::memcpy(dst, dramTarget_.data(), size);
                break;
            case VRAM_SEG:
                ASSERT_EQ(cudaMemcpy(dst, mem_agent_1_, size, cudaMemcpyDeviceToHost), cudaSuccess);
                break;
            case FILE_SEG:
                ASSERT_EQ(fsync(targetFd_), 0);
                ASSERT_EQ(pread(targetFd_, dst, size, 0), static_cast<ssize_t>(size));
                break;
            default:
                FAIL() << "Unsupported target memory type";
            }
        }
    };

    TEST_P(nixlServiceAsymmetricModeTest, ValidatesRequestShapesTypesAndPhases) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        nixl_xfer_dlist_t valid_local(VRAM_SEG);
        valid_local.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_, device_id_));
        const auto valid_remote = targetDescs();
        auto expect_rejected = [&](nixl_xfer_op_t operation,
                                   const nixl_xfer_dlist_t &local_descs,
                                   const nixl_xfer_dlist_t &remote_descs,
                                   nixl_marshal_phase_t phase,
                                   nixl_status_t expected_status) {
            auto extra_params = makeExtraParams(phase);
            nixlServiceXferReqH *req_hndl = nullptr;
            EXPECT_EQ(agent_0_->createXferReq(operation,
                                              local_descs,
                                              remote_descs,
                                              targetAgentName(),
                                              req_hndl,
                                              &extra_params),
                      expected_status);
            EXPECT_EQ(req_hndl, nullptr);
        };

        auto expect_accepted = [&](nixl_xfer_op_t operation,
                                   const nixl_xfer_dlist_t &local_descs,
                                   const nixl_xfer_dlist_t &remote_descs,
                                   nixl_marshal_phase_t phase) {
            auto extra_params = makeExtraParams(phase);
            nixlServiceXferReqH *req_hndl = nullptr;
            ASSERT_EQ(agent_0_->createXferReq(operation,
                                              local_descs,
                                              remote_descs,
                                              targetAgentName(),
                                              req_hndl,
                                              &extra_params),
                      NIXL_SUCCESS);
            ASSERT_NE(req_hndl, nullptr);
            EXPECT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
        };

        nixl_xfer_dlist_t short_local(VRAM_SEG);
        short_local.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_ - 1, device_id_));
        expect_rejected(NIXL_WRITE,
                        short_local,
                        valid_remote,
                        nixl_marshal_phase_t::PRE_TRANSFER,
                        NIXL_ERR_INVALID_PARAM);

        nixl_xfer_dlist_t short_remote(GetParam().targetType);
        auto short_remote_desc = valid_remote[0];
        --short_remote_desc.len;
        short_remote.addDesc(short_remote_desc);
        expect_rejected(NIXL_READ,
                        valid_local,
                        short_remote,
                        nixl_marshal_phase_t::POST_TRANSFER,
                        NIXL_ERR_INVALID_PARAM);

        const size_t oversized_length = max_descriptor_size + default_chunked_payload_size;
        nixl_xfer_dlist_t oversized_local(VRAM_SEG);
        oversized_local.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), oversized_length, device_id_));
        nixl_xfer_dlist_t oversized_remote(GetParam().targetType);
        auto oversized_remote_desc = valid_remote[0];
        oversized_remote_desc.len = oversized_length;
        oversized_remote.addDesc(oversized_remote_desc);
        expect_rejected(NIXL_WRITE,
                        oversized_local,
                        oversized_remote,
                        nixl_marshal_phase_t::PRE_TRANSFER,
                        NIXL_ERR_INVALID_PARAM);

        nixl_xfer_dlist_t multiple_local(VRAM_SEG);
        multiple_local.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_, device_id_));
        multiple_local.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_, device_id_));
        expect_rejected(NIXL_WRITE,
                        multiple_local,
                        valid_remote,
                        nixl_marshal_phase_t::PRE_TRANSFER,
                        NIXL_ERR_INVALID_PARAM);

        nixl_xfer_dlist_t multiple_remote(GetParam().targetType);
        multiple_remote.addDesc(valid_remote[0]);
        multiple_remote.addDesc(valid_remote[0]);
        expect_rejected(NIXL_READ,
                        valid_local,
                        multiple_remote,
                        nixl_marshal_phase_t::POST_TRANSFER,
                        NIXL_ERR_INVALID_PARAM);

        nixl_xfer_dlist_t host_local(DRAM_SEG);
        host_local.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(src_data_.data()), src_data_.size(), 0));
        for (const auto operation : {NIXL_WRITE, NIXL_READ}) {
            const auto phase = operation == NIXL_WRITE ? nixl_marshal_phase_t::PRE_TRANSFER :
                                                         nixl_marshal_phase_t::POST_TRANSFER;
            expect_accepted(operation, valid_local, valid_remote, phase);
        }

        expect_rejected(NIXL_WRITE,
                        host_local,
                        valid_remote,
                        nixl_marshal_phase_t::PRE_TRANSFER,
                        NIXL_ERR_INVALID_PARAM);
        expect_rejected(NIXL_READ,
                        host_local,
                        valid_remote,
                        nixl_marshal_phase_t::POST_TRANSFER,
                        NIXL_ERR_INVALID_PARAM);

        expect_rejected(NIXL_WRITE,
                        valid_local,
                        valid_remote,
                        nixl_marshal_phase_t::POST_TRANSFER,
                        NIXL_ERR_NOT_SUPPORTED);
        expect_rejected(NIXL_READ,
                        valid_local,
                        valid_remote,
                        nixl_marshal_phase_t::PRE_TRANSFER,
                        NIXL_ERR_NOT_SUPPORTED);
    }

    TEST_P(nixlServiceAsymmetricModeTest, WriteReturnsSuccessOnCompletion) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        nixl_xfer_dlist_t local_descs(VRAM_SEG);
        local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_, device_id_));
        const auto remote_descs = targetDescs();
        auto extra_params = makeExtraParams(nixl_marshal_phase_t::PRE_TRANSFER);
        nixlServiceXferReqH *req_hndl = nullptr;
        ASSERT_EQ(
            agent_0_->createXferReq(
                NIXL_WRITE, local_descs, remote_descs, targetAgentName(), req_hndl, &extra_params),
            NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        ASSERT_EQ(agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

        nixl_status_t status = NIXL_IN_PROG;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (status == NIXL_IN_PROG) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline);
            status = agent_0_->getXferStatus(req_hndl);
        }
        EXPECT_EQ(status, NIXL_SUCCESS);
        ASSERT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    }

    TEST_P(nixlServiceAsymmetricModeTest, RejectsMalformedPostHeadersAndActiveRepost) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        nixl_xfer_dlist_t local_descs(VRAM_SEG);
        local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_, device_id_));
        const auto remote_descs = targetDescs();
        using header_t = std::array<uint64_t, max_chunks_per_desc>;
        std::array<header_t, 5> malformed_headers{};
        const size_t active_chunk_count = 1 + (transferSize_ - 1) / default_chunked_payload_size;
        ASSERT_LT(active_chunk_count, max_chunks_per_desc);
        for (size_t case_index = 1; case_index < malformed_headers.size(); ++case_index) {
            std::fill_n(malformed_headers[case_index].begin(), active_chunk_count, 1);
        }
        auto &oversized_entry_header = malformed_headers[1];
        oversized_entry_header[0] = std::numeric_limits<uint64_t>::max();
        auto &out_of_bounds_header = malformed_headers[2];
        std::fill_n(out_of_bounds_header.begin(), active_chunk_count, default_chunked_payload_size);
        auto &missing_active_entry_header = malformed_headers[3];
        missing_active_entry_header[active_chunk_count - 1] = 0;
        auto &nonzero_inactive_entry_header = malformed_headers[4];
        nonzero_inactive_entry_header[active_chunk_count] = 1;

        for (size_t case_index = 0; case_index < malformed_headers.size(); ++case_index) {
            SCOPED_TRACE(case_index);
            writeTarget(malformed_headers[case_index].data(), CompressedObjectLayout::header_size);

            auto extra_params = makeExtraParams(nixl_marshal_phase_t::POST_TRANSFER);
            nixlServiceXferReqH *req_hndl = nullptr;
            ASSERT_EQ(agent_0_->createXferReq(NIXL_READ,
                                              local_descs,
                                              remote_descs,
                                              targetAgentName(),
                                              req_hndl,
                                              &extra_params),
                      NIXL_SUCCESS);
            ASSERT_NE(req_hndl, nullptr);
            EXPECT_EQ(agent_0_->getXferStatus(req_hndl), NIXL_ERR_NOT_POSTED);
            ASSERT_EQ(agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);
            EXPECT_EQ(agent_0_->postXferReq(req_hndl, &extra_params), NIXL_ERR_REPOST_ACTIVE);

            nixl_status_t status = NIXL_IN_PROG;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (status == NIXL_IN_PROG) {
                ASSERT_LT(std::chrono::steady_clock::now(), deadline);
                status = agent_0_->getXferStatus(req_hndl);
            }
            EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);
            ASSERT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
        }
    }

    TEST_P(nixlServiceAsymmetricModeTest, PreWriteThenPostReadPreservesData) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        fillWithRepeatingRandomPattern(src_data_, transferSize_);
        ASSERT_EQ(cudaMemcpy(mem_agent_0_, src_data_.data(), transferSize_, cudaMemcpyHostToDevice),
                  cudaSuccess);

        nixl_xfer_dlist_t local_descs(VRAM_SEG);
        local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_, device_id_));
        const auto remote_descs = targetDescs();
        auto extra_params = makeExtraParams(nixl_marshal_phase_t::PRE_TRANSFER);
        nixlServiceXferReqH *req_hndl = nullptr;
        ASSERT_EQ(
            agent_0_->createXferReq(
                NIXL_WRITE, local_descs, remote_descs, targetAgentName(), req_hndl, &extra_params),
            NIXL_SUCCESS);
        ASSERT_EQ(agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

        nixl_status_t status = NIXL_IN_PROG;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (status == NIXL_IN_PROG) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline);
            status = agent_0_->getXferStatus(req_hndl);
        }
        ASSERT_EQ(status, NIXL_SUCCESS);
        ASSERT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);

        using header_t = std::array<uint64_t, max_chunks_per_desc>;
        header_t persisted_header{};
        readTarget(persisted_header.data(), CompressedObjectLayout::header_size);
        const size_t active_chunk_count = 1 + (transferSize_ - 1) / default_chunked_payload_size;
        for (size_t chunk_index = 0; chunk_index < active_chunk_count; ++chunk_index) {
            EXPECT_NE(persisted_header[chunk_index], 0);
        }
        for (size_t chunk_index = active_chunk_count; chunk_index < max_chunks_per_desc;
             ++chunk_index) {
            EXPECT_EQ(persisted_header[chunk_index], 0);
        }

        ASSERT_EQ(cudaMemset(mem_agent_0_, 0, transferSize_), cudaSuccess);
        extra_params.phase = nixl_marshal_phase_t::POST_TRANSFER;
        extra_params.notif.reset();
        req_hndl = nullptr;
        ASSERT_EQ(
            agent_0_->createXferReq(
                NIXL_READ, local_descs, remote_descs, targetAgentName(), req_hndl, &extra_params),
            NIXL_SUCCESS);
        if (GetParam().targetType != FILE_SEG) {
            extra_params.notif = "asymmetric_post_done";
        }
        ASSERT_EQ(agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

        status = NIXL_IN_PROG;
        deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (status == NIXL_IN_PROG) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline);
            status = agent_0_->getXferStatus(req_hndl);
        }
        EXPECT_EQ(status, NIXL_SUCCESS);
        ASSERT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);

        ASSERT_EQ(cudaMemcpy(dst_data_.data(), mem_agent_0_, transferSize_, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(dst_data_, src_data_);

        if (GetParam().targetType != FILE_SEG) {
            nixl_notifs_t notifs;
            deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (notifs.empty()) {
                ASSERT_LT(std::chrono::steady_clock::now(), deadline);
                ASSERT_EQ(targetAgent().getNotifs(notifs), NIXL_SUCCESS);
            }
            ASSERT_EQ(notifs.size(), 1);
            ASSERT_EQ(notifs["agent_0"].size(), 1);
            EXPECT_EQ(notifs["agent_0"].front(), "asymmetric_post_done");
        }
    }

    TEST_P(nixlServiceAsymmetricModeTest, XferTelemetry) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        fillWithRepeatingRandomPattern(src_data_, transferSize_);
        ASSERT_EQ(cudaMemcpy(mem_agent_0_, src_data_.data(), transferSize_, cudaMemcpyHostToDevice),
                  cudaSuccess);

        nixl_xfer_dlist_t local_descs(VRAM_SEG);
        local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), transferSize_, device_id_));
        const auto remote_descs = targetDescs();
        auto run_xfer = [&](nixl_xfer_op_t operation, nixl_marshal_phase_t phase) {
            SCOPED_TRACE(operation);
            size_t total_bytes = 0;
            [&] {
                if (operation == NIXL_READ) {
                    ASSERT_EQ(cudaMemset(mem_agent_0_, 0, transferSize_), cudaSuccess);
                }

                auto extra_params = makeExtraParams(phase);
                nixlServiceXferReqH *req_hndl = nullptr;
                ASSERT_EQ(agent_0_->createXferReq(operation,
                                                  local_descs,
                                                  remote_descs,
                                                  targetAgentName(),
                                                  req_hndl,
                                                  &extra_params),
                          NIXL_SUCCESS);
                ASSERT_NE(req_hndl, nullptr);
                ASSERT_EQ(agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

                nixl_status_t status = NIXL_IN_PROG;
                // 10 seconds for one xfer should be enough.
                // TODO-Eyal: remove magic number TOs from this file, make them smaller than 60.
                const auto constexpr timeout_seconds = 10;
                const auto deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
                while (status == NIXL_IN_PROG) {
                    ASSERT_LT(std::chrono::steady_clock::now(), deadline);
                    status = agent_0_->getXferStatus(req_hndl);
                }
                ASSERT_EQ(status, NIXL_SUCCESS);

                nixl_xfer_telem_t telemetry{};
                ASSERT_EQ(agent_0_->getXferTelemetry(req_hndl, telemetry), NIXL_SUCCESS);
                EXPECT_EQ(telemetry.descCount, 1);
                EXPECT_GT(telemetry.postDuration, chrono_period_us_t(0));
                EXPECT_GE(telemetry.xferDuration, telemetry.postDuration);
                EXPECT_GT(telemetry.totalBytes, 0);
                total_bytes = telemetry.totalBytes;

                ASSERT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
            }();
            return total_bytes;
        };

        const auto write_total_bytes = run_xfer(NIXL_WRITE, nixl_marshal_phase_t::PRE_TRANSFER);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        const auto read_total_bytes = run_xfer(NIXL_READ, nixl_marshal_phase_t::POST_TRANSFER);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        EXPECT_EQ(read_total_bytes, write_total_bytes);
    }

    INSTANTIATE_TEST_SUITE_P(
        TargetMemoryTypes,
        nixlServiceAsymmetricModeTest,
        ::testing::Values(asymmetricTestParam{DRAM_SEG, 4 * default_chunked_payload_size},
                          asymmetricTestParam{DRAM_SEG, 4 * default_chunked_payload_size - 2},
                          asymmetricTestParam{VRAM_SEG, 4 * default_chunked_payload_size},
                          asymmetricTestParam{VRAM_SEG, 4 * default_chunked_payload_size - 2},
                          asymmetricTestParam{FILE_SEG, 4 * default_chunked_payload_size},
                          asymmetricTestParam{FILE_SEG, 4 * default_chunked_payload_size - 2}),
        [](const ::testing::TestParamInfo<asymmetricTestParam> &info) {
            const auto size_name =
                info.param.transferSize % default_chunked_payload_size == 0 ? "Whole" : "Partial";
            return nixlEnumStrings::memTypeStr(info.param.targetType) + "_" + size_name;
        });

    TYPED_TEST(nixlServiceTest, CreateAgentValid) {
        ASSERT_NE(this->agent_0_, nullptr);
        ASSERT_NE(this->agent_1_, nullptr);
    }

    TEST_F(nixlServiceMockMarshalTest, InjectedBackendRunsOnWrite) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        EXPECT_CALL(*mock_, outboundProcessSlot(testing::_, testing::_)).Times(testing::AtLeast(1));
        EXPECT_CALL(*mock_, inboundProcessSlot(testing::_, testing::_, testing::_))
            .Times(testing::AtLeast(1));

        fillWithRepeatingRandomPattern(src_data_, memSize);
        ASSERT_EQ(cudaMemcpy(mem_agent_0_, src_data_.data(), memSize, cudaMemcpyHostToDevice),
                  cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));

        nixlServiceXferReqH *req_hndl = nullptr;
        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs = makeMarshalOptArgs();
        ASSERT_EQ(agent_0_->createXferReq(NIXL_WRITE,
                                          local_xfer_descs,
                                          remote_xfer_descs,
                                          "agent_1",
                                          req_hndl,
                                          &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        const auto post_status = agent_0_->postXferReq(req_hndl, &extra_params);
        ASSERT_TRUE(post_status == NIXL_SUCCESS || post_status == NIXL_IN_PROG);

        ASSERT_EQ(progressUntilTerminal(*agent_0_, *agent_1_, req_hndl), NIXL_SUCCESS)
            << "injected-backend WRITE did not complete within the time bound";

        ASSERT_EQ(cudaMemcpy(dst_data_.data(), mem_agent_1_, memSize, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(dst_data_, src_data_);

        ASSERT_EQ(agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    }

    // Marshalled (non-direct-split) NIXL_READ is now supported (see createReadReceiveXfer),
    // but a loopback marshalled READ (remote_agent == self) remains explicitly rejected
    // until the same agent both encoding and decoding into itself is specifically tested.
    // memSize (1024 * chunkSize) exceeds direct_desc_threshold (64 MB) for every fixture
    // type below, so a single memSize-sized descriptor always takes the marshal split path.
    TYPED_TEST(nixlServiceTest, LoopbackMarshalReadRejected) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        std::vector<unsigned char> zeros(this->memSize, 0);
        ASSERT_EQ(
            cudaMemcpy(this->mem_agent_1_, zeros.data(), this->memSize, cudaMemcpyHostToDevice),
            cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), this->memSize, this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), this->memSize, this->device_id_));

        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs = this->makeMarshalOptArgs();
        nixlServiceXferReqH *req_hndl = nullptr;
        // "agent_0" is agent_0_'s own name (see the fixture's SetUp) - a genuine loopback.
        ASSERT_EQ(
            this->agent_0_->createXferReq(
                NIXL_READ, local_xfer_descs, remote_xfer_descs, "agent_0", req_hndl, &extra_params),
            NIXL_ERR_NOT_SUPPORTED);
        ASSERT_EQ(req_hndl, nullptr);

        // Neither side's buffer should have been touched by the rejected call.
        std::vector<unsigned char> dst_check(this->memSize);
        ASSERT_EQ(
            cudaMemcpy(dst_check.data(), this->mem_agent_0_, this->memSize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(dst_check, zeros);
        ASSERT_EQ(
            cudaMemcpy(dst_check.data(), this->mem_agent_1_, this->memSize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(dst_check, zeros);
    }

    // Covers staging, plain-ANS compression, and plain delta - one descriptor per side, which
    // satisfies delta's single-descriptor restriction. See runReadValidTest()'s definition.
    TYPED_TEST(nixlServiceTest, ReadValid) {
        this->runReadValidTest();
    }

    TYPED_TEST(nixlServiceBothDirectionsTest, CreateXferRejectsTooBigDescriptor) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        nixl_xfer_dlist_t local_descs(VRAM_SEG);
        local_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_0_),
                                          max_descriptor_size + 1,
                                          this->device_id_));
        nixl_xfer_dlist_t remote_descs(VRAM_SEG);
        remote_descs.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_1_),
                                           max_descriptor_size + 1,
                                           this->device_id_));

        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs = (TypeParam::operation == NIXL_READ) ?
            this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_) :
            this->makeMarshalOptArgs();

        nixlServiceXferReqH *req_hndl = nullptr;
        EXPECT_EQ(this->agent_0_->createXferReq(TypeParam::operation,
                                                local_descs,
                                                remote_descs,
                                                "agent_1",
                                                req_hndl,
                                                &extra_params),
                  NIXL_ERR_INVALID_PARAM);
        EXPECT_EQ(req_hndl, nullptr);
    }

    // Compression configured with the ANS_DELTA algo, in its own dedicated suite (see
    // nixlServiceAnsDeltaTest's declaration for why).
    TYPED_TEST(nixlServiceAnsDeltaTest, ReadValid) {
        this->runReadValidTest();
    }

    // Regression for a notif-presence bug: postXferReq's READ branch used to unconditionally
    // overwrite the receive context's notification with postXferReq's own extra_params, even
    // when that call omitted one entirely - silently discarding a notification that was only
    // set at createXferReq() time. This sets .notif on a separate object passed only to
    // createXferReq, then calls postXferReq(req_hndl) with no extra_params at all, mirroring
    // the base nixlAgent::postXferReq convention of carrying the create-time value over unless
    // post time explicitly overrides it (see postXferReq's READ branch).
    TYPED_TEST(nixlServiceTest, ReadNotifPersistsFromCreateWhenPostOmitsIt) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        std::vector<unsigned char> pattern(this->memSize, 0x5A);
        ASSERT_EQ(
            cudaMemcpy(this->mem_agent_1_, pattern.data(), this->memSize, cudaMemcpyHostToDevice),
            cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), this->memSize, this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), this->memSize, this->device_id_));

        nixlServiceXferReqH *req_hndl = nullptr;
        nixl_service_opt_args_t create_params{};
        create_params.marshalOptArgs =
            this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_);
        create_params.notif = "create_time_notif";
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_READ,
                                                local_xfer_descs,
                                                remote_xfer_descs,
                                                "agent_1",
                                                req_hndl,
                                                &create_params),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        // No extra_params at all here - unlike every other READ test, which reuses the same
        // populated extra_params object for both calls and so never exercises this path.
        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl), NIXL_IN_PROG);

        nixl_notifs_t notifs;
        while (this->agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_LE(notifs.size(), 1);
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);

        while (notifs.size() == 0) {
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
        }
        ASSERT_EQ(notifs.size(), 1);
        ASSERT_EQ(notifs["agent_0"].size(), 1);
        ASSERT_EQ(notifs["agent_0"].front(), "create_time_notif");

        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    }

    // TODO-Eyal: add TOs.
    TYPED_TEST(nixlServiceTest, PostXferReq) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        std::mt19937 rng(42);
        // TODO-Roee: change to (0, 255) once fallback is implemented.
        std::uniform_int_distribution<int> dist(0, 4);
        for (auto &v : this->src_data_) {
            v = static_cast<unsigned char>(dist(rng));
        }
        ASSERT_EQ(
            cudaMemcpy(
                this->mem_agent_0_, this->src_data_.data(), this->memSize, cudaMemcpyHostToDevice),
            cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), this->memSize, this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), this->memSize, this->device_id_));

        nixlServiceXferReqH *req_hndl = nullptr;
        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs = this->makeMarshalOptArgs();
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_WRITE,
                                                local_xfer_descs,
                                                remote_xfer_descs,
                                                "agent_1",
                                                req_hndl,
                                                &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl), NIXL_ERR_NOT_POSTED);
        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl), NIXL_SUCCESS);
        nixl_notifs_t notifs;
        while (this->agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);
        nixlServiceXferReqH *req_hndl_1 = nullptr;
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_WRITE,
                                                local_xfer_descs,
                                                remote_xfer_descs,
                                                "agent_1",
                                                req_hndl_1,
                                                &extra_params),
                  NIXL_SUCCESS); // the first request is done so we have free slots.
        ASSERT_NE(req_hndl_1, nullptr);
        ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
        ASSERT_EQ(notifs.size(), 0);
        ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
        ASSERT_EQ(notifs.size(), 0);
        ASSERT_EQ(
            cudaMemcpy(
                this->dst_data_.data(), this->mem_agent_1_, this->memSize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(this->dst_data_, this->src_data_);
        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl_1), NIXL_SUCCESS);
        for (int i = 0; i < 10; i++) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
        }
    }

    // Validate getXferTelemetry() on a completed transfer, in both directions
    TYPED_TEST(nixlServiceTest, XferTelemetry) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        this->fillWithRepeatingRandomPattern(this->src_data_, this->memSize);
        ASSERT_EQ(
            cudaMemcpy(
                this->mem_agent_0_, this->src_data_.data(), this->memSize, cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                this->mem_agent_1_, this->src_data_.data(), this->memSize, cudaMemcpyHostToDevice),
            cudaSuccess);
        nixl_xfer_telem_t telemetry{};
        for (const auto operation : {NIXL_WRITE, NIXL_READ}) {
            const auto marshal_opt_args = (operation == NIXL_READ) ?
                this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_) :
                this->makeMarshalOptArgs();
            this->postSingleDescXferAndGetTelemetry(operation, marshal_opt_args, telemetry);
            if constexpr (std::is_same_v<typename TestFixture::marshal_config_t,
                                         nixlMarshalStagingConfig>) {
                EXPECT_EQ(telemetry.totalBytes, this->memSize);
            } else {
                EXPECT_GT(telemetry.totalBytes, 0);
            }

            this->postSingleDescXferAndGetTelemetry(
                operation, nixlMarshalDirectOptArgs{}, telemetry);
            EXPECT_EQ(telemetry.totalBytes, this->memSize);
        }
    }

    TYPED_TEST(nixlServiceTest, MixedSmallAndLargeDescriptors) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }
        if constexpr (std::is_same_v<typename TestFixture::marshal_config_t,
                                     nixlMarshalDeltaConfig>) {
            GTEST_SKIP() << "Delta mode does not support multiple descriptors";
        }

        constexpr size_t small_desc_size = 1024;
        constexpr size_t large_desc_size = 16 * 1024 * 1024 + 1024;
        constexpr size_t transfer_size = small_desc_size + large_desc_size;
        static_assert(large_desc_size > 16 * 1024 * 1024); // direct_desc_threshold
        static_assert(small_desc_size <= 16 * 1024 * 1024); // direct_desc_threshold
        ASSERT_LE(transfer_size, this->memSize);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 4);
        std::vector<unsigned char> src_data(transfer_size);
        std::vector<unsigned char> dst_data(transfer_size, 0);
        for (auto &v : src_data) {
            v = static_cast<unsigned char>(dist(rng));
        }
        ASSERT_EQ(
            cudaMemcpy(this->mem_agent_0_, src_data.data(), transfer_size, cudaMemcpyHostToDevice),
            cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), small_desc_size, this->device_id_));
        local_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_0_) + small_desc_size,
                          large_desc_size,
                          this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), small_desc_size, this->device_id_));
        remote_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_1_) + small_desc_size,
                          large_desc_size,
                          this->device_id_));

        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs = this->makeMarshalOptArgs();
        nixlServiceXferReqH *req_hndl = nullptr;
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_WRITE,
                                                local_xfer_descs,
                                                remote_xfer_descs,
                                                "agent_1",
                                                req_hndl,
                                                &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl, &extra_params), NIXL_SUCCESS);

        nixl_notifs_t notifs;
        while (this->agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);
        ASSERT_EQ(
            cudaMemcpy(dst_data.data(), this->mem_agent_1_, transfer_size, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(dst_data, src_data);
        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    }

    // READ counterpart of MixedSmallAndLargeDescriptors: a single READ whose two descriptors span
    // both the direct (base NIXL_READ) and marshal (RREQ/RPOSTED/RRSLOT) paths. Unlike that WRITE
    // test, large_desc_size here is sized against the real direct_desc_threshold (64 MB) rather
    // than the stale 16 MB the WRITE test still (incorrectly, but out of scope to fix here) uses.
    // Verifies the two children's completion aggregates correctly regardless of which finishes
    // first, with exactly one user notification delivered only once both are done.
    TYPED_TEST(nixlServiceTest, ReadMixed) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }
        if constexpr (std::is_same_v<typename TestFixture::marshal_config_t,
                                     nixlMarshalDeltaConfig>) {
            GTEST_SKIP() << "Delta mode does not support multiple descriptors";
        }

        constexpr size_t small_desc_size = 1024;
        constexpr size_t large_desc_size = 64 * 1024 * 1024 + 1024;
        constexpr size_t transfer_size = small_desc_size + large_desc_size;
        static_assert(large_desc_size > 64 * 1024 * 1024); // direct_desc_threshold
        static_assert(small_desc_size <= 64 * 1024 * 1024); // direct_desc_threshold
        ASSERT_LE(transfer_size, this->memSize);

        std::vector<unsigned char> src_data(transfer_size);
        std::vector<unsigned char> dst_data(transfer_size, 0);
        this->fillWithRepeatingRandomPattern(src_data, transfer_size);
        ASSERT_EQ(
            cudaMemcpy(this->mem_agent_1_, src_data.data(), transfer_size, cudaMemcpyHostToDevice),
            cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG); // agent_0_'s (I's) destination.
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), small_desc_size, this->device_id_));
        local_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_0_) + small_desc_size,
                          large_desc_size,
                          this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG); // agent_1_'s (P's) source.
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), small_desc_size, this->device_id_));
        remote_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_1_) + small_desc_size,
                          large_desc_size,
                          this->device_id_));

        nixlServiceXferReqH *req_hndl = nullptr;
        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs =
            this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_);
        extra_params.notif = "read_mixed_done";
        ASSERT_EQ(
            this->agent_0_->createXferReq(
                NIXL_READ, local_xfer_descs, remote_xfer_descs, "agent_1", req_hndl, &extra_params),
            NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

        nixl_notifs_t notifs;
        while (this->agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_LE(notifs.size(), 1);
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);

        ASSERT_EQ(
            cudaMemcpy(dst_data.data(), this->mem_agent_0_, transfer_size, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(dst_data, src_data);

        while (notifs.size() == 0) {
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
        }
        ASSERT_EQ(notifs.size(), 1);
        ASSERT_EQ(notifs["agent_0"].size(), 1);
        ASSERT_EQ(notifs["agent_0"].front(), "read_mixed_done");

        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    }

    // Regression test for a mixed READ's double-notify bug: the direct child used to carry
    // over the create-time notif and fire it as soon as its own (small) sub-transfer completed,
    // in addition to the real, single delivery from tryCompleteReadReceive once both sub-parts
    // are done. Same two-descriptor setup as ReadMixed, but the notif is captured only via
    // create_params and postXferReq is called bare - the "create-time-only, bare post" pattern
    // from ReadNotifPersistsFromCreateWhenPostOmitsIt, applied to a mixed (not marshal-only)
    // transfer, where the bug actually lived.
    TYPED_TEST(nixlServiceTest, ReadMixedNotifPersistsFromCreateWhenPostOmitsIt) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }
        if constexpr (std::is_same_v<typename TestFixture::marshal_config_t,
                                     nixlMarshalDeltaConfig>) {
            GTEST_SKIP() << "Delta mode does not support multiple descriptors";
        }

        constexpr size_t small_desc_size = 1024;
        constexpr size_t large_desc_size = 64 * 1024 * 1024 + 1024;
        constexpr size_t transfer_size = small_desc_size + large_desc_size;
        static_assert(large_desc_size > 64 * 1024 * 1024); // direct_desc_threshold
        static_assert(small_desc_size <= 64 * 1024 * 1024); // direct_desc_threshold
        ASSERT_LE(transfer_size, this->memSize);

        std::vector<unsigned char> src_data(transfer_size);
        std::vector<unsigned char> dst_data(transfer_size, 0);
        this->fillWithRepeatingRandomPattern(src_data, transfer_size);
        ASSERT_EQ(
            cudaMemcpy(this->mem_agent_1_, src_data.data(), transfer_size, cudaMemcpyHostToDevice),
            cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), small_desc_size, this->device_id_));
        local_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_0_) + small_desc_size,
                          large_desc_size,
                          this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), small_desc_size, this->device_id_));
        remote_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_1_) + small_desc_size,
                          large_desc_size,
                          this->device_id_));

        nixlServiceXferReqH *req_hndl = nullptr;
        nixl_service_opt_args_t create_params{};
        create_params.marshalOptArgs =
            this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_);
        create_params.notif = "read_mixed_create_only";
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_READ,
                                                local_xfer_descs,
                                                remote_xfer_descs,
                                                "agent_1",
                                                req_hndl,
                                                &create_params),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        // Bare post: extra_params carries no notif of its own, so the only one in play is the
        // create-time one. Before the fix, the direct child would still carry it (from
        // createXferReq's own extra_params) and fire it as soon as it completed.
        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl), NIXL_IN_PROG);

        nixl_notifs_t notifs;
        while (this->agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);

        ASSERT_EQ(
            cudaMemcpy(dst_data.data(), this->mem_agent_0_, transfer_size, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(dst_data, src_data);

        // Both sub-parts are done by now, so any notification - the single correct delivery, or,
        // if the bug regresses, also the direct child's early spurious one - has already been
        // sent; this only waits for it to become retrievable, the same settle margin
        // ReadCancelInEverySlotState uses.
        for (int i = 0; i < 10; ++i) {
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ASSERT_EQ(notifs.size(), 1);
        ASSERT_EQ(notifs["agent_0"].size(), 1);
        ASSERT_EQ(notifs["agent_0"].front(), "read_mixed_create_only");

        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    }

    // Regression test for the getXferStatus-driven mixed-READ completion gap: before the fix,
    // a mixed READ's direct child lived on the caller-owned outer handle, unreachable from
    // progressService(), so only getXferStatus() (never getNotifs() alone) could advance it to
    // completion. Same two-descriptor setup as ReadMixed, but the wait loop below deliberately
    // calls only getNotifs() on both agents - never getXferStatus() - to prove the transfer (and
    // its single notification) completes without it. Bounded by a wall-clock deadline so a
    // regression here fails cleanly instead of hanging the whole suite.
    TYPED_TEST(nixlServiceTest, ReadMixedCompletesViaGetNotifsAlone) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }
        if constexpr (std::is_same_v<typename TestFixture::marshal_config_t,
                                     nixlMarshalDeltaConfig>) {
            GTEST_SKIP() << "Delta mode does not support multiple descriptors";
        }

        constexpr size_t small_desc_size = 1024;
        constexpr size_t large_desc_size = 64 * 1024 * 1024 + 1024;
        constexpr size_t transfer_size = small_desc_size + large_desc_size;
        static_assert(large_desc_size > 64 * 1024 * 1024); // direct_desc_threshold
        static_assert(small_desc_size <= 64 * 1024 * 1024); // direct_desc_threshold
        ASSERT_LE(transfer_size, this->memSize);

        std::vector<unsigned char> src_data(transfer_size);
        std::vector<unsigned char> dst_data(transfer_size, 0);
        this->fillWithRepeatingRandomPattern(src_data, transfer_size);
        ASSERT_EQ(
            cudaMemcpy(this->mem_agent_1_, src_data.data(), transfer_size, cudaMemcpyHostToDevice),
            cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG); // agent_0_'s (I's) destination.
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), small_desc_size, this->device_id_));
        local_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_0_) + small_desc_size,
                          large_desc_size,
                          this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG); // agent_1_'s (P's) source.
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), small_desc_size, this->device_id_));
        remote_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(this->mem_agent_1_) + small_desc_size,
                          large_desc_size,
                          this->device_id_));

        nixlServiceXferReqH *req_hndl = nullptr;
        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs =
            this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_);
        extra_params.notif = "read_mixed_getnotifs_done";
        ASSERT_EQ(
            this->agent_0_->createXferReq(
                NIXL_READ, local_xfer_descs, remote_xfer_descs, "agent_1", req_hndl, &extra_params),
            NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

        // Deliberately no getXferStatus() call in this loop: only getNotifs() on both agents may
        // drive progressService(). The READ's notification always addresses remoteAgent
        // (agent_1 here - see tryCompleteReadReceive), so that is where it must land.
        nixl_notifs_t self_notifs;
        nixl_notifs_t peer_notifs;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (peer_notifs.size() == 0) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline)
                << "mixed READ did not complete via getNotifs() alone within the time bound";
            ASSERT_EQ(this->agent_0_->getNotifs(self_notifs), NIXL_SUCCESS);
            ASSERT_EQ(self_notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(peer_notifs), NIXL_SUCCESS);
        }

        ASSERT_EQ(
            cudaMemcpy(dst_data.data(), this->mem_agent_0_, transfer_size, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(dst_data, src_data);

        ASSERT_EQ(peer_notifs.size(), 1);
        ASSERT_EQ(peer_notifs["agent_0"].size(), 1);
        ASSERT_EQ(peer_notifs["agent_0"].front(), "read_mixed_getnotifs_done");

        // Completion was already observed via getNotifs() alone above; this single check just
        // confirms the state machine (not merely the notification) agrees.
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);

        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    }

    TYPED_TEST(nixlServiceTest, MidTransferRelease) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), this->memSize, this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), this->memSize, this->device_id_));

        nixlServiceXferReqH *req_hndl_0 = nullptr;
        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs = this->makeMarshalOptArgs();
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_WRITE,
                                                local_xfer_descs,
                                                remote_xfer_descs,
                                                "agent_1",
                                                req_hndl_0,
                                                &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl_0, nullptr);

        extra_params.notif = "first_req";

        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl_0, &extra_params), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl_0), NIXL_IN_PROG);
        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl_0),
                  NIXL_SUCCESS); // Mid transfer release.
        nixl_notifs_t notifs;
        // Force RTS followed by DELETE through one receiver progress tick. FIFO dispatch must
        // create the inbound state before cancelling it; type-based reordering used to assert in
        // handleDelete because the deferred RTS had not created that state yet.
        ASSERT_EQ(this->dispatchServiceNotificationsInOneProgressCall(
                      *this->agent_0_, "agent_1", *this->agent_1_, "rts-delete-ready"),
                  NIXL_SUCCESS);
        // Consume the now-stale CTS generated while handling RTS. The sender already released the
        // request, so handleCTS must ignore it.
        ASSERT_EQ(this->dispatchServiceNotificationsInOneProgressCall(
                      *this->agent_1_, "agent_0", *this->agent_0_, "stale-cts-ready"),
                  NIXL_SUCCESS);
        nixlServiceXferReqH *req_hndl_1 = nullptr;
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_WRITE,
                                                local_xfer_descs,
                                                remote_xfer_descs,
                                                "agent_1",
                                                req_hndl_1,
                                                &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl_1, nullptr);
        extra_params.notif = "second_req";
        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl_1, &extra_params), NIXL_SUCCESS);
        const auto transfer_deadline = std::chrono::steady_clock::now() + this->testTimeout;
        while (this->agent_0_->getXferStatus(req_hndl_1) == NIXL_IN_PROG) {
            ASSERT_LT(std::chrono::steady_clock::now(), transfer_deadline)
                << "replacement WRITE did not complete within the time bound";
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_LE(notifs.size(), 1);
            if (notifs.size() == 1) {
                ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl_1), NIXL_SUCCESS);
            }
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl_1), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl_1), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
        ASSERT_EQ(notifs.count("agent_1"), 0);
        const auto notif_deadline = std::chrono::steady_clock::now() + this->testTimeout;
        while (notifs.size() == 0) {
            ASSERT_LT(std::chrono::steady_clock::now(), notif_deadline)
                << "replacement WRITE completion notification was not delivered";
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
        }
        ASSERT_EQ(notifs.size(), 1);
        ASSERT_EQ(notifs["agent_0"].size(), 1);
        ASSERT_EQ(notifs["agent_0"].front(), "second_req");
    }

    TEST_F(nixlServiceOrderingTest, WriteCompletionThenReplacementAdmissionSameProgressTick) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        fillWithRepeatingRandomPattern(src_data_, memSize);
        ASSERT_EQ(cudaMemcpy(mem_agent_0_, src_data_.data(), memSize, cudaMemcpyHostToDevice),
                  cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));

        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs = makeMarshalOptArgs();

        nixlServiceXferReqH *first_request = nullptr;
        ASSERT_EQ(agent_0_->createXferReq(NIXL_WRITE,
                                          local_xfer_descs,
                                          remote_xfer_descs,
                                          "agent_1",
                                          first_request,
                                          &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(first_request, nullptr);
        ASSERT_EQ(agent_0_->postXferReq(first_request, &extra_params), NIXL_SUCCESS);
        ASSERT_EQ(progressUntilTerminal(*agent_0_, *agent_1_, first_request), NIXL_SUCCESS)
            << "initial WRITE did not complete within the time bound";

        // Completion generated DELETE, but agent_1 has not service-progressed since then.
        ASSERT_EQ(agent_0_->releaseXferReq(first_request), NIXL_SUCCESS);

        nixlServiceXferReqH *replacement_request = nullptr;
        ASSERT_EQ(agent_0_->createXferReq(NIXL_WRITE,
                                          local_xfer_descs,
                                          remote_xfer_descs,
                                          "agent_1",
                                          replacement_request,
                                          &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(replacement_request, nullptr);
        ASSERT_EQ(agent_0_->postXferReq(replacement_request, &extra_params), NIXL_SUCCESS);

        // Force DELETE(old) and RTS(new) through one progressService() call. The old receive
        // slots must be reclaimed before the replacement RTS is admitted.
        ASSERT_EQ(dispatchServiceNotificationsInOneProgressCall(
                      *agent_0_, "agent_1", *agent_1_, "delete-rts-ready"),
                  NIXL_SUCCESS);
        ASSERT_EQ(progressUntilTerminal(*agent_0_, *agent_1_, replacement_request), NIXL_SUCCESS)
            << "replacement WRITE did not complete within the time bound";

        ASSERT_EQ(cudaMemcpy(dst_data_.data(), mem_agent_1_, memSize, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(dst_data_, src_data_);
        ASSERT_EQ(agent_0_->releaseXferReq(replacement_request), NIXL_SUCCESS);
    }

    TEST_F(nixlServiceOrderingTest, WriteCompletionThenReadAdmissionSameProgressTick) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        fillWithRepeatingRandomPattern(src_data_, memSize);
        ASSERT_EQ(cudaMemcpy(mem_agent_0_, src_data_.data(), memSize, cudaMemcpyHostToDevice),
                  cudaSuccess);

        nixl_xfer_dlist_t write_local_descs(VRAM_SEG);
        write_local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
        nixl_xfer_dlist_t write_remote_descs(VRAM_SEG);
        write_remote_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));

        nixl_service_opt_args_t write_params{};
        write_params.marshalOptArgs = makeMarshalOptArgs();
        nixlServiceXferReqH *write_request = nullptr;
        ASSERT_EQ(agent_0_->createXferReq(NIXL_WRITE,
                                          write_local_descs,
                                          write_remote_descs,
                                          "agent_1",
                                          write_request,
                                          &write_params),
                  NIXL_SUCCESS);
        ASSERT_NE(write_request, nullptr);
        ASSERT_EQ(agent_0_->postXferReq(write_request, &write_params), NIXL_SUCCESS);
        ASSERT_EQ(progressUntilTerminal(*agent_0_, *agent_1_, write_request), NIXL_SUCCESS)
            << "initial WRITE did not complete within the time bound";
        ASSERT_EQ(agent_0_->releaseXferReq(write_request), NIXL_SUCCESS);

        ASSERT_EQ(cudaMemset(mem_agent_0_, 0, memSize), cudaSuccess);
        nixl_xfer_dlist_t read_local_descs(VRAM_SEG);
        read_local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
        nixl_xfer_dlist_t read_remote_descs(VRAM_SEG);
        read_remote_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));

        nixl_service_opt_args_t read_params{};
        read_params.marshalOptArgs = makeMarshalOptArgs(delta_ref_agent_1_, delta_ref_agent_0_);
        nixlServiceXferReqH *read_request = nullptr;
        ASSERT_EQ(agent_0_->createXferReq(NIXL_READ,
                                          read_local_descs,
                                          read_remote_descs,
                                          "agent_1",
                                          read_request,
                                          &read_params),
                  NIXL_SUCCESS);
        ASSERT_NE(read_request, nullptr);
        ASSERT_EQ(agent_0_->postXferReq(read_request, &read_params), NIXL_IN_PROG);

        // DELETE(old WRITE) and RREQ(new READ) target the same one-group pool. The admission
        // barrier applies equally to RREQ; otherwise the peer rejects this reclaimable request.
        ASSERT_EQ(dispatchServiceNotificationsInOneProgressCall(
                      *agent_0_, "agent_1", *agent_1_, "delete-rreq-ready"),
                  NIXL_SUCCESS);
        ASSERT_EQ(progressUntilTerminal(*agent_0_, *agent_1_, read_request), NIXL_SUCCESS)
            << "READ admission did not reuse slots released by the preceding DELETE";

        ASSERT_EQ(cudaMemcpy(dst_data_.data(), mem_agent_0_, memSize, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(dst_data_, src_data_);
        ASSERT_EQ(agent_0_->releaseXferReq(read_request), NIXL_SUCCESS);
    }

    TEST_F(nixlServiceOrderingTest, ReadStartThenCancelSameProgressTick) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        fillWithRepeatingRandomPattern(src_data_, memSize);
        ASSERT_EQ(cudaMemcpy(mem_agent_1_, src_data_.data(), memSize, cudaMemcpyHostToDevice),
                  cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));

        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs = makeMarshalOptArgs(delta_ref_agent_1_, delta_ref_agent_0_);

        nixlServiceXferReqH *cancelled_request = nullptr;
        ASSERT_EQ(agent_0_->createXferReq(NIXL_READ,
                                          local_xfer_descs,
                                          remote_xfer_descs,
                                          "agent_1",
                                          cancelled_request,
                                          &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(cancelled_request, nullptr);
        ASSERT_EQ(agent_0_->postXferReq(cancelled_request, &extra_params), NIXL_IN_PROG);
        ASSERT_EQ(agent_0_->releaseXferReq(cancelled_request), NIXL_SUCCESS);

        // RREQ must be admitted before the following RABORT is dispatched. Reordering these
        // controls would acknowledge a nonexistent serve and then resurrect it.
        ASSERT_EQ(dispatchServiceNotificationsInOneProgressCall(
                      *agent_0_, "agent_1", *agent_1_, "rreq-rabort-ready"),
                  NIXL_SUCCESS);
        ASSERT_EQ(dispatchServiceNotificationsInOneProgressCall(
                      *agent_1_, "agent_0", *agent_0_, "rabort-ack-ready"),
                  NIXL_SUCCESS);

        nixlServiceXferReqH *replacement_request = nullptr;
        ASSERT_EQ(agent_0_->createXferReq(NIXL_READ,
                                          local_xfer_descs,
                                          remote_xfer_descs,
                                          "agent_1",
                                          replacement_request,
                                          &extra_params),
                  NIXL_SUCCESS);
        ASSERT_NE(replacement_request, nullptr);
        ASSERT_EQ(agent_0_->postXferReq(replacement_request, &extra_params), NIXL_IN_PROG);
        ASSERT_EQ(progressUntilTerminal(*agent_0_, *agent_1_, replacement_request), NIXL_SUCCESS)
            << "READ replacement failed after same-tick cancellation";

        ASSERT_EQ(cudaMemcpy(dst_data_.data(), mem_agent_0_, memSize, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(dst_data_, src_data_);
        ASSERT_EQ(agent_0_->releaseXferReq(replacement_request), NIXL_SUCCESS);
    }

    TEST_F(nixlServiceOrderingTest, ReadAbortAckThenWriteAdmissionSameProgressTick) {
        if (!etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        fillWithRepeatingRandomPattern(src_data_, memSize);
        ASSERT_EQ(cudaMemcpy(mem_agent_1_, src_data_.data(), memSize, cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(mem_agent_0_, 0, memSize), cudaSuccess);

        nixl_xfer_dlist_t read_local_descs(VRAM_SEG);
        read_local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));
        nixl_xfer_dlist_t read_remote_descs(VRAM_SEG);
        read_remote_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));

        nixl_service_opt_args_t read_params{};
        read_params.marshalOptArgs = makeMarshalOptArgs(delta_ref_agent_1_, delta_ref_agent_0_);
        nixlServiceXferReqH *read_request = nullptr;
        ASSERT_EQ(agent_0_->createXferReq(NIXL_READ,
                                          read_local_descs,
                                          read_remote_descs,
                                          "agent_1",
                                          read_request,
                                          &read_params),
                  NIXL_SUCCESS);
        ASSERT_NE(read_request, nullptr);
        ASSERT_EQ(agent_0_->postXferReq(read_request, &read_params), NIXL_IN_PROG);
        ASSERT_EQ(agent_0_->releaseXferReq(read_request), NIXL_SUCCESS);

        // Dispatch RREQ+RABORT at agent_1. This sends RABORT_ACK but deliberately leaves it
        // queued at agent_0 so the ACK can share a progress tick with the RTS below.
        ASSERT_EQ(dispatchServiceNotificationsInOneProgressCall(
                      *agent_0_, "agent_1", *agent_1_, "rreq-rabort-before-write-ready"),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t write_local_descs(VRAM_SEG);
        write_local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1_), memSize, device_id_));
        nixl_xfer_dlist_t write_remote_descs(VRAM_SEG);
        write_remote_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0_), memSize, device_id_));

        nixl_service_opt_args_t write_params{};
        write_params.marshalOptArgs = makeMarshalOptArgs();
        nixlServiceXferReqH *write_request = nullptr;
        ASSERT_EQ(agent_1_->createXferReq(NIXL_WRITE,
                                          write_local_descs,
                                          write_remote_descs,
                                          "agent_0",
                                          write_request,
                                          &write_params),
                  NIXL_SUCCESS);
        ASSERT_NE(write_request, nullptr);
        ASSERT_EQ(agent_1_->postXferReq(write_request, &write_params), NIXL_SUCCESS);

        // RABORT_ACK makes the cancelled READ's receive slots reclaimable. They must be freed
        // before the following RTS attempts to allocate the same slot group.
        ASSERT_EQ(dispatchServiceNotificationsInOneProgressCall(
                      *agent_1_, "agent_0", *agent_0_, "rabort-ack-rts-ready"),
                  NIXL_SUCCESS);
        ASSERT_EQ(progressUntilTerminal(*agent_1_, *agent_0_, write_request), NIXL_SUCCESS)
            << "WRITE admission did not reuse slots released by the preceding RABORT_ACK";

        ASSERT_EQ(cudaMemcpy(dst_data_.data(), mem_agent_0_, memSize, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(dst_data_, src_data_);
        ASSERT_EQ(agent_1_->releaseXferReq(write_request), NIXL_SUCCESS);
    }

    // Cancels a marshalled READ at a sweep of different points in its lifecycle - before ever
    // posting, immediately after posting, at a range of increasing delays after posting, and
    // after letting it fully complete - so that across the sweep a release is likely to land
    // while a slot is in each of local_slot_state_t's BUSY_MARSHAL/READY_TO_SEND/BUSY_NIXL/FREE at
    // least once, without needing invasive test-only synchronization hooks. Before every release,
    // never-posted guard requests reserve all remaining transfer groups. After the cancellation
    // settles, exactly one replacement must fit and an additional request must fail, proving that
    // the active request's complete slot group was reclaimed regardless of the dependency-specific
    // physical slot stride. A full, independent READ (runReadValidTest) runs once at the very end
    // to prove the pool is left not just numerically free but functionally correct.
    TYPED_TEST(nixlServiceTest, ReadCancelInEverySlotState) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), this->memSize, this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), this->memSize, this->device_id_));

        const auto make_extra_params = [&] {
            nixl_service_opt_args_t extra_params{};
            extra_params.marshalOptArgs =
                this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_);
            return extra_params;
        };

        // Releasing a never-posted READ takes releaseXferReq's PRE_START branch specifically -
        // exercised once, up front, since it has no "posted" counterpart in the sweep below.
        {
            auto extra_params = make_extra_params();
            nixlServiceXferReqH *req_hndl = nullptr;
            ASSERT_EQ(this->agent_0_->createXferReq(NIXL_READ,
                                                    local_xfer_descs,
                                                    remote_xfer_descs,
                                                    "agent_1",
                                                    req_hndl,
                                                    &extra_params),
                      NIXL_SUCCESS);
            ASSERT_NE(req_hndl, nullptr);
            ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);

            req_hndl = nullptr;
            ASSERT_EQ(this->agent_0_->createXferReq(NIXL_READ,
                                                    local_xfer_descs,
                                                    remote_xfer_descs,
                                                    "agent_1",
                                                    req_hndl,
                                                    &extra_params),
                      NIXL_SUCCESS);
            ASSERT_NE(req_hndl, nullptr);
            ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
        }

        // A log-scale sweep of delays between posting and releasing, plus std::nullopt meaning
        // "let it fully complete first" (the DONE release branch).
        const std::vector<std::optional<std::chrono::microseconds>> release_points = {
            std::chrono::microseconds(0),
            std::chrono::microseconds(200),
            std::chrono::microseconds(2000),
            std::chrono::microseconds(20000),
            std::chrono::microseconds(200000),
            std::nullopt,
        };

        for (size_t trial = 0; trial < release_points.size(); ++trial) {
            auto extra_params = make_extra_params();
            nixlServiceXferReqH *req_hndl = nullptr;
            ASSERT_EQ(this->agent_0_->createXferReq(NIXL_READ,
                                                    local_xfer_descs,
                                                    remote_xfer_descs,
                                                    "agent_1",
                                                    req_hndl,
                                                    &extra_params),
                      NIXL_SUCCESS)
                << "trial " << trial;
            ASSERT_NE(req_hndl, nullptr);
            ASSERT_EQ(this->agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG)
                << "trial " << trial;

            nixl_notifs_t notifs;
            if (release_points[trial].has_value()) {
                std::this_thread::sleep_for(*release_points[trial]);
            } else {
                const auto completion_deadline =
                    std::chrono::steady_clock::now() + this->testTimeout;
                while (this->agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
                    ASSERT_LT(std::chrono::steady_clock::now(), completion_deadline)
                        << "trial " << trial << ": READ did not complete before release";
                    ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
                    ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
                }
            }
            ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS) << "trial " << trial;
            req_hndl = nullptr;

            // Let the drain-to-quiescence (RABORT/RABORT_ACK), or the immediate free path if it
            // had already reached DONE, fully settle before checking for a stale write. 20ms
            // steps match the settle loop in ReadPeerRejectsFingerprintMismatch.
            for (int i = 0; i < 10; ++i) {
                ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
                ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }

            ASSERT_EQ(this->agent_0_->createXferReq(NIXL_READ,
                                                    local_xfer_descs,
                                                    remote_xfer_descs,
                                                    "agent_1",
                                                    req_hndl,
                                                    &extra_params),
                      NIXL_SUCCESS)
                << "trial " << trial << ": released slot group was not reclaimed";
            ASSERT_NE(req_hndl, nullptr);
            ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS) << "trial " << trial;

            std::vector<unsigned char> settled(this->memSize);
            ASSERT_EQ(
                cudaMemcpy(
                    settled.data(), this->mem_agent_0_, this->memSize, cudaMemcpyDeviceToHost),
                cudaSuccess);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::vector<unsigned char> after_settle(this->memSize);
            ASSERT_EQ(
                cudaMemcpy(
                    after_settle.data(), this->mem_agent_0_, this->memSize, cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(settled, after_settle)
                << "trial " << trial << ": destination buffer changed after cancellation settled"
                << " - a stale write landed after the cancellation was believed complete";
        }

        // Both agents' slot pools must be fully usable again after the whole sweep above: run one
        // complete, independent READ end-to-end (see this test's header comment for why once,
        // here, is enough).
        this->runReadValidTest();
    }

    // Injects a batch of malformed/adversarial/duplicate READ protocol messages - targeting a
    // real in-flight READ's own xfer id where a handler's deeper validation needs one to reach -
    // via the public genNotif(), the same entry point any peer's messages arrive through. None
    // of this should be observable in the outcome: every message below must be silently dropped
    // by the validation added throughout the READ implementation, and the real transfer must
    // still complete correctly. Deliberately does not test truncated/short messages: every
    // _NIXLS_* payload (READ's and the pre-existing WRITE ones alike) decodes via readScalar(),
    // which bounds-checks with NIXL_ASSERT (DCHECK) rather than a graceful, always-enforced
    // check - a pre-existing protocol-wide design choice, not something introduced by READ.
    // The real transfer only needs to be large enough for a few rounds of slot reuse to
    // interleave with the injected messages, not the fixture's full memSize - that many-chunk
    // pipeline is already covered by ReadValid/ReadMixed/PostXferReq. It does need to stay above
    // direct_desc_threshold (64 MB, mirrored below - see createXferReq's split), though, or this
    // silently degrades to a direct-only READ that bypasses the marshal protocol entirely, which
    // is what every message injected below targets.
    TYPED_TEST(nixlServiceTest, ReadMalformedAndDuplicateMessagesIgnored) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        constexpr size_t direct_desc_threshold = 64 * 1024 * 1024;
        constexpr size_t transfer_size =
            direct_desc_threshold + TypeParam::chunkSize * slots_per_xfer * 2;
        static_assert(transfer_size > direct_desc_threshold);
        static_assert(transfer_size <= TypeParam::memSize);

        // A third agent that only exchanges metadata with agent_0_ (never agent_1_), used to
        // verify handleRPosted's sender check - without it, any agent able to reach agent_0_
        // could inject data into this READ merely by guessing its xfer id.
        nixlServiceAgentConfig impostor_cfg;
        impostor_cfg.mode = nixlMarshalStagingConfig{};
        impostor_cfg.useProgThread = true;
        auto impostor =
            std::make_unique<testServiceAgent>("impostor", impostor_cfg, this->chunkSize);
        nixlBackendH *impostor_backend = nullptr;
        ASSERT_EQ(impostor->createBackend("UCX", {}, impostor_backend), NIXL_SUCCESS);
        nixl_blob_t impostor_md;
        ASSERT_EQ(impostor->getLocalMD(impostor_md), NIXL_SUCCESS);
        std::string impostor_remote_name;
        ASSERT_EQ(this->agent_0_->loadRemoteMD(impostor_md, impostor_remote_name), NIXL_SUCCESS);
        ASSERT_EQ(impostor_remote_name, "impostor");
        nixl_blob_t agent_0_md;
        ASSERT_EQ(this->agent_0_->getLocalMD(agent_0_md), NIXL_SUCCESS);
        std::string agent_0_remote_name;
        ASSERT_EQ(impostor->loadRemoteMD(agent_0_md, agent_0_remote_name), NIXL_SUCCESS);
        ASSERT_EQ(agent_0_remote_name, "agent_0");

        this->fillWithRepeatingRandomPattern(this->src_data_, transfer_size);
        ASSERT_EQ(
            cudaMemcpy(
                this->mem_agent_1_, this->src_data_.data(), transfer_size, cudaMemcpyHostToDevice),
            cudaSuccess);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_0_), transfer_size, this->device_id_));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(nixlBasicDesc(
            reinterpret_cast<uintptr_t>(this->mem_agent_1_), transfer_size, this->device_id_));

        nixlServiceXferReqH *req_hndl = nullptr;
        nixl_service_opt_args_t extra_params{};
        extra_params.marshalOptArgs =
            this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_);
        extra_params.notif = "malformed_test_done";
        ASSERT_EQ(
            this->agent_0_->createXferReq(
                NIXL_READ, local_xfer_descs, remote_xfer_descs, "agent_1", req_hndl, &extra_params),
            NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        ASSERT_TRUE(req_hndl->readReceiveXferId.has_value());
        const size_t xfer_id =
            *req_hndl->readReceiveXferId; // Also rReqPayload.xferId - see genRREQ.
        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

        const auto backend = marshalBackendFromConfig(typename TypeParam::marshal_config_t{});
        constexpr size_t wire_size = 16;

        // Unrecognized _NIXLS_ subtype: serviceNotifCallback must drop it, not assert.
        ASSERT_EQ(this->agent_1_->genNotif("agent_0", "_NIXLS_BOGUS_no-such-message-kind"),
                  NIXL_SUCCESS);
        ASSERT_EQ(this->agent_0_->genNotif("agent_1", "_NIXLS_BOGUS_no-such-message-kind"),
                  NIXL_SUCCESS);

        // RPOSTED for a nonexistent xfer id.
        ASSERT_EQ(
            this->agent_1_->genNotif(
                "agent_0",
                rPostedPayload(xfer_id + 999983, 0, 16, wire_size, 0, "", backend).serialize()),
            NIXL_SUCCESS);
        // RPOSTED for the real xfer id, but an out-of-range slot index.
        ASSERT_EQ(this->agent_1_->genNotif(
                      "agent_0",
                      rPostedPayload(xfer_id, slots_per_xfer + 7, 16, wire_size, 0, "", backend)
                          .serialize()),
                  NIXL_SUCCESS);

        // RPOSTED for the real xfer id and a valid slot index, but a wildly out-of-bounds
        // destination offset ("oversized" relative to the actual destination region).
        ASSERT_EQ(this->agent_1_->genNotif(
                      "agent_0",
                      rPostedPayload(xfer_id, 0, 16, wire_size, transfer_size * 4, "", backend)
                          .serialize()),
                  NIXL_SUCCESS);
        // RPOSTED for the real xfer id from an agent that is not this READ's actual peer.
        ASSERT_EQ(
            impostor->genNotif(
                "agent_0", rPostedPayload(xfer_id, 0, 16, wire_size, 0, "", backend).serialize()),
            NIXL_SUCCESS);

        // RPOSTED for the real xfer id and a valid slot index, with dstByteOffset/originalSize
        // chosen so their sum overflows size_t and wraps back to (in this case, exactly) 0, well
        // within the destination region's bounds. resolveAbsoluteOffset() must reject this via
        // the overflow-safe "size > desc.len - offset_in_desc" comparison, not the overflow-prone
        // "offset_in_desc + size > desc.len" form it replaced (which this would have bypassed).
        {
            const size_t overflow_offset = transfer_size / 2;
            const size_t overflow_size = std::numeric_limits<size_t>::max() - overflow_offset + 1;
            ASSERT_EQ(this->agent_1_->genNotif(
                          "agent_0",
                          rPostedPayload(
                              xfer_id, 0, overflow_size, wire_size, overflow_offset, "", backend)
                              .serialize()),
                      NIXL_SUCCESS);
        }

        // RRSLOT/RABORT/RABORT_ACK/RNAK for nonexistent xfer ids, sent to whichever side would
        // real ones of each kind normally go to.
        ASSERT_EQ(
            this->agent_0_->genNotif("agent_1", rrSlotPayload(xfer_id + 999983, 0, 0).serialize()),
            NIXL_SUCCESS);
        ASSERT_EQ(this->agent_0_->genNotif("agent_1", rAbortPayload(xfer_id + 999983).serialize()),
                  NIXL_SUCCESS);
        ASSERT_EQ(
            this->agent_1_->genNotif("agent_0", rAbortAckPayload(xfer_id + 999983).serialize()),
            NIXL_SUCCESS);
        ASSERT_EQ(this->agent_1_->genNotif(
                      "agent_0",
                      rNakPayload(xfer_id + 999983, static_cast<int32_t>(NIXL_ERR_INVALID_PARAM))
                          .serialize()),
                  NIXL_SUCCESS);

        // RRSLOT for the real xfer id and slot 0, but a generation nothing has used yet:
        // dropped as stale/mismatched by handleRRSlot's remoteSlotStates/generation check.
        ASSERT_EQ(this->agent_0_->genNotif(
                      "agent_1", rrSlotPayload(xfer_id, 0, 0xFFFFFFFFFFFFFFFFull).serialize()),
                  NIXL_SUCCESS);

        // Duplicate RREQ for the same (sender, xfer id) as the already in-flight serve: dropped
        // by handleRREQ's own duplicate check, rather than tearing down the real serve's slots.
        {
            const marshalLayoutFingerprint bogus_fp{};
            const std::array<nixlBasicDesc, slots_per_xfer> bogus_slots{
                nixlBasicDesc(0x1000, 16, 0), nixlBasicDesc(0x2000, 16, 0)};
            const std::array<nixlMarshal::mem_space_t, slots_per_xfer> bogus_mem_spaces{
                nixlMarshal::mem_space_t::DEVICE, nixlMarshal::mem_space_t::DEVICE};
            // rReqPayload's constructor requires a non-empty serialized source list; its
            // content is irrelevant here since handleRREQ's duplicate check (below) returns
            // before ever parsing it.
            const rReqPayload duplicate_rreq(
                xfer_id, std::string("unused"), bogus_slots, bogus_mem_spaces, bogus_fp);
            ASSERT_EQ(this->agent_0_->genNotif("agent_1", duplicate_rreq.serialize()),
                      NIXL_SUCCESS);
        }

        // None of the above should have disturbed the real transfer: it must still run to
        // completion normally, ending up byte-for-byte equal to the source, with exactly one
        // notification delivered to the peer.
        nixl_notifs_t notifs;
        while (this->agent_0_->getXferStatus(req_hndl) == NIXL_IN_PROG) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_LE(notifs.size(), 1);
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl), NIXL_SUCCESS);
        // Compares the full (memSize) buffers, not just the transferred transfer_size prefix:
        // both remain zero-filled past transfer_size (untouched since SetUp), so this doubles as
        // a check that the transfer didn't write anywhere beyond its own descriptor.
        ASSERT_EQ(
            cudaMemcpy(
                this->dst_data_.data(), this->mem_agent_0_, this->memSize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(this->dst_data_, this->src_data_);
        while (notifs.size() == 0) {
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
        }
        ASSERT_EQ(notifs.size(), 1);
        ASSERT_EQ(notifs["agent_0"].size(), 1);
        ASSERT_EQ(notifs["agent_0"].front(), "malformed_test_done");

        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl), NIXL_SUCCESS);
    }

    // The peer's own RREQ admission check (handleRREQ) rejects a mismatched
    // marshalLayoutFingerprint - here, by giving each agent a different chunk size - via RNAK,
    // and the initiator surfaces the resulting terminal error through getXferStatus rather than
    // hanging. Deliberately not built on the shared nixlServiceTestF fixture, which forces both
    // agents to share one nixlServiceAgentConfig; staging-only, since the mismatch itself is
    // what is under test, not any particular marshal mode.
    TEST(NixlServiceReadRobustness, ReadPeerRejectsFingerprintMismatch) {
        const auto env_etcd_endpoints_set = (std::getenv("NIXL_ETCD_ENDPOINTS") != nullptr);
        const auto etcd_running = isPortOpen("127.0.0.1", 2379);
        if (env_etcd_endpoints_set && !etcd_running) {
            GTEST_FAIL() << "NIXL_ETCD_ENDPOINTS is set, but etcd is not running, skipping test";
        }
        if (!env_etcd_endpoints_set) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }

        constexpr size_t chunk_size_initiator = 128 * 1024;
        constexpr size_t chunk_size_peer = 256 * 1024; // Deliberately mismatched.
        constexpr size_t mem_size = 1024 * chunk_size_initiator; // Exceeds direct_desc_threshold.
        constexpr uint32_t max_concurrent_transfers = 1;
        constexpr uint32_t slot_groups = max_concurrent_transfers + 1;
        constexpr size_t svc_mem_size_initiator =
            chunk_size_initiator * MarshalBackendSizing::slots_per_transfer * slot_groups;
        constexpr size_t svc_mem_size_peer =
            chunk_size_peer * MarshalBackendSizing::slots_per_transfer * slot_groups;

        nixlServiceAgentConfig cfg_initiator;
        cfg_initiator.mode = nixlMarshalStagingConfig{};
        cfg_initiator.useProgThread = true;
        nixlServiceAgentConfig cfg_peer;
        cfg_peer.mode = nixlMarshalStagingConfig{};
        cfg_peer.useProgThread = true;

        auto agent_0 =
            std::make_unique<testServiceAgent>("agent_0", cfg_initiator, chunk_size_initiator);
        auto agent_1 = std::make_unique<testServiceAgent>("agent_1", cfg_peer, chunk_size_peer);

        nixlBackendH *backend_handle_0 = nullptr;
        nixlBackendH *backend_handle_1 = nullptr;
        ASSERT_EQ(agent_0->createBackend("UCX", {}, backend_handle_0), NIXL_SUCCESS);
        ASSERT_EQ(agent_1->createBackend("UCX", {}, backend_handle_1), NIXL_SUCCESS);

        int device_count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        int device_id = 0;
        ASSERT_EQ(cudaGetDevice(&device_id), cudaSuccess);

        void *mem_agent_0 = nullptr;
        void *svc_mem_agent_0 = nullptr;
        void *mem_agent_1 = nullptr;
        void *svc_mem_agent_1 = nullptr;
        ASSERT_EQ(cudaMalloc(&mem_agent_0, mem_size), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&svc_mem_agent_0, svc_mem_size_initiator), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&mem_agent_1, mem_size), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&svc_mem_agent_1, svc_mem_size_peer), cudaSuccess);
        // Stop the agents (and their progress threads) before freeing the GPU memory they have
        // registered, so no in-flight progress can touch freed device memory. absl::Cleanup runs
        // in reverse construction order, so a single combined cleanup makes the ordering explicit
        // rather than relying on the relative declaration order of two separate cleanups.
        absl::Cleanup cleanup = [&] {
            agent_0.reset();
            agent_1.reset();
            (void)cudaFree(mem_agent_0);
            (void)cudaFree(svc_mem_agent_0);
            (void)cudaFree(mem_agent_1);
            (void)cudaFree(svc_mem_agent_1);
        };
        ASSERT_EQ(cudaMemset(mem_agent_0, 0, mem_size), cudaSuccess);
        ASSERT_EQ(cudaMemset(mem_agent_1, 0, mem_size), cudaSuccess);

        nixl_reg_dlist_t mem_descs_agent_0(VRAM_SEG);
        mem_descs_agent_0.addDesc(
            nixlBlobDesc(reinterpret_cast<uintptr_t>(mem_agent_0), mem_size, device_id));
        nixl_reg_dlist_t svc_descs_agent_0(VRAM_SEG);
        svc_descs_agent_0.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(svc_mem_agent_0), svc_mem_size_initiator, device_id));
        nixl_reg_dlist_t mem_descs_agent_1(VRAM_SEG);
        mem_descs_agent_1.addDesc(
            nixlBlobDesc(reinterpret_cast<uintptr_t>(mem_agent_1), mem_size, device_id));
        nixl_reg_dlist_t svc_descs_agent_1(VRAM_SEG);
        svc_descs_agent_1.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(svc_mem_agent_1), svc_mem_size_peer, device_id));

        ASSERT_EQ(agent_0->registerMem(mem_descs_agent_0), NIXL_SUCCESS);
        ASSERT_EQ(agent_0->registerServiceMem(svc_descs_agent_0), NIXL_SUCCESS);
        ASSERT_EQ(agent_1->registerMem(mem_descs_agent_1), NIXL_SUCCESS);
        ASSERT_EQ(agent_1->registerServiceMem(svc_descs_agent_1), NIXL_SUCCESS);

        nixl_blob_t md_agent_0;
        nixl_blob_t md_agent_1;
        ASSERT_EQ(agent_0->getLocalMD(md_agent_0), NIXL_SUCCESS);
        ASSERT_EQ(agent_1->getLocalMD(md_agent_1), NIXL_SUCCESS);
        std::string remote_name_0;
        std::string remote_name_1;
        ASSERT_EQ(agent_1->loadRemoteMD(md_agent_0, remote_name_1), NIXL_SUCCESS);
        ASSERT_EQ(agent_0->loadRemoteMD(md_agent_1, remote_name_0), NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_descs(VRAM_SEG);
        local_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_0), mem_size, device_id));
        nixl_xfer_dlist_t remote_xfer_descs(VRAM_SEG);
        remote_xfer_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(mem_agent_1), mem_size, device_id));

        nixlServiceXferReqH *req_hndl = nullptr;
        nixl_service_opt_args_t extra_params{};
        // Unlike nixlServiceTestF's tests, there is no fixture default here - without this,
        // marshalOptArgs stays nixlMarshalDirectOptArgs{}, which takes createXferReq's
        // direct-path branch and never sends an RREQ (a plain RDMA READ that bypasses the
        // service protocol entirely, so the two agents' differing staging configs would never
        // even be compared).
        extra_params.marshalOptArgs = nixlMarshalStagingOptArgs{};
        ASSERT_EQ(
            agent_0->createXferReq(
                NIXL_READ, local_xfer_descs, remote_xfer_descs, "agent_1", req_hndl, &extra_params),
            NIXL_SUCCESS);
        ASSERT_NE(req_hndl, nullptr);
        ASSERT_EQ(agent_0->postXferReq(req_hndl, &extra_params), NIXL_IN_PROG);

        nixl_notifs_t notifs;
        nixl_status_t status = NIXL_IN_PROG;
        for (int i = 0; i < 200 && status == NIXL_IN_PROG; ++i) {
            status = agent_0->getXferStatus(req_hndl);
            ASSERT_EQ(agent_0->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(agent_1->getNotifs(notifs), NIXL_SUCCESS);
            if (status == NIXL_IN_PROG) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        // handleRREQ's fingerprint check rejects the mismatched chunk size with
        // send_r_nak(NIXL_ERR_INVALID_PARAM), which handleRNak surfaces on the initiator as this
        // same terminal status, rather than leaving the request hanging in NIXL_IN_PROG forever.
        EXPECT_EQ(status, NIXL_ERR_INVALID_PARAM);

        std::vector<unsigned char> zeros(mem_size, 0);
        std::vector<unsigned char> dst_check(mem_size);
        ASSERT_EQ(cudaMemcpy(dst_check.data(), mem_agent_0, mem_size, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(dst_check, zeros)
            << "a rejected READ must not have written any data to the destination";

        ASSERT_EQ(agent_0->releaseXferReq(req_hndl), NIXL_SUCCESS);
        // Let the RABORT/RABORT_ACK round trip settle before the agents are torn down below.
        for (int i = 0; i < 10; ++i) {
            ASSERT_EQ(agent_0->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(agent_1->getNotifs(notifs), NIXL_SUCCESS);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    TYPED_TEST(nixlServiceBidirectionalTest, SimultaneousBidirectionalTransfers) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }

        const size_t transfer_size = this->memSize / 4;
        const size_t avg_desc_size = transfer_size / 4;
        const size_t src_offset = 0;
        const size_t dst_offset = transfer_size;
        const std::vector<size_t> desc_sizes = {
            avg_desc_size - 256,
            avg_desc_size + 256,
            avg_desc_size + 512,
            avg_desc_size - 512,
        };

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 4);
        std::vector<unsigned char> agent_0_src(transfer_size);
        std::vector<unsigned char> agent_1_src(transfer_size);
        std::vector<unsigned char> agent_0_dst(transfer_size, 0);
        std::vector<unsigned char> agent_1_dst(transfer_size, 0);
        for (auto &v : agent_0_src) {
            v = static_cast<unsigned char>(dist(rng));
        }
        for (auto &v : agent_1_src) {
            v = static_cast<unsigned char>(dist(rng));
        }

        ASSERT_EQ(cudaMemcpy(reinterpret_cast<char *>(this->mem_agent_0_) + src_offset,
                             agent_0_src.data(),
                             transfer_size,
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(reinterpret_cast<char *>(this->mem_agent_1_) + src_offset,
                             agent_1_src.data(),
                             transfer_size,
                             cudaMemcpyHostToDevice),
                  cudaSuccess);

        const auto add_descs = [&](nixl_xfer_dlist_t &descs, void *base, size_t offset) {
            const auto addr = reinterpret_cast<uintptr_t>(base) + offset;
            if constexpr (std::is_same_v<typename TestFixture::marshal_config_t,
                                         nixlMarshalDeltaConfig>) {
                descs.addDesc(nixlBasicDesc(addr, transfer_size, this->device_id_));
            } else {
                size_t desc_offset = 0;
                for (const auto desc_size : desc_sizes) {
                    descs.addDesc(nixlBasicDesc(addr + desc_offset, desc_size, this->device_id_));
                    desc_offset += desc_size;
                }
                ASSERT_EQ(desc_offset, transfer_size);
            }
        };

        nixl_xfer_dlist_t agent_0_local_xfer_descs(VRAM_SEG);
        add_descs(agent_0_local_xfer_descs, this->mem_agent_0_, src_offset);
        nixl_xfer_dlist_t agent_1_remote_xfer_descs(VRAM_SEG);
        add_descs(agent_1_remote_xfer_descs, this->mem_agent_1_, dst_offset);

        nixl_xfer_dlist_t agent_1_local_xfer_descs(VRAM_SEG);
        add_descs(agent_1_local_xfer_descs, this->mem_agent_1_, src_offset);
        nixl_xfer_dlist_t agent_0_remote_xfer_descs(VRAM_SEG);
        add_descs(agent_0_remote_xfer_descs, this->mem_agent_0_, dst_offset);

        nixl_service_opt_args_t extra_params_0{};
        extra_params_0.marshalOptArgs = this->makeMarshalOptArgs();
        nixlServiceXferReqH *req_hndl_0 = nullptr;
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_WRITE,
                                                agent_0_local_xfer_descs,
                                                agent_1_remote_xfer_descs,
                                                "agent_1",
                                                req_hndl_0,
                                                &extra_params_0),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl_0, nullptr);

        nixl_service_opt_args_t extra_params_1{};
        extra_params_1.marshalOptArgs =
            this->makeMarshalOptArgs(this->delta_ref_agent_1_, this->delta_ref_agent_0_);
        nixlServiceXferReqH *req_hndl_1 = nullptr;
        ASSERT_EQ(this->agent_1_->createXferReq(NIXL_WRITE,
                                                agent_1_local_xfer_descs,
                                                agent_0_remote_xfer_descs,
                                                "agent_0",
                                                req_hndl_1,
                                                &extra_params_1),
                  NIXL_SUCCESS);
        ASSERT_NE(req_hndl_1, nullptr);

        ASSERT_EQ(this->agent_0_->postXferReq(req_hndl_0, &extra_params_0), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_1_->postXferReq(req_hndl_1, &extra_params_1), NIXL_SUCCESS);

        nixl_notifs_t notifs;
        while (this->agent_0_->getXferStatus(req_hndl_0) == NIXL_IN_PROG ||
               this->agent_1_->getXferStatus(req_hndl_1) == NIXL_IN_PROG) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs), NIXL_SUCCESS);
            ASSERT_EQ(notifs.size(), 0);
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(req_hndl_0), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_1_->getXferStatus(req_hndl_1), NIXL_SUCCESS);

        ASSERT_EQ(cudaMemcpy(agent_1_dst.data(),
                             reinterpret_cast<char *>(this->mem_agent_1_) + dst_offset,
                             transfer_size,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(agent_0_dst.data(),
                             reinterpret_cast<char *>(this->mem_agent_0_) + dst_offset,
                             transfer_size,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(agent_1_dst, agent_0_src);
        ASSERT_EQ(agent_0_dst, agent_1_src);

        ASSERT_EQ(this->agent_0_->releaseXferReq(req_hndl_0), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_1_->releaseXferReq(req_hndl_1), NIXL_SUCCESS);
    }

    // A READ and a WRITE, in opposite directions between the same two agents, landing under the
    // same numeric xfer id but in different, role-separated maps: agent_0_'s own WRITE
    // (outboundXferReqs_[0]) alongside its serving of agent_1_'s READ (readServeReqs_[0]); and
    // symmetrically on agent_1_, the WRITE arriving (inboundXferReqs_[0]) alongside its own READ
    // (readReceiveReqs_[0]). Both agents are freshly constructed by SetUp(), so each one's own
    // first marshalled transfer naturally gets xferId 0 - no seed hook needed to force the
    // collision. Runs on nixlServiceBidirectionalTest (4 slots/agent) since a marshalled WRITE and
    // READ-serve run concurrently on one agent, each needing its own 2-slot group. Uses dedicated
    // buffers sized above the marshal threshold instead of sub-regions of the fixture's own (too
    // small to fit two such regions for every fixture type). Skips delta: its sender/receiver refs
    // are tied to mem_agent_0_/mem_agent_1_'s own content, not these dedicated buffers.
    TYPED_TEST(nixlServiceBidirectionalTest, SimultaneousReadAndWriteSameId) {
        if (!this->etcd_valid_) {
            GTEST_SKIP() << "NIXL_ETCD_ENDPOINTS is not set or etcd is not running";
        }
        if (!this->cuda_valid_) {
            GTEST_SKIP() << "No CUDA device available";
        }
        if constexpr (std::is_same_v<typename TestFixture::marshal_config_t,
                                     nixlMarshalDeltaConfig>) {
            GTEST_SKIP() << "See this test's header comment on why delta is out of scope here";
        }

        constexpr size_t region_size = 64 * 1024 * 1024 + 4096; // > direct_desc_threshold (64 MB).

        void *write_src = nullptr; // agent_0_: source of its WRITE to agent_1_.
        void *write_dst = nullptr; // agent_1_: destination of that WRITE.
        void *read_src = nullptr; // agent_0_: source agent_1_ reads from (agent_0_ is the peer).
        void *read_dst = nullptr; // agent_1_: destination of its READ from agent_0_.
        ASSERT_EQ(cudaMalloc(&write_src, region_size), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&write_dst, region_size), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&read_src, region_size), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&read_dst, region_size), cudaSuccess);

        nixl_reg_dlist_t agent_0_new_descs(VRAM_SEG);
        agent_0_new_descs.addDesc(
            nixlBlobDesc(reinterpret_cast<uintptr_t>(write_src), region_size, this->device_id_));
        agent_0_new_descs.addDesc(
            nixlBlobDesc(reinterpret_cast<uintptr_t>(read_src), region_size, this->device_id_));
        ASSERT_EQ(this->agent_0_->registerMem(agent_0_new_descs), NIXL_SUCCESS);
        nixl_reg_dlist_t agent_1_new_descs(VRAM_SEG);
        agent_1_new_descs.addDesc(
            nixlBlobDesc(reinterpret_cast<uintptr_t>(write_dst), region_size, this->device_id_));
        agent_1_new_descs.addDesc(
            nixlBlobDesc(reinterpret_cast<uintptr_t>(read_dst), region_size, this->device_id_));
        ASSERT_EQ(this->agent_1_->registerMem(agent_1_new_descs), NIXL_SUCCESS);

        // Deregister these buffers from the (still-alive, fixture-owned) agents before freeing the
        // device memory, so neither agent's progress thread can touch a freed region during the
        // fixture's later TearDown(). Declared after registration so the deregister dlists are in
        // scope, and (via reverse-order cleanup) so cudaFree runs only after deregistration.
        absl::Cleanup free_bufs = [&] {
            (void)this->agent_0_->deregisterMem(agent_0_new_descs);
            (void)this->agent_1_->deregisterMem(agent_1_new_descs);
            (void)cudaFree(write_src);
            (void)cudaFree(write_dst);
            (void)cudaFree(read_src);
            (void)cudaFree(read_dst);
        };

        // Refresh the metadata each agent holds for the other, so these newly-registered regions
        // (registered after the fixture's own SetUp()-time exchange) are addressable remotely.
        nixl_blob_t md_agent_0;
        nixl_blob_t md_agent_1;
        ASSERT_EQ(this->agent_0_->getLocalMD(md_agent_0), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_1_->getLocalMD(md_agent_1), NIXL_SUCCESS);
        std::string remote_name_0;
        std::string remote_name_1;
        ASSERT_EQ(this->agent_1_->loadRemoteMD(md_agent_0, remote_name_1), NIXL_SUCCESS);
        ASSERT_EQ(remote_name_1, "agent_0");
        ASSERT_EQ(this->agent_0_->loadRemoteMD(md_agent_1, remote_name_0), NIXL_SUCCESS);
        ASSERT_EQ(remote_name_0, "agent_1");

        std::vector<unsigned char> write_src_data(region_size);
        std::vector<unsigned char> read_src_data(region_size);
        // Different seeds: write_src_data and read_src_data must stay distinguishable, so a
        // WRITE/READ cross-wiring bug can't hide behind both sides happening to hold equal data.
        this->fillWithRepeatingRandomPattern(write_src_data, region_size, 42);
        this->fillWithRepeatingRandomPattern(read_src_data, region_size, 43);
        ASSERT_EQ(cudaMemcpy(write_src, write_src_data.data(), region_size, cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(read_src, read_src_data.data(), region_size, cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(write_dst, 0, region_size), cudaSuccess);
        ASSERT_EQ(cudaMemset(read_dst, 0, region_size), cudaSuccess);

        nixl_xfer_dlist_t write_local_descs(VRAM_SEG); // agent_0_'s source.
        write_local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(write_src), region_size, this->device_id_));
        nixl_xfer_dlist_t write_remote_descs(VRAM_SEG); // agent_1_'s destination.
        write_remote_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(write_dst), region_size, this->device_id_));

        nixl_xfer_dlist_t read_local_descs(VRAM_SEG); // agent_1_'s destination.
        read_local_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(read_dst), region_size, this->device_id_));
        nixl_xfer_dlist_t read_remote_descs(VRAM_SEG); // agent_0_'s source.
        read_remote_descs.addDesc(
            nixlBasicDesc(reinterpret_cast<uintptr_t>(read_src), region_size, this->device_id_));

        nixl_service_opt_args_t write_params{};
        write_params.marshalOptArgs = this->makeMarshalOptArgs();
        write_params.notif = "write_done";
        nixlServiceXferReqH *write_hndl = nullptr;
        ASSERT_EQ(this->agent_0_->createXferReq(NIXL_WRITE,
                                                write_local_descs,
                                                write_remote_descs,
                                                "agent_1",
                                                write_hndl,
                                                &write_params),
                  NIXL_SUCCESS);
        ASSERT_NE(write_hndl, nullptr);

        nixl_service_opt_args_t read_params{};
        read_params.marshalOptArgs = this->makeMarshalOptArgs();
        read_params.notif = "read_done";
        nixlServiceXferReqH *read_hndl = nullptr;
        ASSERT_EQ(
            this->agent_1_->createXferReq(
                NIXL_READ, read_local_descs, read_remote_descs, "agent_0", read_hndl, &read_params),
            NIXL_SUCCESS);
        ASSERT_NE(read_hndl, nullptr);

        ASSERT_EQ(this->agent_0_->postXferReq(write_hndl, &write_params), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_1_->postXferReq(read_hndl, &read_params), NIXL_IN_PROG);

        // getNotifs() only ever appends newly-arrived notifications, so a single shared map
        // accumulates both transfers' notifications correctly regardless of which of the two
        // completes (and so delivers its notification) first.
        nixl_notifs_t notifs_at_agent_0; // Will hold the READ's notif, keyed "agent_1".
        nixl_notifs_t notifs_at_agent_1; // Will hold the WRITE's notif, keyed "agent_0".
        while (this->agent_0_->getXferStatus(write_hndl) == NIXL_IN_PROG ||
               this->agent_1_->getXferStatus(read_hndl) == NIXL_IN_PROG) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs_at_agent_0), NIXL_SUCCESS);
            ASSERT_LE(notifs_at_agent_0.size(), 1);
            ASSERT_EQ(this->agent_1_->getNotifs(notifs_at_agent_1), NIXL_SUCCESS);
            ASSERT_LE(notifs_at_agent_1.size(), 1);
        }
        ASSERT_EQ(this->agent_0_->getXferStatus(write_hndl), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_1_->getXferStatus(read_hndl), NIXL_SUCCESS);

        std::vector<unsigned char> write_dst_check(region_size);
        std::vector<unsigned char> read_dst_check(region_size);
        ASSERT_EQ(
            cudaMemcpy(write_dst_check.data(), write_dst, region_size, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpy(read_dst_check.data(), read_dst, region_size, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(write_dst_check, write_src_data) << "the WRITE's data was corrupted or misrouted";
        ASSERT_EQ(read_dst_check, read_src_data) << "the READ's data was corrupted or misrouted";

        // The WRITE's notif goes to its receiver (agent_1_); the READ's notif goes to its peer
        // (agent_0_) - see tryCompleteReadReceive. Both statuses are terminal at this point, but
        // the very last progressService() tick that set a status terminal may be the same one
        // that queued its notif, so one more round of polling may still be needed to observe it.
        while (notifs_at_agent_1.empty()) {
            ASSERT_EQ(this->agent_1_->getNotifs(notifs_at_agent_1), NIXL_SUCCESS);
        }
        ASSERT_EQ(notifs_at_agent_1.size(), 1);
        ASSERT_EQ(notifs_at_agent_1["agent_0"].size(), 1);
        ASSERT_EQ(notifs_at_agent_1["agent_0"].front(), "write_done");
        while (notifs_at_agent_0.empty()) {
            ASSERT_EQ(this->agent_0_->getNotifs(notifs_at_agent_0), NIXL_SUCCESS);
        }
        ASSERT_EQ(notifs_at_agent_0.size(), 1);
        ASSERT_EQ(notifs_at_agent_0["agent_1"].size(), 1);
        ASSERT_EQ(notifs_at_agent_0["agent_1"].front(), "read_done");

        ASSERT_EQ(this->agent_0_->releaseXferReq(write_hndl), NIXL_SUCCESS);
        ASSERT_EQ(this->agent_1_->releaseXferReq(read_hndl), NIXL_SUCCESS);
    }
} // namespace services
} // namespace gtest
