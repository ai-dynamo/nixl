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

#ifndef GPUDIRECT_TCPXO_STUB_RXDM_DXS_H_
#define GPUDIRECT_TCPXO_STUB_RXDM_DXS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"

#define ASSIGN_OR_RETURN(lvalue, statusor)     \
    {                                          \
        auto status_or_lvalue_ = statusor;     \
        if (!status_or_lvalue_.ok()) {         \
            return status_or_lvalue_.status(); \
        }                                      \
    }                                          \
    lvalue = std::move(*statusor);

#define RETURN_IF_ERROR(status_func)     \
    {                                    \
        const auto status = status_func; \
        if (!status.ok()) {              \
            return status;               \
        }                                \
    }

namespace dxs {

typedef uint64_t Reg;
typedef uint64_t OpId;

inline constexpr absl::string_view kLlcmDeviceDirectory = "/sys/bus/pci/devices";
constexpr char kDefaultDxsAddr[] = "169.254.169.254";
constexpr char kDefaultDxsPort[] = "55555";
constexpr Reg kInvalidRegistration = 0ul;

class OpInterface {
public:
    virtual ~OpInterface() = default;
};

class LinearizedRecvOpInterface : public OpInterface {
public:
    virtual ~LinearizedRecvOpInterface() = default;

    virtual std::optional<absl::StatusOr<uint64_t>>
    Test() {
        constexpr uint64_t dummy_size = 10;
        return dummy_size;
    }
};

class SendOpInterface : public OpInterface {
public:
    virtual ~SendOpInterface() = default;

    virtual std::optional<absl::Status>
    Test() {
        return std::nullopt;
    }
};

class ConnectedSocketInterface {
public:
    virtual ~ConnectedSocketInterface() = default;

    virtual std::optional<absl::Status>
    SocketReady() {
        return absl::OkStatus();
    }
};

class SendSocketInterface : public ConnectedSocketInterface {
public:
    virtual ~SendSocketInterface() = default;

    virtual absl::StatusOr<std::unique_ptr<SendOpInterface>>
    Send(uint64_t offset, size_t size, Reg reg_handle) {
        return std::make_unique<SendOpInterface>();
    }
};

class RecvSocketInterface : public ConnectedSocketInterface {
public:
    virtual ~RecvSocketInterface() = default;

    virtual absl::StatusOr<std::unique_ptr<LinearizedRecvOpInterface>>
    RecvLinearized(uint64_t offset, size_t size, Reg reg_handle) {
        return std::make_unique<LinearizedRecvOpInterface>();
    }
};

class ListenSocketInterface {
public:
    virtual ~ListenSocketInterface() = default;

    virtual std::optional<absl::Status>
    SocketReady() {
        return absl::OkStatus();
    }

    virtual int
    Port() const {
        int port = 0;
        if (!absl::SimpleAtoi(kDefaultDxsPort, &port)) {
            return 0;
        }
        return port;
    }

    virtual std::string
    Address() const {
        return std::string(kDefaultDxsAddr);
    };

    virtual absl::StatusOr<absl_nullable std::unique_ptr<RecvSocketInterface>>
    Accept() {
        return std::make_unique<RecvSocketInterface>();
    };
};

class DxsClientInterface {
public:
    virtual ~DxsClientInterface() = default;
    virtual absl::StatusOr<std::unique_ptr<ListenSocketInterface>>
    Listen() = 0;
    virtual absl::StatusOr<std::unique_ptr<SendSocketInterface>>
    Connect(std::string addr, uint16_t port) = 0;
};

class DxsClient : public DxsClientInterface {
public:
    static absl::StatusOr<std::unique_ptr<DxsClientInterface>>
    Create(std::string nic_addr,
           std::string dxs_addr,
           std::string dxs_port,
           std::string source_port,
           bool enable_llcm = false,
           std::string llcm_device_directory = std::string(kLlcmDeviceDirectory),
           bool send_close_on_teardown = true) {
        return std::make_unique<DxsClient>();
    }

    absl::StatusOr<std::unique_ptr<ListenSocketInterface>>
    Listen() override {
        return std::make_unique<ListenSocketInterface>();
    }

    absl::StatusOr<std::unique_ptr<SendSocketInterface>>
    Connect(std::string addr, uint16_t port) override {
        return std::make_unique<SendSocketInterface>();
    }
};

} // namespace dxs

namespace tcpdirect {

inline constexpr char kSysfsPciDevicesPath[] = "/sys/bus/pci/devices";
inline constexpr absl::string_view kNvidiaVendorId = "0x10de";
inline constexpr absl::string_view kH100DeviceId = "0x2330";

class BufferManagerClientInterface {
public:
    virtual ~BufferManagerClientInterface() = default;

    virtual absl::StatusOr<dxs::Reg>
    RegBuf(int fd, size_t size) = 0;

    virtual absl::Status
    DeregBuf(dxs::Reg reg_handle) = 0;
};

class BufferManagerClient : public BufferManagerClientInterface {
public:
    static absl::StatusOr<std::unique_ptr<BufferManagerClientInterface>>
    Create(absl::string_view ip_addr) {
        return absl::WrapUnique(new BufferManagerClient());
    }

    absl::StatusOr<dxs::Reg>
    RegBuf(int fd, size_t size) {
        return dxs::kInvalidRegistration;
    }

    absl::Status
    DeregBuf(dxs::Reg reg_handle) {
        return absl::OkStatus();
    }

private:
    BufferManagerClient() = default;
};

class GetNicMappingResp {
public:
    class NicInfo {
    public:
        const std::vector<std::string> &
        closest_nic_ip() const {
            return ips_;
        }

        std::vector<std::string> ips_ = {"127.0.0.1"};
    };

    const absl::flat_hash_map<std::string, NicInfo> &
    pci_nic_map() const {
        return pci_nic_map_;
    }

    absl::flat_hash_map<std::string, NicInfo> pci_nic_map_;
};

inline absl::Status
rxdm_running() {
    return absl::OkStatus();
}

inline std::optional<GetNicMappingResp>
get_nic_mapping() {
    GetNicMappingResp resp;
    return resp;
}

inline int
list_vendor_devices(const char *parent_dir_path,
                    std::vector<std::string> &candidates,
                    absl::string_view vendor_id,
                    std::optional<absl::string_view> device_id = std::nullopt) {
    return 0;
}

} // namespace tcpdirect

#endif // GPUDIRECT_TCPXO_STUB_RXDM_DXS_H_
