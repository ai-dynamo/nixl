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

#ifndef GPUDIRECT_TCPXO_TEST_COMMON_H_
#define GPUDIRECT_TCPXO_TEST_COMMON_H_

#include <gmock/gmock.h>
#include <memory>
#include <utility>
#include <variant>

#include "absl/base/nullability.h"
#include "absl/time/time.h"

#include "control_channel.h"
#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/dxs-client-interface.h"
#else
#include "rxdm_dxs_stub.h"
#endif

namespace tcpxo {
namespace test {

    inline constexpr absl::Duration kHeartbeatSendPeriod = absl::Seconds(1);
    inline constexpr absl::Duration kHeartbeatTimeout = absl::Seconds(10);
    inline constexpr int kDefaultListenSocketPort = 55555;
    inline constexpr absl::string_view kDefaultNicAddr = "127.0.0.1";

    inline std::unique_ptr<ControlChannel>
    MakeControlChannel(
        ConnectionCallback absl_nonnull conn_cb = [](...) {},
        DisconnectionCallback absl_nonnull disc_cb = [](...) {},
        NotificationCallback absl_nonnull notif_cb = [](...) {},
        DxsAddressExchangeCallback absl_nonnull dxs_cb = [](...) {}) {
        return std::make_unique<ControlChannel>(
            "lo",
            kHeartbeatSendPeriod,
            kHeartbeatTimeout,
            [conn_cb = std::move(conn_cb),
             disc_cb = std::move(disc_cb),
             notif_cb = std::move(notif_cb),
             dxs_cb = std::move(dxs_cb)](PeerEvent &&event) mutable {
                std::visit(
                    [&conn_cb, &disc_cb, &notif_cb, &dxs_cb](auto &&event) {
                        using T = std::decay_t<decltype(event)>;
                        if constexpr (std::is_same_v<T, ConnectedEvent>) {
                            conn_cb(event.handle,
                                    event.socket_addr,
                                    event.service_addr,
                                    event.agent_name,
                                    event.dxs_msg);
                        } else if constexpr (std::is_same_v<T, DisconnectedEvent>) {
                            disc_cb(event.handle, event.socket_addr, event.service_addr);
                        } else if constexpr (std::is_same_v<T, MessageReceivedEvent>) {
                            auto [msg, is_valid_message] = Worker::ParseMessage(event.msg_data);
                            if (!is_valid_message) {
                                return;
                            }

                            if (msg.has_notif()) {
                                notif_cb(event.handle,
                                         event.service_addr,
                                         std::move(*msg.mutable_notif()));
                            } else if (msg.has_dxs()) {
                                dxs_cb(event.handle, event.service_addr, msg.dxs());
                            }
                        }
                    },
                    event);
            });
    }

    class MockListenSocket : public dxs::ListenSocketInterface {
    public:
        explicit MockListenSocket(int port, absl::string_view address)
            : port_(port),
              address_(std::string(address)){};

        std::optional<absl::Status>
        SocketReady() override {
            return absl::OkStatus();
        }

        int
        Port() const override {
            return port_;
        }

        std::string
        Address() const override {
            return address_;
        }

        MOCK_METHOD(absl::StatusOr<absl_nullable std::unique_ptr<dxs::RecvSocketInterface>>,
                    Accept,
                    (),
                    (override));

    private:
        int port_;
        std::string address_;
    };

} // namespace test
} // namespace tcpxo

#endif // GPUDIRECT_TCPXO_TEST_COMMON_H_
