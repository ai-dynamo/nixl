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

#include <cstddef>
#include <cstdint>

#include <string>

#include "absl/cleanup/cleanup.h"
#include "gtest/gtest.h"
#include "nixl_types.h"

#ifndef TCPXO_STUB_RXDM_DXS
#include "dxs/client/dxs-client-types.h"
#else
#include "rxdm_dxs_stub.h"
#endif
#include "tcpxo_nixl_memory_metadata.h"

namespace tcpxo {
namespace {

    TEST(TcpxoNixlMemoryMetadataTest, SerializeDeserializeRoundtrip) {
        dxs::Reg reg_handle = 12345;
        void *start_addr = reinterpret_cast<void *>(0xdeadbeef);
        size_t size = 1024 * 1024;
        int dmabuf_fd = 42;
        uint8_t fastrak_idx = 1;

        nixlTcpxoLocalMemoryMetadata original_md(
            reg_handle, start_addr, size, dmabuf_fd, fastrak_idx);
        std::string serialized_str;
        EXPECT_EQ(SerializeMemoryMetadata(&original_md, serialized_str), NIXL_SUCCESS);
        EXPECT_FALSE(serialized_str.empty());

        nixlTcpxoLocalMemoryMetadata *deserialized_md = nullptr;
        EXPECT_EQ(DeserializeMemoryMetadata(serialized_str, deserialized_md), NIXL_SUCCESS);
        ASSERT_NE(deserialized_md, nullptr);
        absl::Cleanup md_cleanup = [&] { delete deserialized_md; };

        EXPECT_EQ(deserialized_md->GetFastrakIdx(), fastrak_idx);
        EXPECT_EQ(deserialized_md->GetMemHandle(), original_md.GetMemHandle());
    }

    TEST(TcpxoNixlMemoryMetadataTest, DeserializeInvalidStringFailsGracefully) {
        std::string invalid_str = "this_is_an_invalid_serialized_payload";
        nixlTcpxoLocalMemoryMetadata *deserialized_md = nullptr;

        nixl_status_t status = DeserializeMemoryMetadata(invalid_str, deserialized_md);
        EXPECT_NE(status, NIXL_SUCCESS);
        EXPECT_EQ(deserialized_md, nullptr);
    }

} // namespace
} // namespace tcpxo

int
main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
