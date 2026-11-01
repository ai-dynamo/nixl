/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "../allocation_counter.h"
#include "nvtx_events.h"
#include "nvtx_span.h"
#include "nvtx_span.h"

namespace {

namespace nvtx = nixl::trace::nvtx_internal;

constexpr std::string_view fixed_names[] = {
    "nixl::createXferReq",
    "nixl::postXferReq.write",
    "nixl::postXferReq.read",
    "nixl::xfer.complete",
    "nixl::releaseXferReq",
    "nixl::submit",
    "nixl::wire.submitted",
    "nixl::wire.completed",
};

[[nodiscard]] std::vector<nvtxStringHandle_t>
unregisteredHandles() {
    return std::vector<nvtxStringHandle_t>(nvtx::registerSpanNames(nullptr).size(), nullptr);
}

[[nodiscard]] std::vector<nvtxStringHandle_t>
registeredHandles() {
    std::vector<nvtxStringHandle_t> handles = unregisteredHandles();
    for (std::size_t i = 0; i < handles.size(); ++i) {
        handles[i] = reinterpret_cast<nvtxStringHandle_t>(static_cast<std::uintptr_t>(i + 1));
    }
    return handles;
}

} // namespace

TEST(NvtxEvents, FixedNameWithoutRegistrationCarriesItsOwnText) {
    const auto handles = unregisteredHandles();
    for (const std::string_view name : fixed_names) {
        nvtx::NvtxEvent event;
        nvtx::eventForName(event, name, nixl::trace::Kind::Generic, nullptr, handles);
        EXPECT_EQ(event.attributes.messageType, NVTX_MESSAGE_TYPE_ASCII) << name;
        EXPECT_EQ(std::string_view(event.attributes.message.ascii), name);
        EXPECT_NE(event.attributes.message.ascii, event.message.data()) << name;
    }
}

TEST(NvtxEvents, FixedNameCarriesItsRegisteredHandle) {
    const auto handles = registeredHandles();
    std::set<nvtxStringHandle_t> seen;
    for (const std::string_view name : fixed_names) {
        nvtx::NvtxEvent event;
        nvtx::eventForName(event, name, nixl::trace::Kind::Generic, nullptr, handles);
        EXPECT_EQ(event.attributes.messageType, NVTX_MESSAGE_TYPE_REGISTERED) << name;
        EXPECT_NE(event.attributes.message.registered, nullptr) << name;
        seen.insert(event.attributes.message.registered);
    }
    EXPECT_EQ(seen.size(), std::size(fixed_names));
}

TEST(NvtxEvents, OtherNameIsCopiedIntoTheEvent) {
    const auto handles = registeredHandles();
    for (const std::string_view name : {std::string_view("rail0.write"),
                                        std::string_view("nixl::postXferReq"),
                                        std::string_view("nixl::submit.")}) {
        nvtx::NvtxEvent event;
        nvtx::eventForName(event, name, nixl::trace::Kind::Metadata, nullptr, handles);
        EXPECT_EQ(event.attributes.messageType, NVTX_MESSAGE_TYPE_ASCII) << name;
        EXPECT_EQ(event.attributes.message.ascii, event.message.data()) << name;
        EXPECT_EQ(std::string_view(event.attributes.message.ascii), name);
    }
}

TEST(NvtxEvents, OtherNameThatDoesNotFitIsRegistered) {
    const std::string name(nvtx::NvtxEvent{}.message.size(), 'x');
    nvtx::NvtxEvent event;
    nvtx::eventForName(event, name, nixl::trace::Kind::Metadata, nullptr, unregisteredHandles());
    EXPECT_EQ(event.attributes.messageType, NVTX_MESSAGE_TYPE_REGISTERED);
}

TEST(NvtxEvents, FixedAndShortNamesAllocateNothing) {
    const auto unregistered = unregisteredHandles();
    const auto registered = registeredHandles();
    nvtx::NvtxEvent event;
    const gtest::AllocationCounter allocations;
    nvtx::eventForName(
        event, "nixl::postXferReq.write", nixl::trace::Kind::CommSend, nullptr, unregistered);
    nvtx::eventForName(
        event, "nixl::wire.completed", nixl::trace::Kind::Metadata, nullptr, registered);
    nvtx::eventForName(event, "rail0.write", nixl::trace::Kind::Metadata, nullptr, registered);
    EXPECT_EQ(allocations.count(), 0u);
}

TEST(NvtxSpan, AttributesThatFitAllocateNothing) {
    const nvtx::PayloadSchemaIds schema_ids{};
    std::make_unique<nvtx::NvtxSpan>(nullptr, schema_ids).reset();

    const gtest::AllocationCounter allocations;
    {
        const auto span = std::make_unique<nvtx::NvtxSpan>(nullptr, schema_ids);
        span->addAttribute(
            "nixl.traceparent",
            std::string_view("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"));
        span->addAttribute("nixl.xfer_request_id", std::string_view("18364758544493064720"));
        span->addAttribute("nixl.xfer_execution", std::int64_t{3});
        span->addAttribute("ratio", 0.5);
    }
    {
        nvtx::NvtxSpan range(nullptr, schema_ids);
        for (std::size_t i = 0; i < nvtx::NvtxSpan::inlineAttributes; ++i) {
            range.addAttribute("nixl.start_us", static_cast<std::int64_t>(i));
        }
    }
    EXPECT_EQ(allocations.count(), 0u);
}

TEST(NvtxSpan, AttributesThatDoNotFitGoToTheHeap) {
    const std::string long_key(nvtx::NvtxSpan::inlineKeySize, 'k');
    const std::string long_value(nvtx::NvtxSpan::inlineValueSize, 'v');
    for (const auto add :
         {+[](nvtx::NvtxSpan &span, const std::string &key, const std::string &) {
              span.addAttribute(key, std::int64_t{1});
          },
          +[](nvtx::NvtxSpan &span, const std::string &, const std::string &value) {
              span.addAttribute("value", std::string_view(value));
          }}) {
        const gtest::AllocationCounter allocations;
        {
            nvtx::NvtxSpan span(nullptr, {});
            span.addAttribute("first", std::int64_t{0});
            add(span, long_key, long_value);
            span.addAttribute("after", std::int64_t{2});
        }
        EXPECT_GT(allocations.count(), 0u);
    }

    const gtest::AllocationCounter allocations;
    {
        nvtx::NvtxSpan span(nullptr, {});
        for (std::size_t i = 0; i <= nvtx::NvtxSpan::inlineAttributes; ++i) {
            span.addAttribute("nixl.start_us", static_cast<std::int64_t>(i));
        }
    }
    EXPECT_GT(allocations.count(), 0u);
}
