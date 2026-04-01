/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

/*
 * LTTng-UST tracepoint definitions for NIXL libfabric backend.
 * All timed operations use begin/end pairs for duration measurement.
 *
 * Usage:
 *   lttng create nixl-session
 *   lttng enable-event -u 'nixl_libfabric:*'
 *   lttng start
 *   <run workload>
 *   lttng stop && lttng destroy
 *   babeltrace ~/lttng-traces/nixl-session*
 */

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER nixl_libfabric

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "libfabric_tracepoints.h"

#if HAVE_LIBLTTNG_UST == 1

#if !defined(NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TRACEPOINTS_H) || \
    defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TRACEPOINTS_H

#include <lttng/tracepoint.h>

/* ===== Transfer lifecycle ===== */

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    transfer_begin,
    LTTNG_UST_TP_ARGS(int,
                      dev,
                      int,
                      op_type,
                      size_t,
                      transfer_size,
                      int,
                      num_rails,
                      int,
                      use_striping,
                      uint16_t,
                      xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(int, op_type, op_type)
                                lttng_ust_field_integer(size_t, transfer_size, transfer_size)
                                    lttng_ust_field_integer(int, num_rails, num_rails)
                                        lttng_ust_field_integer(int, use_striping, use_striping)
                                            lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    transfer_submitted,
    LTTNG_UST_TP_ARGS(int, dev, size_t, submitted_count, uint16_t, xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(size_t, submitted_count, submitted_count)
                                lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

/* ===== RDMA write ===== */

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    post_write_begin,
    LTTNG_UST_TP_ARGS(int, dev, size_t, rail_id, size_t, length, uint16_t, xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(size_t, rail_id, rail_id)
                                lttng_ust_field_integer(size_t, length, length)
                                    lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    post_write_end,
    LTTNG_UST_TP_ARGS(int, dev, size_t, rail_id, size_t, length, int, retries, uint16_t, xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(size_t, rail_id, rail_id)
                                lttng_ust_field_integer(size_t, length, length)
                                    lttng_ust_field_integer(int, retries, retries)
                                        lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

/* ===== RDMA read ===== */

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    post_read_begin,
    LTTNG_UST_TP_ARGS(int, dev, size_t, rail_id, size_t, length, uint16_t, xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(size_t, rail_id, rail_id)
                                lttng_ust_field_integer(size_t, length, length)
                                    lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    post_read_end,
    LTTNG_UST_TP_ARGS(int, dev, size_t, rail_id, size_t, length, int, retries, uint16_t, xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(size_t, rail_id, rail_id)
                                lttng_ust_field_integer(size_t, length, length)
                                    lttng_ust_field_integer(int, retries, retries)
                                        lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

/* ===== Send ===== */

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    post_send_begin,
    LTTNG_UST_TP_ARGS(int, dev, size_t, rail_id, size_t, length, uint16_t, xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(size_t, rail_id, rail_id)
                                lttng_ust_field_integer(size_t, length, length)
                                    lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    post_send_end,
    LTTNG_UST_TP_ARGS(int, dev, size_t, rail_id, size_t, length, int, retries, uint16_t, xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(size_t, rail_id, rail_id)
                                lttng_ust_field_integer(size_t, length, length)
                                    lttng_ust_field_integer(int, retries, retries)
                                        lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

/* ===== Local completion events (initiator side) ===== */

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    local_transfer_completion,
    LTTNG_UST_TP_ARGS(int, dev, size_t, rail_id, int, op_type, uint16_t, xfer_id),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, dev, dev)
                            lttng_ust_field_integer(size_t, rail_id, rail_id)
                                lttng_ust_field_integer(int, op_type, op_type)
                                    lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)))

/* ===== Remote completion events (target side) ===== */

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    remote_write_completion,
    LTTNG_UST_TP_ARGS(size_t, rail_id, uint16_t, agent_idx, uint16_t, xfer_id, size_t, length),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(size_t, rail_id, rail_id)
                            lttng_ust_field_integer(uint16_t, agent_idx, agent_idx)
                                lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)
                                    lttng_ust_field_integer(size_t, length, length)))

LTTNG_UST_TRACEPOINT_EVENT(
    nixl_libfabric,
    recv_completion,
    LTTNG_UST_TP_ARGS(size_t, rail_id, uint16_t, agent_idx, uint16_t, xfer_id, size_t, length),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(size_t, rail_id, rail_id)
                            lttng_ust_field_integer(uint16_t, agent_idx, agent_idx)
                                lttng_ust_field_integer(uint16_t, xfer_id, xfer_id)
                                    lttng_ust_field_integer(size_t, length, length)))

#endif /* !defined(NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TRACEPOINTS_H) || \
          defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ) */

#include <lttng/tracepoint-event.h>

#else /* HAVE_LIBLTTNG_UST != 1 */

#define lttng_ust_tracepoint(...)

#endif /* HAVE_LIBLTTNG_UST == 1 */

/*
 * Wrapper macros — compile to nothing when LTTng is disabled.
 */

#define NIXL_TP_OP_WRITE 0
#define NIXL_TP_OP_READ 1
#define NIXL_TP_OP_SEND 2

#define NIXL_TRACE_TRANSFER_BEGIN(dev, op, sz, nr, stripe, xid) \
    lttng_ust_tracepoint(nixl_libfabric, transfer_begin, dev, op, sz, nr, stripe, xid)
#define NIXL_TRACE_TRANSFER_SUBMITTED(dev, cnt, xid) \
    lttng_ust_tracepoint(nixl_libfabric, transfer_submitted, dev, cnt, xid)

#define NIXL_TRACE_POST_WRITE_BEGIN(dev, rail, len, xid) \
    lttng_ust_tracepoint(nixl_libfabric, post_write_begin, dev, rail, len, xid)
#define NIXL_TRACE_POST_WRITE_END(dev, rail, len, att, xid) \
    lttng_ust_tracepoint(nixl_libfabric, post_write_end, dev, rail, len, att, xid)

#define NIXL_TRACE_POST_READ_BEGIN(dev, rail, len, xid) \
    lttng_ust_tracepoint(nixl_libfabric, post_read_begin, dev, rail, len, xid)
#define NIXL_TRACE_POST_READ_END(dev, rail, len, att, xid) \
    lttng_ust_tracepoint(nixl_libfabric, post_read_end, dev, rail, len, att, xid)

#define NIXL_TRACE_POST_SEND_BEGIN(dev, rail, len, xid) \
    lttng_ust_tracepoint(nixl_libfabric, post_send_begin, dev, rail, len, xid)
#define NIXL_TRACE_POST_SEND_END(dev, rail, len, att, xid) \
    lttng_ust_tracepoint(nixl_libfabric, post_send_end, dev, rail, len, att, xid)

#define NIXL_TRACE_LOCAL_TRANSFER_COMPLETION(dev, rail, op, xid) \
    lttng_ust_tracepoint(nixl_libfabric, local_transfer_completion, dev, rail, op, xid)

#define NIXL_TRACE_REMOTE_WRITE_COMPLETION(rail, aidx, xid, len) \
    lttng_ust_tracepoint(nixl_libfabric, remote_write_completion, rail, aidx, xid, len)

#define NIXL_TRACE_RECV_COMPLETION(rail, aidx, xid, len) \
    lttng_ust_tracepoint(nixl_libfabric, recv_completion, rail, aidx, xid, len)
