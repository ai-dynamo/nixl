<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Integrated PD sender-component evidence

## Historical measured reference, not the refreshed PR head

Code: `73f057608e9b6aa92805de733ec42f41a79cbbdb`.
Two H20 hosts, DOCA 3.5.0098, CUDA 12.8. Workload: 27 layers, 64 rows per
layer/peer, 576 bytes per row, four logical peers, three rotating windows.
One selected GPU per host; receiver agents share a GPU. Not four receiver
GPUs, multi-NIC or peer-MR evidence.

The research stack combined compact affine WRITE, per-QP-group progress
owners, retained notification receive-buffer ownership, early fenced SEND
on the data SQ, and balanced LAG placement. No payload copy, completion check
or notification was removed.

Five alternating fresh-process pairs, 20 warmup and 600 measured requests
per arm/pair, against UCX no-pack + SGL with unrestricted paths:

| Pooled metric | UCX | Integrated GPUNETIO |
| --- | ---: | ---: |
| Complete sender request p50 | 284.618 us | 184.159 us |
| Complete sender request p99 | 334.239 us | 195.762 us |
| post + DONE p50 | 159.946 us | 164.311 us |
| Primary requests >=1 ms | 1/3000 | 0/3000 |

Median paired complete-request p50 reduction **35.4178%**, range
34.344–35.572%, all five p50/p99 pairs improve. Percentages are medians of
paired reductions; table values are pooled quantiles. Do not add independent
experiments or derive an exact paired percentage from rounded pooled values.

The primary timer includes dynamic planning, fresh preparation, posting and
request DONE. Preparation savings outweigh the remaining 4.3645-us post+DONE
difference. This is not SGLang serving TTFT/TPOT, GEMM-overlap, NIXLBench
bandwidth or peer-MR performance. Receiver GPU validation/TCP ACK is outside
the interval and has millisecond tails. UCX immediate AM API completion and
GPUNETIO SEND-CQ completion are not identical hardware-CQ milestones.

## LAG boundary

The reference used a startup-only experimental DevX adapter to balance four
active payload QPs across two physical bond members. Both backends used both
members: UCX ~51.85/48.15, GPUNETIO 50/50; UCX was not restricted to one path.
The adapter changed only owned-QP affinity, not global bond settings, and is
not included as a supported production dependency in this Draft.

Public GPUNetIO main `b2b7a291ec70fdeeb1b27ba4b0ab8c8e260a729b` exposes
`doca_verbs_qp_attr_set_lag_tx_port_affinity()` and capability queries in
open-source mode; its changelog places the feature in 4.1.0. Latest tagged
release v4.0.1 lacks it. The 5.0.0 development tree adds SDK wrappers, but
does not identify the first SDK binary exporting it. SDK 3.5.0098 lacks those
symbols. Qualification of a compatible supported path remains necessary.

Sources: [pinned changelog](https://github.com/NVIDIA-DOCA/gpunetio/blob/b2b7a291ec70fdeeb1b27ba4b0ab8c8e260a729b/CHANGELOG.md),
[public header](https://github.com/NVIDIA-DOCA/gpunetio/blob/b2b7a291ec70fdeeb1b27ba4b0ab8c8e260a729b/include/host/doca_verbs.h),
[NCCL caller](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/transport/net_ib/gdaki/gin_host_gdaki.cc).
NCCL sets affinity during RESET->INIT before traffic and probes capability
only for OPEN mode. NIXL SDK-backed integration and public per-QP readback are
not yet qualified. Do not cast opaque QP handles or require unsupported symbols.

## Correctness at the measured reference

- 131,152 rolling operations, 24 live handles, changing slot/peer assignments,
  SQ/CQ and 16-bit WQE-counter wrap; exact operation/notification counts,
  full GPU payload/guard/arrival checks and deliberate corruption detection.
- 10,020 operations from two source agents into one receiver engine with
  distinct-source data/notifications. Sources share one GPU/process.
- 1,200 rounds of notified WRITE->READ->WRITE plus standalone notifications,
  4,800 notifications, content/count/FIFO and payload/guard validation.
- Real stale-rkey WRITE error followed by preposted SEND flush on the same SQ;
  API error, no application notification and successful cleanup.
- Dense/fragmented READ, descriptor/MR boundaries, partial shapes and reposts.

These results do not replace qualification of the exact submitted source,
plugin ABI and immutable configuration. The native command in README does not
reproduce the historical 35.4% configuration without the LAG path.
