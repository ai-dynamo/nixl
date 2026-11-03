# NIXL/UCX NIXL-EPv2 Write with IMM

## Rationale

NIXL-EPv2 kernels exchange tokens between GPUs. During dispatch, a peer needs to know when it can forward a token. During combine, it needs to know when a token’s results are ready to reduce. We tested two ways to signal arrival with relaxed PCIe ordering (Hard Requirement).

- **Atomics.** After writing a token data batch, the sender increments a counter on the receiving GPU. The receiver polls the counter to learn how many tokens are ready.
- **Write with Immediate (IMM).** The sender includes a 32-bit value in the data write. The receiver reads that value from its completion queue and knows exactly which token slot or result is ready.

Atomics fell short of the required performance. IMM identifies ready tokens without a separate remote update and had the lowest dispatch + combine latency in all 12 Prenyx cases.

### Prenyx benchmark

4 nodes × 8 GPUs · 26 communication SMs per implementation.

| Format | Batch | IMM µs | DeepEP µs | Atomics µs | Atomics vs DeepEP | IMM vs Atomics | IMM vs DeepEP |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 1 | 178.93 | 191.37 | 195.30 | +2.05% | −8.38% | $\color{green}{-6.50\%}$ |
| BF16 | 16 | 205.76 | 234.09 | 235.48 | +0.59% | −12.62% | $\color{green}{-12.10\%}$ |
| BF16 | 128 | 368.62 | 409.32 | 403.97 | −1.31% | −8.75% | $\color{green}{-9.94\%}$ |
| BF16 | 256 | 592.92 | 646.12 | 631.11 | −2.32% | −6.05% | $\color{green}{-8.23\%}$ |
| BF16 | 1024 | 1913.10 | 2015.04 | 2083.75 | +3.41% | −8.19% | $\color{green}{-5.06\%}$ |
| BF16 | 4096 | 6951.55 | 7275.29 | 7442.82 | +2.30% | −6.60% | $\color{green}{-4.45\%}$ |
| FP8 | 1 | 172.77 | 189.30 | 188.89 | −0.22% | −8.54% | $\color{green}{-8.73\%}$ |
| FP8 | 16 | 185.27 | 224.26 | 219.11 | −2.30% | −15.45% | $\color{green}{-17.39\%}$ |
| FP8 | 128 | 318.28 | 362.50 | 353.43 | −2.50% | −9.95% | $\color{green}{-12.20\%}$ |
| FP8 | 256 | 495.65 | 545.47 | 539.64 | −1.07% | −8.15% | $\color{green}{-9.13\%}$ |
| FP8 | 1024 | 1514.02 | 1612.18 | 1634.70 | +1.40% | −7.38% | $\color{green}{-6.09\%}$ |
| FP8 | 4096 | 5343.66 | 5558.77 | 5799.04 | +4.32% | −7.85% | $\color{green}{-3.87\%}$ |

## Design

- **Where IMM applies.** Hybrid kernels use IMM between NVLink domains. Transfers within a domain use GPU-memory signaling.
- **IMM usage.** The last packet of a token’s final RDMA write to a peer carries the immediate value. The receiving NIC posts a completion queue entry (CQE) once the data from that write and earlier writes on the same queue pair is visible, even with relaxed PCIe ordering ([see below](#hardware-ordering-references)). The GPU polls the CQE with a system-scope acquire before reading the token.

- **Dispatch.** The write IMM identifies the receiving warp’s logical channel and the token’s buffer slot. The warp can forward whichever of its tokens arrives first.
- **Combine.** Each contributing remote domain attaches the original token ID to its Write with Immediate for that token. The receiving warp counts down the domains still outstanding and reduces the token when the count reaches zero.
- **Shared completion queues.** Each warp scans a shared completion queue (CQ) with its own cursor. It skips entries for other warps and counts only its own.
- **Queue reuse.** At the start of the next dispatch, one warp per data queue pair returns the receive credits consumed by dispatch and combine, in one update per used peer queue.

## NIXL IMM API

<style>
.nixl-imm-signature, .nixl-imm-signature * { background: transparent !important; }
</style>

The GPU kernels use three calls on the UCX direct backend.

#### <span style="color: #037ACC;">nixlPutWithIMM</span> — Send a write with immediate

<div class="nixl-imm-signature" style="font-size: 85%;" markdown="1">

```cpp
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlPutWithIMM(const nixlMemViewElem &src,
               const nixlMemViewElem &dst,
               size_t size,
               uint32_t immediate,
               unsigned channel_id = 0,
               uint64_t flags = 0,
               nixlGpuXferStatusH *xfer_status = nullptr);
```

</div>

Posts an RDMA write from <span style="color: #c7254e;">src</span> to <span style="color: #c7254e;">dst</span>. The receiver gets the 32-bit <span style="color: #c7254e;">immediate</span> value in its CQE.

#### <span style="color: #037ACC;">nixlPollIMM</span> — Poll a receive completion with IMM

<div class="nixl-imm-signature" style="font-size: 85%;" markdown="1">

```cpp
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlPollIMM(const nixlMemViewElem &peer,
            unsigned channel_id,
            uint64_t *consumer_index,
            uint32_t *immediate);
```

</div>

Polls the receive CQ for <span style="color: #c7254e;">peer</span> on <span style="color: #c7254e;">channel_id</span> at the calling thread’s private consumer index. On success, the call fills the <span style="color: #c7254e;">immediate</span> parameter, advances the cursor (<span style="color: #c7254e;">consumer_index</span>) by one, and returns <span style="color: #c7254e;">NIXL_SUCCESS</span>. An empty poll returns <span style="color: #c7254e;">NIXL_IN_PROG</span> and leaves the cursor in place.

#### <span style="color: #037ACC;">nixlAdvanceRx</span> — Return receive credits

<div class="nixl-imm-signature" style="font-size: 85%;" markdown="1">

```cpp
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD,
         nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::DYNAMIC>
__device__ nixl_status_t
nixlAdvanceRx(const nixlMemViewElem &peer,
              unsigned channel_id,
              uint64_t consumed,
              uint64_t *consumer_index);
```

</div>

Once all readers are done, one thread calls this for the peer and channel. It advances the shared CQ index by <span style="color: #c7254e;">consumed</span>, returns that many receive credits, and stores the new index in <span style="color: #c7254e;">consumer_index</span>.

## UCX IMM API

When UCX_RC_GDA_RX_QUEUE_LEN is greater than zero, UCX creates a private receive queue (RQ) and receive completion queue (CQ) for each GDA queue pair (QP). During CPU-side setup, UCX preposts the empty receive entries once. The default queue length is zero, which disables these queues.

#### <span style="color: #037ACC;">ucp_device_put_imm</span> — Send a write with immediate

<div class="nixl-imm-signature" style="font-size: 85%;" markdown="1">

<pre><code><span style="color: #0000FF;">template</span>&lt;<span style="color: #A41615;">ucs_device_level_t</span> level = UCS_DEVICE_LEVEL_THREAD&gt;
UCS_F_DEVICE <span style="color: #A41615;">ucs_status_t</span> <span style="color: #037ACC;">ucp_device_put_imm</span>(
        <span style="color: #A41615;">const</span> ucp_device_local_mem_list_h src_mem_list_h,
        <span style="color: #A41615;">unsigned</span> src_mem_list_index, <span style="color: #A41615;">size_t</span> src_offset,
        <span style="color: #A41615;">const</span> ucp_device_remote_mem_list_h dst_mem_list_h,
        <span style="color: #A41615;">unsigned</span> dst_mem_list_index, <span style="color: #A41615;">size_t</span> dst_offset, <span style="color: #A41615;">size_t</span> length,
        <span style="color: #A41615;">uint32_t</span> immediate, <span style="color: #A41615;">unsigned</span> channel_id, <span style="color: #A41615;">uint64_t</span> flags,
        <span style="color: #A41615;">ucp_device_request_t</span> *req);</code></pre>

</div>

Posts an RDMA write from the selected source region to the selected destination region. The receiver gets the 32-bit <span style="color: #c7254e;">immediate</span> value in its CQE.

#### <span style="color: #037ACC;">ucp_device_poll_imm_at</span> — Poll a receive completion with IMM

<div class="nixl-imm-signature" style="font-size: 85%;" markdown="1">

<pre><code>UCS_F_DEVICE <span style="color: #A41615;">ucs_status_t</span> <span style="color: #037ACC;">ucp_device_poll_imm_at</span>(
        <span style="color: #A41615;">const</span> ucp_device_remote_mem_list_h mem_list_h, <span style="color: #A41615;">unsigned</span> index,
        <span style="color: #A41615;">unsigned</span> channel_id, <span style="color: #A41615;">uint64_t</span> *consumer_index_p, <span style="color: #A41615;">uint32_t</span> *immediate_p);</code></pre>

</div>

Polls the receive CQ for the peer selected by <span style="color: #c7254e;">mem_list_h</span> and <span style="color: #c7254e;">index</span> on <span style="color: #c7254e;">channel_id</span> at the calling thread’s private consumer index. On success, the call fills the <span style="color: #c7254e;">immediate_p</span> parameter, advances the cursor (<span style="color: #c7254e;">*consumer_index_p</span>) by one, and returns <span style="color: #c7254e;">UCS_OK</span>. An empty poll returns <span style="color: #c7254e;">UCS_INPROGRESS</span> and leaves the cursor in place.

#### <span style="color: #037ACC;">ucp_device_advance_rx</span> — Return receive credits

<div class="nixl-imm-signature" style="font-size: 85%;" markdown="1">

<pre><code>UCS_F_DEVICE <span style="color: #A41615;">ucs_status_t</span> <span style="color: #037ACC;">ucp_device_advance_rx</span>(
        <span style="color: #A41615;">const</span> ucp_device_remote_mem_list_h mem_list_h, <span style="color: #A41615;">unsigned</span> index,
        <span style="color: #A41615;">unsigned</span> channel_id, <span style="color: #A41615;">uint64_t</span> consumed, <span style="color: #A41615;">uint64_t</span> *consumer_index_p);</code></pre>

</div>

Once all readers are done, one thread calls this for the peer and channel. It advances the shared CQ index by <span style="color: #c7254e;">consumed</span>, returns that many receive credits, and stores the new index in <span style="color: #c7254e;">*consumer_index_p</span>.

The patch applies to UCX 1.24.0, commit 74b413ab19d63c277ffa3fb553ed83e42bc50f6f.

## Hardware ordering references

- **InfiniBand Architecture Specification**, Vol. 1, release 1.4, §9.5 “Transaction Ordering,” pp. 291–292. Rules C9-28 and C9-29 cover responder execution and completion order on one queue pair. Clause o9-20 identifies when the responder can rely on its RDMA Write buffer. [IBTA specification access](https://infinibandta.org/ibta-specification/) · [Public text of the same clause in release 1.2.1, p. 272](https://www.afs.enea.it/asantoro/V1r1_2_1.Release_12062007.pdf#page=272)

  <sub><strong>InfiniBand Architecture, Vol. 1, release 1.4, §9.5, p. 292</strong></sub><br>
  <sub>o9-20: An application shall not depend on the contents of an RDMA WRITE buffer at the responder until one of the following has occurred:</sub><br>
  <sub>• Arrival and Completion of the last RDMA WRITE request packet when used with Immediate data.</sub><br>
  <sub>• Arrival and completion of a subsequent SEND message.</sub><br>
  <sub>• Update of a memory element by a subsequent ATOMIC operation.</sub>

- **PCI Express Base Specification**, revision 5.0, version 1.0, §2.4.1, Table 2-40, pp. 177–178. Rules A2a and A2b define when a later posted write may pass an earlier one within the same traffic class. [PCI-SIG specification](https://pcisig.com/PCIExpress/Specs/Base/_5.0_1.0) · [Public PDF, p. 178](https://www.pedestrian.com.cn/_downloads/cde755483a819e060fc9169e9d71f191/PCIE_V5.0.pdf#page=178)
- **NVIDIA Grace Performance Tuning Guide**, “Optimizing IO Performance,” subsection “Relaxed Ordering.” NVIDIA documents that forcing relaxed ordering on NIC writes with `PCI_WR_ORDERING=1` exempts CQE writes. [NVIDIA NIC ordering documentation](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/optimizing-io.html#relaxed-ordering)
