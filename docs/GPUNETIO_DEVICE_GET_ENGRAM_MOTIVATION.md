# Application motivation: GPU-initiated reads for shared Engram tables

This note records the application context for reviewing a generic NIXL GPU GET contract. It does not claim a GET-kernel speedup or make Engram behavior part of the NIXL API.

## Observed application

SGLang can use a shared, immutable, approximately 202 GB registered Host table for Engram lookups. The consumer computes a GPU hash/index and issues data-dependent reads into that remote table. The measured cross-node case had one table owner and one TP8 consumer group. The P and D processes ran on different nodes; D remained the same instance across the Native and Shared P arms. P-to-D KV transfer used the serving stack's normal transport; it is distinct from Engram's table GET.

The paired test used C8, three rounds of 32 measured requests per arm, 16K-token prompts and 64 output tokens. D used DP8, had decode graphs enabled, SWA ratio 0.05, and configured context length 1M; each test prompt was 16K. Both P arms used HiCache32 per rank. The Native arm read its local table; the Shared arm used the remote registered table and GPU-initiated GET path. The C1 functional smokes were excluded from timing.

Across 192 measured responses, all outputs completed at 64 tokens and all 96 paired output text hashes matched. Median-of-three-round-quantiles results were:

| Metric | Native local table | Shared remote table | Shared change |
|---|---:|---:|---:|
| TTFT p50 | 1638.417 ms | 1653.214 ms | +0.903% |
| TPOT p50 | 2.116503 ms | 2.116785 ms | +0.0133% |
| Output throughput | 279.958 tokens/s | 278.693 tokens/s | −0.452% |

There were zero cache hits, request errors, or retractions. This is application-level evidence that the GPU GET path participates in a real cross-node workload. It is not a GET-vs-host-READ microbenchmark: the Native reference uses a local table, so these numbers do not establish that the GET kernel is faster than a host-issued remote read. The result instead reports the complete serving tradeoff for this one owner/one TP8 consumer setup.

## Why review a generic GET contract

The workload needs a consumer to read from an already registered remote source view into a local destination view after the GPU computes the lookup address. That makes submission acceptance, completion, destination visibility, terminal errors, and view/MR lifetime part of the public design. The initial reference is bounded to one caller and one in-flight request; it does not settle concurrency, timeout recovery, arbitrary memory types, or production SDK support.

Current upstream main has no public GPU transfer-GET entry point or GPUNETIO compatibility implementation. The checked-in legacy reference is specific to DOCA 3.1 `DUMP_AUTO` and private view/QP internals; porting it to the supported SDK is future work. The public prototype and owner/consumer reproducer are therefore design artifacts, not runnable current-main code.

The dependency audit also inspects the separate #2234 proposal. Its native view-builder path is not present in upstream main; that proposed path currently admits VRAM views only. The measured Engram owner is pinned Host/`DRAM_SEG`, so the public GET port must explicitly decide and validate supported memory types rather than inherit a #2234 limitation by assumption. This Draft's base remains upstream main.

In the preserved dependency-audit document, items 2–5 and the `src/plugins/gpunetio/gpunetio_device_memview.cpp` check refer to #2234's head, not this branch's `HEAD`. To inspect that public head without changing this worktree, fetch it as a read-only remote-tracking ref and use `git show upstream/pr-2234:<path>`; do not rebase this branch onto it. The `git rev-parse upstream/pr-2234` command in the original audit assumes that ref has already been fetched.

The preserved dependency-audit document names some files introduced by #2234. On this upstream-main branch, verify the source tree with:

```bash
git ls-tree -r --name-only HEAD | rg 'src/api/device/gpu/impl/gpunetio|gpunetio_device_memview.cpp'
git grep -n -E '(^|::)get[[:space:]]*\(' HEAD -- src/api/device/gpu/nixl_device.cuh src/api/device/gpu/impl
```

The first command has no matching GPUNETIO native-view adapter paths in upstream main; the second finds no transfer-GET call. The VRAM-only view check is specific to #2234's proposed tree, not this branch base.

The related [#2234 PUT draft](https://github.com/ai-dynamo/nixl/pull/2234) remains unchanged and covers PUT/completion. It is linked for context only; its PUT measurements are not attributed to GET.
