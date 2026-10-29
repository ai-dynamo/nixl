<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Yunxiao Ning
SPDX-License-Identifier: Apache-2.0
-->

# SGLang shared Engram: four-node 3P1D application evidence

This is application evidence for the GPU-initiated GET design discussion. It
adds no backend, public API, SDK port, or model integration to this Draft.
The measurements use the same legacy research implementation as the preceding
1P1D experiment; scale-out adds deployment and validation orchestration.

## Workload and data path

Three independent TP8 prefill groups and one TP8/DP8 decoder ran on four
physical nodes, using 32 H20 GPUs. Each P computed lookup IDs on the GPU and
read its needed rows from one immutable pinned Host table bank on D's node.
The bank held 202,758,032,400 bytes in one backing with two registered table
views. All 24 TP readers attached to the same generation and had zero full-table
parameter elements. D retained its native local-table lookup.

P/D prompt KV transfer remained Mooncake in both arms. This is a local-Engram
versus remote-Engram comparison, not a comparison of KV transfer backends or
GPU GET versus a host-issued remote READ.

- Native: each P uses its local table. Shared: the three P groups use one bank.
- P: CP interleave, chunk 16,384, SWA ratio 0.05, memory fraction 0.82,
  max running 128, HiCache parameter 32 per rank.
- D: DP8 attention, memory fraction 0.85, max running 256, decode graph max32.
- Both arms: DSPARK speculative decoding, block size 5; D graphs enabled.
- Actual inputs: 16,384 tokens; outputs: 64 tokens. Configured context 1M is not
  evidence that 1M-token requests were tested.
- Generated document/number-sequence workload, not production traffic. The fixed
  384-record source has SHA256
  `609bb3e1a1c64f2de4536cb465712f3a91f7727912663fa50fe8b71e010c1434`.
- Each arm: client concurrency 24, three rounds of 96 measured requests, plus 24
  disjoint warm prompts per round. Measured indices are 0-95, 96-191, 192-287;
  the same 24 warm records, 360-383, repeat each round after idle cache flushes
  on all P/D groups. Thus 288 distinct measured inputs and 72 warm observations
  per arm are not 360 distinct inputs.

The client maintains up to 24 requests in flight. TTFT starts when its worker
begins sending a request; it excludes waiting inside the client thread pool.
This is closed-loop batch replay, not a production open-loop arrival process.

## Results and controls

Values are medians of the three per-round quantiles/rates, not pooled quantiles.
The six author-reported run summaries are in
[gpunetio_shared_engram_3p1d_rounds.csv](gpunetio_shared_engram_3p1d_rounds.csv).

| Metric | Native local tables | Shared remote table | Shared change |
|---|---:|---:|---:|
| TTFT p50 | 1632.793 ms | 1647.930 ms | +0.93% |
| TTFT p90 | 1695.663 ms | 1685.860 ms | -0.58% |
| TTFT p99 | 1978.608 ms | 2031.544 ms | +2.68% |
| TPOT p50 | 2.213 ms | 2.203 ms | -0.48% |
| TPOT p90 | 2.337 ms | 2.470 ms | +5.69% |
| TPOT p99 | 3.084 ms | 3.025 ms | -1.91% |
| Output throughput | 834.304 tokens/s | 828.517 tokens/s | -0.69% |

All 576 measured outputs completed correctly; all 288 paired input and output
hashes matched. Including warmups, 720 requests completed. Correctness here is
number-sequence semantics, output agreement, and the separately checked lookup
bytes/scales/BF16 path; it is not a broad model-quality evaluation.

Every 120-request window, including warmups, recorded 40 successful requests and
KV handoffs per P. D recorded 120 successes, 120 KV allocation/bootstrap handoffs,
7,680 generated tokens, and zero prefill-effective tokens. Request errors,
cache hits, retractions and recorded IB error/retry deltas were zero.

P abort-submission counters were not zero: Native 24, Shared 26 in total; D 0.
All client outputs completed. The counters record abort submissions, not failed
outputs. Unconsumed P responses and connection warnings were observed in the
debug MiniLB, but no per-request correlation establishes the exact abort cause.
Do not call the counter zero, prove it harmless, or attribute it to GPUNETIO.

The same D, gateway, source generation and MR backing persisted across arms.
All 48 weight shards plus config/index matched by SHA256 on all four nodes.
The full-checkpoint manifest SHA256 was
`b4f27caf78594703b44a205d5539bd745a260b50234dc30dc649700a013efddd`.

## Interpretation and limitations

The result supports sharing P-side table ownership in this real-model workload,
with an observed median TTFT difference of +15.14 ms and throughput difference
of -0.69%. It is not a speedup claim or production SLO qualification:

1. Native's three rounds preceded Shared's three rounds, with one loaded model
   instance per arm. There was no Native-post bracket or randomized alternation;
   the approximately 1% differences are not isolated causal estimates of GET cost.
2. TTFT p99 +2.68% and TPOT p90 +5.69% are retained. Small per-round samples do not
   establish stable long-run tail behavior.
3. P abort-submission reasons remain unresolved.
4. Earlier receiver-NIC mapping controls support the selected configuration;
   they do not uniquely divide the benefit into topology distance, traffic
   distribution and NIC-internal processing.

Three P table requirements become one shared bank: 66.7% less logical P-table
payload. Including D's unchanged local table gives four logical copies versus
two, or 50% less table payload. Neither figure is a measured whole-process RSS
reduction in this experiment. The inactive shared bank stayed allocated during
Native to preserve the environment; do not count it as a new Native deployment
requirement. Earlier 2P2D physical-memory measurements are a different experiment.

Reader workspaces/control state still cost memory. With the same memory fraction,
Shared P token capacities were about 0.65% lower; they were not equalized by
changing scheduler settings. One P retained the same small pre-existing control
context in both arms. The gateway was a debug MiniLB. HiCache32 was used in
both arms after the deployment's130-per-rank setting exceeded the tested Host
budget. These are explicit deviations from a production deployment.

## Reproduce the published arithmetic

From this PR checkout, run the standard-library-only command below. It validates
the six row summaries and recomputes the displayed medians and changes. It does
not run GPU work, validate the underlying payloads independently, or reproduce
application performance.

```bash
python3 - <<'PY'
import csv
import statistics
from pathlib import Path

path = Path("docs/gpunetio_shared_engram_3p1d_rounds.csv")
with path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 6
arms = {}
for arm in ("native", "shared"):
    selected = [r for r in rows if r["arm"] == arm]
    assert sorted(int(r["round"]) for r in selected) == [0, 1, 2]
    for r in selected:
        assert int(r["measured_requests"]) == 96
        assert int(r["warmup_requests"]) == 24
        assert all(float(r[k]) == 40 for k in ("p0_success", "p1_success", "p2_success"))
        assert float(r["d_success"]) == 120
        assert all(float(r[k]) == 0 for k in (
            "request_errors", "d_abort_submissions", "cache_hit_measured",
            "cache_hit_warmup", "retractions_measured", "retractions_warmup"))
    expected_aborts = 24 if arm == "native" else 26
    assert sum(float(r["p_abort_submissions"]) for r in selected) == expected_aborts
    arms[arm] = selected
metrics = [f"{m}_{q}_ms" for m in ("ttft", "tpot") for q in ("p50", "p90", "p99")]
for metric in metrics + ["tokens_per_s"]:
    native, shared = (statistics.median(float(r[metric]) for r in arms[a])
                      for a in ("native", "shared"))
    print(f"{metric}: {native:.3f} -> {shared:.3f} ({100*(shared/native-1):+.2f}%)")
PY
```

## Full GPU rerun prerequisites

A full rerun needs the matching legacy research backend/shim, SGLang model
adapter and owner, complete checkpoint, frozen inputs, and four suitable nodes.
These are not all distributed by this interface Draft. There is no standalone
current-main GPU reproduction command here; the checked-in GET reference does
not supply a supported SDK implementation.

The measured adapter checkpoint was `c2ba4222787820e5d46069a163b49ded95a94aee`,
SGLang snapshot `9a9ae36865e6c7c24fcbf3ce44a14b5e2945709a`, and shim SHA256
`9c158ae71ce75ca68cd45e3d4a02890d59f44cfa52b31f7e4dcb5954c737d46a`, using
legacy DOCA 3.1. No query algorithm or runtime binary changed for 3P1D: the added
work was two P deployments, multi-P routing, and expanded measurement/identity
checks. Large lookups retained 16 QP, FIFO depth 2, per-lookup deduplication,
deferred layer 14 and the same per-GPU receiver-NIC selection.

For a matched rerun, preserve the configuration above, capture fresh process and
role identities, verify the same bank with 24 leases and no reader table replicas,
then measure Native and Shared with the fixed indices/warmups and per-role
handoff counters. Stop admission and retire readers before releasing leases or
backing. Exact current-SDK porting, automatic crash recovery, production traffic,
and independent-run tail qualification remain future work.
