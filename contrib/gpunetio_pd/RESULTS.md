# Native compact PD component: no-LAG comparison

Historical compact-only evidence, not a measurement of the refreshed integrated
Draft. See [integrated results and boundaries](INTEGRATED_RESULTS.md).

Measured backend source: `972c6364a65e8a7a2f98fbca9dc5d79d94a35bc3`, based on
per-QP progress revision `01b928041563855c9ed95f283048721fdfa240ce`.
Companion documentation/runner additions do not change that C++/CUDA source.

Two H20 hosts, DOCA3.5.0098 and CUDA12.8; one common freshly built NIXL core,
the same NIC-local sender placement and fixed worker CPU in both arms.
27 layers x64 rows x576 bytes x4 peers, changing windows/payloads and fresh
request handles. UCX SGL enabled; neither caller packs payloads. UCX path/rail
selection is unrestricted. No LAG hook, interposer, or multi-QP striping.

Five alternating fresh pairs, 20 warmup and600 measured requests per arm/pair:

| Primary request metric | UCX no-pack+SGL | Compact GPUNETIO |
|---|---:|---:|
| Pooled p50 | 284.301 us | 265.155 us |
| Pooled p99, linearly interpolated | 297.916 us | 279.522 us |
| Maximum | 400.804 us | 357.671 us |
| Samples >=1ms | 0/3000 | 0/3000 |

**Median paired p50 reduction6.68%; all five p50 and p99 pairs improve.**
Paired p50 reductions:6.61%,6.73%,6.58%,6.68%,6.94%.
All20 workers exit0 with GPU payload/hole/marker checks after each dispatch.
This covers24,800 four-peer window validations including warmup.

The primary timer includes dynamic address planning, request preparation,
posting, and completion of all four requests including notifications. Source
payload production precedes timing; receiver validation/control ACK follows it.
GPUNETIO preparation p50 is18.675us versus106.896us, while its post-to-DONE
p50 remains slower:227.226us versus128.212us. This is not faster raw NIC
transfer, serving TTFT/TPOT, overlap with GEMM, or peer-MR evidence.

The validation ACK must not inherit the primary latency claim: GPUNETIO's
full request-plus-ACK maximum is22.735ms versus UCX4.442ms; both arms have
request-plus-ACK samples above1ms. The pooled request-plus-ACK p99 values are
1578us and1645us respectively.

Functional qualification additionally covers ordinary WRITE, odd row counts,
distinct layer MRs, six invalid-input rejection cases, five reposts, and two
interleaved standalone notifications. Retained ordinary READ controls pass on
both baseline and candidate for dense/fragmented addresses, key boundaries,
513/1024/4096/16384 descriptors and35 repost epochs. READ payload validation
is functional, not a GPU-side acquire test. Fault injection is not qualified.
Baseline and candidate both emit SDK initialization/capability diagnostics
(`PRM command=0xb0c`, firmware syndrome `0x6c4d48`). No transfer CQE errors
were found in the formal logs, but passing payload checks are not a claim
that SDK initialization logs contain no errors.

The companion runner's summary uses nearest-rank p99; this table uses linear
interpolation of the retained raw samples, so tiny p99 rounding differences
are expected. Raw samples are retained for either calculation. Reproduction
commands and measurement boundaries are in [README.md](README.md).

## Public-command qualification

The exact companion build/run commands were subsequently exercised with a
freshly compiled replay, not only the retained private replay binary.
Another five alternating600-sample pairs pass the runner's acceptance gate:
median paired request p50 reduction6.78%, all five p50 wins. Pooled UCX/GP
request p50 is286.427/265.301us and nearest-rank p99 is384.696/278.599us,
with0/3000 primary samples>=1ms in either arm. The higher UCX p99 in this
separate campaign is retained, not substituted into the first table.
Validation-ACK maxima are19.058/14.206ms, so that interval still has tails.
Both campaigns use the same backend/plugin source and common core.
