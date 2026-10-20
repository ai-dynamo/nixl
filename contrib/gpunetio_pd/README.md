# GPUNETIO PD component replay

Manual companion for the integrated PD Draft on #2222's independent progress.
It exercises compact affine WRITE, independent progress owners and opt-in
ordered data/notification submission without payload staging. Backend defaults
remain one progress owner and separate data/notification QPs. The replay example
opts into four owners and ordered notifications.

[Integrated research results](INTEGRATED_RESULTS.md) are commit-pinned historical
evidence, not a claim for this refreshed head. The balanced-LAG research adapter
is NOT included. This runner measures the native path; reproducing the complete
historical setup requires a qualified supported LAG path, a Draft exit criterion.

Build and install this revision with GPUNETIO and UCX enabled. Use a supported
CUDA/DOCA toolchain and a UCX build exposing the experimental SGL API; NIXL must
report `HAVE_UCX_SGL_API`. Build the replay with:

```sh
bash contrib/gpunetio_pd/build.sh "$NIXL_INCLUDE" "$NIXL_LIB" "$CUDA_ROOT"
```

`PD_BENCH_ARCH` defaults to `sm_90`. The build runs the CPU payload-identity
check; it does not depend on a backend-private span-test header. The CPU check
can also run independently:

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Werror contrib/gpunetio_pd/payload_test.cpp -o /tmp/pd-payload-test
/tmp/pd-payload-test
```

Copy the replay and one identical NIXL core/plugin runtime to both endpoints.
Fill in every endpoint, fabric, path, GPU/CPU, and port field in a private copy
of `config.example.json`. `shell_prefix` receives one shell-quoted command.
Keep the remote shell environment clean. The configuration is trusted command
data; raw logs, hashes, and mapped-library records are private evidence.

```sh
python3 contrib/gpunetio_pd/run.py "$PRIVATE_CONFIG" "$NEW_RESULT_DIR"
```

The runner alternates five fresh UCX/GPUNETIO process pairs by default, with
20 warmup and 600 measured requests per arm. It enables UCX SGL and fixes
`PD_NOTIFICATION_PIPELINE=0` for both arms. It rejects `LD_PRELOAD`,
`PD_LAG_POLICY`, alternate layer MRs, and UCX environment controls other than
matching device/GID selection and the module directory. No LAG adapter is
included. Keep UCX transport and rail-selection defaults.

Only the selected GPU is checked: free memory must meet `min_free_mib`
(default 4096; a private deployment may set 1500) and utilization must be at
most 5%. Workers use the configured fixed CPU and a 90-second timeout.

`progress_owners` is an integer in [1, 4] and `ordered_notifications` is a
boolean in the replay JSON. They become immutable GPUNETIO backend parameters
at engine creation, not production environment toggles. Ordered notifications
currently require `data_qp_count=1`; existing multi-QP WRITE remains available
with ordered notifications disabled. A peer's striped QPs share one progress
owner because they share a notification CQ.

The fixed workload has 27 layers, 64 rows per layer/peer, 576 bytes per row,
four peers, and three rotating windows. GPUNETIO uses compact strided runs;
UCX uses physical row descriptors. Both transfer identical bytes and retain
notification completion. Payload generation is synchronized before timing.
The receiver acquires the marker and checks payload bytes, prior windows, and
untouched regions after sender completion.

`SAMPLE` columns are planning, preparation, posting, post-to-DONE, total request,
and receiver-validation ACK time, in microseconds. Request time is the primary
metric; ACK time is reported separately. The runner retains all samples and
reports pair and pooled statistics plus its finite-sample gate. This companion
makes no serving or TTFT claim. Request-plus-validation ACK times must not
inherit the sender request latency claim.
