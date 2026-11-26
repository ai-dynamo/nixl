# LIBFABRIC on AWS EFA

## Investigate if needed

Run checks in the same host, container, or pod as the workload and on every
peer.

1. Inside every workload, confirm that an EFA device and its bound `efa` driver
   are visible, using workload-visible `/sys/class/infiniband`,
   `/dev/infiniband`, or equivalent release-matched diagnostics. Host visibility
   is insufficient. If EFA is absent or hidden, stop and follow
   [EFA is unavailable or not exposed to the workload](#efa-is-unavailable-or-not-exposed-to-the-workload) before
   investigating libfabric.
2. Identify the libfabric actually loaded by the workload from process
   mappings or runtime-loader output. Treat plugin dependencies as supporting
   evidence only. If that installation provides `fi_info`, inspect it with:

   ```bash
   fi_info --version
   fi_info -e
   fi_info -p efa
   ```

   Missing `fi_info` proves only that the tool is unavailable; do not install
   another libfabric merely to probe. Leave provider advertisement unverified
   until it can be checked. If libfabric is genuinely absent, follow
   [The EFA provider is missing or a different libfabric is loaded](#the-efa-provider-is-missing-or-a-different-libfabric-is-loaded)
   and the canonical AWS EFA setup.

   `fi_info -p efa` proves only that the queried libfabric advertises EFA. It
   does not prove that NIXL loads the same library or uses EFA. When the tool
   runs, compare its library dependencies with the loaded installation above.
3. Confirm that the deployed NIXL artifact can create LIBFABRIC. Use the
   matching NIXL backend, plugin, and build documentation linked below to
   inspect availability and dependencies.
4. Determine which provider the created LIBFABRIC backend initializes.
   Libfabric provider selection and NIXL backend selection are separate.
5. Inspect the integration's intended and effective backend on every peer.
6. When registration needs testing, register a representative buffer of the
   required memory type through LIBFABRIC alone. For Python, look up
   `nixl_agent.register_memory` in the deployed release’s
   [Python API](../../../../src/api/python/_api.py) and use its documented
   backend-selection option. Determine the buffer requirements from the
   installed NIXL, libfabric, and runtime documentation.
7. If the effective path is still unclear, first look for evidence from an
   existing representative cross-node request. If a new request is needed,
   use an authorized target, a bounded payload, and a timeout. Correlate EFA
   activity with the same request.

When provider or backend initialization remains unclear, use the diagnostic
controls documented by the installed NIXL and libfabric releases.

Relevant NIXL sources in this repository are the
[LIBFABRIC backend documentation](../../../../fern/docs/pages/user-guide/backends/libfabric.md),
[plugin guide](../../../../src/plugins/libfabric/README.md),
[build documentation](../../../../fern/docs/pages/user-guide/building-nixl/index.md),
and [environment-variable reference](../../../../fern/docs/pages/resources/environment-variables.md).

## Fix

1. ### EFA is unavailable or not exposed to the workload

   On an EFA-supported
   target, use the interface attachment or allocation steps in the AWS
   [NIXL on EFA guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nixl.html).
   Change a security-group or network rule only when connectivity evidence
   identifies the blocking rule and the change is authorized. Then use the
   supported AWS host,
   [container](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-installer-containers.html),
   or [EKS](https://docs.aws.amazon.com/eks/latest/userguide/device-management-efa.html)
   setup to expose EFA to every peer.
2. ### The EFA provider is missing or a different libfabric is loaded

   Use one
   compatible EFA/libfabric userspace stack. Attribute the wrong loaded
   libfabric to the integrating software when its image, launcher, or workload
   configuration controls selection; attribute it to the host environment when
   the host loader controls selection. If ownership is not proven, attribute
   it to unknown. Correct only the controlling layer so NIXL, its plugin, and
   diagnostic tools resolve the intended stack instead of mixed installations.
   Follow the canonical
   [AWS EFA setup](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html)
   rather than copying package instructions from this file.
3. **NIXL cannot create LIBFABRIC.** Use a compatible NIXL artifact with
   LIBFABRIC support and satisfy its runtime dependencies. Follow the matching
   NIXL backend, plugin, and build documentation linked above rather than
   inferring build or loading behavior here.
4. **LIBFABRIC initializes the wrong provider.** Follow the installed
   libfabric documentation to diagnose and configure provider selection.
   Consult the installed `fi_efa` manual; use the
   [upstream EFA provider page](https://ofiwg.github.io/libfabric/main/man/fi_efa.7.html)
   only as a fallback because it describes rolling source.
5. **The integration selects another NIXL backend.** Select LIBFABRIC through
   the installed integration's supported interface on every peer. If that
   release cannot select it, treat this as an integration limitation instead
   of patching integration source as an installation fix.
6. **Memory registration fails.** Identify the exact memory type and runtime,
   then follow the matching installed NIXL, libfabric-provider, and runtime
   documentation. Do not infer requirements from another memory type or
   runtime.
7. **An isolated test passes but the deployment does not.** Reproduce with the
   real peers, memory type, library paths, and integration configuration, then
   correct the differing layer.

## Verify if possible

Verify only relevant or changed layers on every peer. Leave unavailable evidence
unverified; do not install or build solely to verify.

1. Every workload sees its assigned EFA device with the `efa` driver bound.
2. When matching `fi_info` is available, the same libfabric loaded by the
   workload advertises `efa`; otherwise leave this unverified.
3. The deployed NIXL artifact creates LIBFABRIC without load or dependency
   errors.
4. Workload-process evidence shows the created backend initializing the `efa`
   provider.
5. Confirm that LIBFABRIC registered the buffer. If the evidence does not
   identify which backend succeeded, leave LIBFABRIC registration unverified.
6. The real tested request uses LIBFABRIC on every peer.
7. Use an existing representative cross-node request when available. Run a new
   transfer through the real integration only when authorized, against approved
   peers, with a bounded payload and timeout. It passes only if the transfer
   completes with the expected data; otherwise leave this layer unverified.
8. EFA activity correlated with that run is supporting evidence, not a
   substitute for backend and provider evidence.

Use the current [AWS EFA metrics guidance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-working-monitor.html)
for available counters. If unrelated traffic cannot be excluded, treat the
result as consistent with EFA rather than verified.
