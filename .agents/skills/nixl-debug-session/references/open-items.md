# NIXL Debug Session Open Items

Use these numbered references whenever the exact fact is unknown, not yet
verified for the user's installed version, or only inferred from an untrusted
report.

Web-search follow-up on 2026-05-21 found public primary-source baselines for
each bucket below. These baselines reduce the unknown surface, but they do not
replace per-issue version matching against the user's installed NIXL/framework
commit, wheel, image, or tag.

## Quick Index

- `Public Source Pins`: current baseline commits and version notes.
- `TBD-1`: NIXL API and transfer lifecycle semantics.
- `TBD-2`: framework connector behavior.
- `TBD-3`: backend or plugin configuration.
- `TBD-4`: hardware, container, Kubernetes, network, and storage requirements.
- `TBD-5`: logs, telemetry, and tracing controls.
- `Public Web Sources Checked On 2026-05-21`: source leads verified against
  primary sources.
- `Version Traps`: known version-mismatch hazards.

## Public Source Pins

Use these pins only as current-source baselines. Match them to the user's
installed package, source commit, framework version, and image before making a
case-specific diagnosis.

| Project | Public source pin |
| --- | --- |
| NIXL | `main` `b458bf0cdc1d21dd7d3130a14a09441109906569`; project version `1.2.0` in `meson.build` at that commit. |
| Dynamo | `main` `5bff35311517b0863549de00e1969810f853c6f5`; package version `1.2.0` in `pyproject.toml` at that commit. |
| vLLM | `main` `b730c4635288d75da4788bc28d8d26b5e5c3726c`; version is generated/fallback in `vllm/version.py`, so use wheel metadata or release tags for installed-version checks. |
| SGLang | `main` `ac83d8a3392ae3881b645e3b7597aee617d018c3`; version is resolved dynamically/fallback in `python/sglang/version.py`, so use wheel metadata or release tags for installed-version checks. |

## TBD-1: NIXL API And Transfer Lifecycle Semantics

Applies to transfer lifecycle and performance hypotheses.

Exact source-verified behavior for NIXL agent creation, descriptors, metadata
exchange, transfer request state, polling, notifications, return codes, errors,
and cleanup for the user's installed version.

Needed evidence:

- NIXL source, release tag, or docs matching the user's installed package or
  commit.
- Minimal reproduction showing the failing call and observed state.

Current public-source baseline:

- NIXL's public lifecycle is create agent/backend, register memory, exchange
  metadata, optionally pre-connect, create a transfer request, post it, poll
  status, release/teardown, and deregister memory. Sources:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md#L73-L90>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md#L116-L159>.
- Public docs and backend guide state that a transfer handle may be posted
  repeatedly only after completion; only one active transfer per handle is
  allowed. Sources:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/nixl.md#L116-L131>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/BackendGuide.md#L194-L224>.
- Public C++ API headers expose agent creation, backend creation, memory
  registration/deregistration, metadata export/load/invalidate, transfer
  descriptor preparation, transfer request creation, post/status/telemetry, and
  release APIs. Sources:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/api/cpp/nixl.h#L42-L131>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/api/cpp/nixl.h#L160-L333>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/api/cpp/nixl.h#L426-L562>.
- Public status codes include `NIXL_IN_PROG`, `NIXL_SUCCESS`, and multiple
  `NIXL_ERR_*` values. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/api/cpp/nixl_types.h#L53-L68>.
- Implementation at the pinned source says reposting an active transfer request
  returns `NIXL_ERR_REPOST_ACTIVE`, and status polling checks the backend while
  a request is in progress. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/core/nixl_agent.cpp#L1014-L1158>.

Still open: exact Python method signatures, status names, return-code mapping,
exception behavior, and cleanup semantics must be checked against the user's
installed version or source commit before diagnosing a specific failure.

## TBD-2: Framework Connector Behavior

Applies to Dynamo, vLLM, SGLang, and custom framework connector hypotheses.

Exact source-verified behavior for the user's installed Dynamo, vLLM, SGLang, or
custom integration version.

Needed evidence:

- Framework version, commit, image tag, or wheel metadata.
- Source/docs for that exact version.
- Startup command and relevant config/log excerpts.

Current public-source baseline:

- Dynamo `nixl_connect` is a Python API for dynamic registration and memory
  transfer between workers; the docs describe readable/writable registration,
  reads/writes from remote registered memory, and fallback when accelerated RDMA
  paths are unavailable. Sources:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/api/nixl-connect/README.md#L7-L22>,
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/api/nixl-connect/README.md#L28-L50>.
- Dynamo `nixl_connect` source creates NIXL agents, descriptors, transfer
  descriptors, transfer handles, and polls transfer state. Source:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/lib/bindings/python/src/dynamo/nixl_connect/__init__.py#L320-L615>.
- vLLM's current NixlConnector docs describe `NixlConnector` for asynchronous
  KV transfer in disaggregated prefilling, `nixl >= 1.1.0`, UCX as the default
  backend, `kv_connector_extra_config.backends` for backend selection, side
  channel host/port environment variables, and lease/TTL options. Sources:
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/docs/features/nixl_connector_usage.md#L1-L68>,
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/docs/features/nixl_connector_usage.md#L115-L145>,
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/requirements/kv_connectors.txt#L1-L3>.
- vLLM compatibility docs require matching vLLM/NIXL connector version, model
  shape, attention backend, and KV dtype. Source:
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/docs/features/nixl_connector_compatibility.md#L72-L104>.
- SGLang documents NIXL as a PD disaggregation backend, uses
  `--disaggregation-transfer-backend nixl`, and supports
  `SGLANG_DISAGGREGATION_NIXL_BACKEND` plus JSON backend params for installed
  NIXL plugins. Sources:
  <https://github.com/sgl-project/sglang/blob/ac83d8a3392ae3881b645e3b7597aee617d018c3/docs/advanced_features/pd_disaggregation.md#L218-L245>,
  <https://github.com/sgl-project/sglang/blob/ac83d8a3392ae3881b645e3b7597aee617d018c3/docs/advanced_features/pd_disaggregation.md#L326-L344>,
  <https://github.com/sgl-project/sglang/blob/ac83d8a3392ae3881b645e3b7597aee617d018c3/python/sglang/srt/disaggregation/nixl/conn.py#L216-L266>.

Still open: exact connector behavior for the user's installed framework version
can differ from `main`, `latest`, `stable`, `dev`, or archived docs. Match the
diagnosis to the installed framework version, image tag, or source commit.

## TBD-3: Backend Or Plugin Configuration

Applies to backend/plugin configuration, plugin path, plugin discovery, and
backend parameter hypotheses.

Exact backend/plugin names, plugin discovery behavior, environment variables,
library paths, and backend-specific parameters for the user's version and
deployment.

Needed evidence:

- NIXL source/docs for the installed version.
- Runtime plugin list or loader evidence gathered from the failing environment.
- Config snippets with secrets redacted.

Current public-source baseline:

- NIXL's build metadata lists plugin names at the pinned commit as `UCX`,
  `LIBFABRIC`, `POSIX`, `OBJ`, `GDS`, `GDS_MT`, `MOONCAKE`, `HF3FS`, `GUSLI`,
  `GPUNETIO`, `UCCL`, and `AZURE_BLOB`. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/meson.build#L24-L48>.
- The plugin manager honors `NIXL_PLUGIN_DIR`, searches for dynamic backend
  plugins as `libplugin_<name>.so`, validates plugin init symbols and API
  version, and combines discovered dynamic plugins with static plugins. Sources:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/core/nixl_plugin_manager.cpp#L33-L35>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/core/nixl_plugin_manager.cpp#L91-L118>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/core/nixl_plugin_manager.cpp#L239-L314>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/core/nixl_plugin_manager.cpp#L370-L419>.
- Telemetry exporters are loaded as `libtelemetry_exporter_<name>.so`. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/core/nixl_plugin_manager.cpp#L356-L419>.
- Config lookup is `NIXL_CONFIG_FILE`, then `$HOME/.nixl.cfg`, then
  `/etc/nixl.cfg`. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/utils/common/configuration.cpp#L25-L75>.
- Build-time plugin filters include `static_plugins`, `enable_plugins`, and
  `disable_plugins`. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/README.md#L154-L162>.
- Dynamo KVBM env pattern is `DYN_KVBM_NIXL_BACKEND_<backend>=<bool>`;
  defaults differ by layer, with KVBM config defaulting to `UCX` and `POSIX`.
  Sources:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/lib/memory/src/nixl/config.rs#L57-L97>,
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/lib/kvbm-config/src/nixl.rs#L54-L67>.

Still open: actual plugin availability, backend parameters, loadability, and
native dependency status remain environment-specific. Use static package and
path evidence first, then dynamic plugin probes only after the paths are trusted.

## TBD-4: Hardware, Container, Kubernetes, Network, And Storage Requirements

Applies to hardware/runtime, container/Kubernetes, multi-node network, and
storage-path hypotheses.

Exact deployment requirements for the failing hardware, container image,
Kubernetes runtime, multi-node fabric, GPU access, storage path, or cloud/on-prem
environment.

Needed evidence:

- Image tag or container build source.
- Pod or job spec excerpts, device visibility, driver/runtime evidence, and
  source-backed runtime requirements.
- Node placement, network interface, or storage path evidence when relevant.

Current public-source baseline:

- NIXL is Linux-only, tested on Ubuntu 22.04/24.04 and Fedora, and its public
  README says `pip install nixl` installs CUDA 12/13 backends and selects by
  PyTorch CUDA. UCX is tested with `1.21.x`; GDRCopy is needed for maximum
  performance, but UCX and NIXL can work without it. Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/README.md#L31-L65>.
- Dynamo NIXL Connect docs say GPUDirect RDMA requires a capable NIC/GPU,
  drivers supporting GPU-NIC zero-copy/RDMA, and an InfiniBand or RoCE network.
  Source:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/api/nixl-connect/README.md#L13-L22>.
- Dynamo Kubernetes disaggregated communication docs say NVLink cannot be used
  between pods, production disaggregation requires RDMA, and NIXL uses UCX or
  libfabric for KV transfer. Sources:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/kubernetes/disagg-communication-guide.md#L12-L18>,
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/kubernetes/disagg-communication-guide.md#L39-L63>.
- Dynamo Kubernetes RDMA evidence includes `IPC_LOCK`, `rdma/ib`, RDMA device
  plugin, and GPUDirect checks. Source:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/kubernetes/disagg-communication-guide.md#L411-L459>.
- Dynamo docs recommend libfabric for AWS EFA, while vLLM docs describe UCX as
  default for NixlConnector. Treat this as deployment-specific, not a universal
  backend default. Sources:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/kubernetes/disagg-communication-guide.md#L296-L318>,
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/docs/features/nixl_connector_usage.md#L28-L45>.

Still open: exact hardware, image, Kubernetes, device-plugin, fabric, storage,
and cloud/on-prem requirements for a user's failure depend on the deployment
shape and installed image. Do not generalize EFA guidance to non-EFA fabrics.

## TBD-5: Logs, Telemetry, And Tracing Controls

Applies to logging, telemetry, tracing, counters, performance-observation, and
debug-verbosity hypotheses.

Exact NIXL, framework, backend, or deployment logging locations and verbosity
controls for the user's installed version.

Needed evidence:

- Source/docs for logging or tracing controls.
- Runtime command line, environment, and observed log paths.
- Relevant log excerpts with sensitive data redacted.

Current public-source baseline:

- NIXL log level is controlled by `NIXL_LOG_LEVEL`; accepted values are
  `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, and `FATAL`, with default `WARN`.
  Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/utils/common/nixl_log.cpp#L35-L77>.
- NIXL uses Abseil logging macros; debug/trace map to `VLOG(1)` / `DVLOG(2)`.
  Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/utils/common/nixl_log.h#L32-L95>.
- NIXL telemetry docs describe runtime telemetry with one active telemetry
  plugin per NIXL instance and public variables including
  `NIXL_TELEMETRY_ENABLE`, `NIXL_TELEMETRY_BUFFER_SIZE`,
  `NIXL_TELEMETRY_RUN_INTERVAL`, `NIXL_TELEMETRY_EXPORTER`, and
  `NIXL_TELEMETRY_DIR`. Sources:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/telemetry.md#L5-L16>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/docs/telemetry.md#L71-L99>,
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/core/telemetry/telemetry.cpp#L72-L167>.
- Prometheus exporter variables are documented in the telemetry plugin README.
  Source:
  <https://github.com/ai-dynamo/nixl/blob/b458bf0cdc1d21dd7d3130a14a09441109906569/src/plugins/telemetry/prometheus/README.md#L36-L66>.
- Dynamo UCX diagnostics use `UCX_LOG_LEVEL`, `UCX_LOG_FILE`, and UCX stats
  when UCX is built with stats. Source:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/kubernetes/disagg-communication-guide.md#L190-L200>.
- vLLM NIXL worker source includes structured transfer-failure context. Source:
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py#L648-L696>.

Still open: exact framework log locations, telemetry output files, and observed
log paths/excerpts remain runtime-specific. NIXL `v1.1.0` release notes changed
Prometheus telemetry metric shape and removed `NIXL_TELEMETRY_BACKEND`, while
current `main` telemetry docs still mention older fields; pin telemetry claims
to the user's installed NIXL version.

## Public Web Sources Checked On 2026-05-21

- DeepWiki leads, verified against primary sources before use:
  <https://deepwiki.com/ai-dynamo/nixl/1.2-core-concepts>,
  <https://deepwiki.com/ai-dynamo/nixl/2-api-reference>,
  <https://deepwiki.com/ai-dynamo/nixl/1-getting-started>,
  <https://deepwiki.com/ai-dynamo/dynamo/5-llm-framework-integration>.
- NIXL source/docs at pinned `main`
  `b458bf0cdc1d21dd7d3130a14a09441109906569`.
- Dynamo source/docs at pinned `main`
  `5bff35311517b0863549de00e1969810f853c6f5`.
- vLLM source/docs at pinned `main`
  `b730c4635288d75da4788bc28d8d26b5e5c3726c`.
- SGLang source/docs at pinned `main`
  `ac83d8a3392ae3881b645e3b7597aee617d018c3`.

## Version Traps

- vLLM current `main` requires `nixl >= 1.1.0`; vLLM `v0.11.0` required
  `nixl >= 0.5.1`. Current vLLM docs are not evidence for older deployments
  until the installed vLLM requirements are checked.
- Dynamo `v1.1.1` release notes pin SGLang, vLLM, and NIXL versions that can
  differ from upstream `main` docs. Match Dynamo runtime images to their pinned
  framework and NIXL versions.
- Dynamo docs recommend libfabric for AWS EFA, while vLLM's public NIXL docs
  describe UCX as the default. Treat backend selection as deployment-specific.
