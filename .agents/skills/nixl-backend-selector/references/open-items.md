# NIXL Backend Selector Open Items

Use these numbered references whenever a backend recommendation depends on a
fact that is unknown, user-environment-specific, or only checked against mutable
public `main`/`latest` sources.

## Quick Index

- `Public Source Pins`: current-source baseline commits for NIXL, Dynamo, vLLM,
  and SGLang.
- `Framework Baseline Sources`: pinned framework docs and source anchors.
- `TBD-0`: source-baseline integrity checks.
- `TBD-1`: installed NIXL backend inventory.
- `TBD-2`: framework backend selector surface.
- `TBD-3`: backend parameters and fallback semantics.
- `TBD-4`: hardware, fabric, storage, and runtime requirements.

## Public Source Pins

These pins are current-source baselines, not guarantees for the user's
installed environment.

| Project | Public source pin |
| --- | --- |
| NIXL | `main` `b458bf0cdc1d21dd7d3130a14a09441109906569`; project version `1.2.0` and plugin list in `meson.build` at that commit. |
| Dynamo | `main` `5bff35311517b0863549de00e1969810f853c6f5`; package version `1.2.0` in `pyproject.toml` at that commit. |
| vLLM | `main` `b730c4635288d75da4788bc28d8d26b5e5c3726c`; use wheel metadata or release tags for installed-version checks. |
| SGLang | `main` `ac83d8a3392ae3881b645e3b7597aee617d018c3`; use wheel metadata or release tags for installed-version checks. |

Detailed current-source anchors for backend/plugin facts live in
`references/verified-baseline.md`.

## Framework Baseline Sources

- vLLM NIXL connector usage, pinned source:
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/docs/features/nixl_connector_usage.md>.
- vLLM NIXL connector compatibility, pinned source:
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/docs/features/nixl_connector_compatibility.md>.
- Dynamo Kubernetes disaggregated communication, pinned source:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/kubernetes/disagg-communication-guide.md>.
- SGLang PD disaggregation docs, pinned source:
  <https://github.com/sgl-project/sglang/blob/ac83d8a3392ae3881b645e3b7597aee617d018c3/docs/advanced_features/pd_disaggregation.md>.
- SGLang NIXL disaggregation source, pinned source:
  <https://github.com/sgl-project/sglang/blob/ac83d8a3392ae3881b645e3b7597aee617d018c3/python/sglang/srt/disaggregation/nixl/conn.py>.

## TBD-0: Source Baseline Integrity

Before treating baseline facts as evidence, confirm the source URLs, commit IDs,
or installed package metadata still match the user's target version. If any
source cannot be rechecked, keep the claim as a numbered `TBD` instead of
promoting it to a recommendation.

## TBD-1: Installed NIXL Backend Inventory

Exact plugin names, plugin loadability, backend creation behavior, memory
support, and plugin parameters for the user's installed NIXL package, wheel,
image, build, or source commit.

Needed evidence:

- NIXL version, wheel metadata, image tag, or source commit.
- `get_plugin_list()`, `get_plugin_params()`, or static plugin/package evidence
  from the same trusted runtime where the workload runs.
- Backend creation result and error text if creation fails.
- Plugin transport capability within a loaded plugin, such as UCX transports,
  libfabric provider, GDS mode, or OBJ engine variant compiled into the user's
  build.

## TBD-2: Framework Backend Selector Surface

Exact backend selector names, config fields, defaults, and accepted values for
the user's installed Dynamo, vLLM, SGLang, or custom integration version.

Needed evidence:

- Framework version, image tag, source commit, or docs link matching the
  workload.
- Startup command and redacted config snippet showing the selected connector or
  backend.
- Logs proving which backend the framework actually created.

## TBD-3: Backend Parameters And Fallback Semantics

Backend-specific options, environment variables, default values, fallback
behavior, and preference ordering for the user's version.

Needed evidence:

- Source or docs for the backend plugin at the installed NIXL version.
- Plugin parameter output or framework source showing how parameters are passed.
- Logs showing fallback, selected provider, or backend creation failure.

## TBD-4: Hardware, Fabric, Container, And Storage Requirements

Whether the user's hardware, network, container, Kubernetes pod, storage mount,
or cloud instance type supports the candidate backend.

Needed evidence:

- Fabric/storage description such as InfiniBand, RoCE, AWS EFA, local NVMe, or
  S3-compatible object storage.
- Device visibility, driver/runtime, RDMA device, GPUDirect, and container or
  pod spec evidence when relevant.
- Source docs for the image or deployment stack.

## TBD-5: Performance And Correctness Validation

Whether the selected backend is actually used and performs correctly for the
target workload.

Needed evidence:

- NIXL/framework logs naming the backend.
- A minimal transfer, benchmark, or workload run using the intended memory and
  storage path.
- Before/after evidence when replacing a backend or disabling fallback.
