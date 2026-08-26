# NIXL Backend Selector Verified Baseline

These facts were checked on 2026-05-21. They are source baselines, not
deployment guarantees. Match all claims to the user's installed NIXL/framework
version before giving a high-confidence recommendation.

## NIXL Source Pin

- NIXL public source baseline: <https://github.com/ai-dynamo/nixl> at commit
  `b458bf0cdc1d21dd7d3130a14a09441109906569`.
- Project version and plugin list are in `meson.build` at that commit:
  `UCX`, `LIBFABRIC`, `POSIX`, `OBJ`, `GDS`, `GDS_MT`, `MOONCAKE`, `HF3FS`,
  `GUSLI`, `GPUNETIO`, `UCCL`, and `AZURE_BLOB`.

## Plugin Discovery

- NIXL uses backend plugin libraries named like `libplugin_<BACKEND>.so`; see
  `src/core/nixl_plugin_manager.cpp` around the `backendPluginPrefix`
  definition.
- Plugin discovery can use `NIXL_PLUGIN_DIR`, a library-relative `plugins`
  directory, plugin-list entries, and static plugins; see
  `src/core/nixl_plugin_manager.cpp` `getPluginDir`,
  `discoverPluginsFromList`, `discoverPluginsFromDir`, `loadBackendPlugin`,
  and `registerBuiltinPlugins`.
- Python exposes `get_plugin_list()`, `get_plugin_params()`,
  `get_plugin_mem_types()`, `create_backend()`, and `get_backend_mem_types()`;
  see `src/api/python/_api.py`.

## Backend Capability Baseline

Use these only as a current-source baseline. Installed builds can differ.

| Backend | Current checked capability | Source anchor |
| --- | --- | --- |
| `UCX` | Local + remote + notifications; supports `DRAM_SEG` and `VRAM_SEG`. | `src/plugins/ucx/ucx_backend.h`; `src/plugins/ucx/ucx_backend.cpp` |
| `LIBFABRIC` | Local + remote + notifications; README describes OFI/libfabric RDMA and validated AWS EFA compatibility. `DRAM_SEG` is present; `VRAM_SEG` is added when the source detects CUDA or AWS Neuron (`FI_HMEM_NEURON`) runtime support. | `src/plugins/libfabric/README.md`; `src/plugins/libfabric/libfabric_backend.h`; `src/plugins/libfabric/libfabric_backend.cpp` |
| `GDS` | Local only, no notifications; supports `DRAM_SEG`, `VRAM_SEG`, and `FILE_SEG`. | `src/plugins/cuda_gds/gds_backend.h`; `src/plugins/cuda_gds/README.md` |
| `GDS_MT` | Local only, no notifications; supports `DRAM_SEG`, `VRAM_SEG`, and `FILE_SEG`. | `src/plugins/gds_mt/gds_mt_backend.h` |
| `POSIX` | Local only, no notifications; supports `FILE_SEG` and `DRAM_SEG`. | `src/plugins/posix/posix_backend.h`; `src/plugins/posix/README.md` |
| `OBJ` | Local only, no notifications. Standard S3 implementation supports `DRAM_SEG` and `OBJ_SEG`; vendor-accelerated implementations such as the Dell `s3_accel` engine can expose `VRAM_SEG`, but this is build/vendor-specific. | `src/plugins/obj/obj_backend.h`; `src/plugins/obj/s3/engine_impl.h`; `src/plugins/obj/s3_accel/dell/engine_impl.h`; `src/plugins/obj/README.md` |

## Selection Baseline

- During transfer request creation, current NIXL source checks local and remote
  registered memory types and common backend engines when no backend is
  explicitly requested.
- If multiple candidates fit, the checked source currently loops through the
  backend set and finds a local match. This is an implementation detail, not a
  stable preference contract; treat ordering as `TBD-3`.
- Source anchor: `src/core/nixl_agent.cpp` around the local/remote backend-set
  query and "loop through and find first local match" comment.

## Framework Baseline

These are pinned public references. Installed wheels, source commits, and
container images can differ.

- vLLM pinned source `b730c4635288d75da4788bc28d8d26b5e5c3726c` documents
  `NixlConnector`, UCX as the default NIXL connector backend, and
  `kv_connector_extra_config.backends` for backend selection:
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/docs/features/nixl_connector_usage.md>.
- vLLM pinned compatibility docs require matching the vLLM/NIXL connector
  version and workload properties:
  <https://github.com/vllm-project/vllm/blob/b730c4635288d75da4788bc28d8d26b5e5c3726c/docs/features/nixl_connector_compatibility.md>.
- Dynamo pinned source `5bff35311517b0863549de00e1969810f853c6f5` documents
  Kubernetes disaggregated communication with UCX/libfabric context and
  EFA-specific libfabric guidance:
  <https://github.com/ai-dynamo/dynamo/blob/5bff35311517b0863549de00e1969810f853c6f5/docs/kubernetes/disagg-communication-guide.md>.
- SGLang pinned source `ac83d8a3392ae3881b645e3b7597aee617d018c3` documents
  `--disaggregation-transfer-backend nixl` for choosing NIXL as the transfer
  mechanism, and source/docs expose `SGLANG_DISAGGREGATION_NIXL_BACKEND` plus
  JSON backend parameters for choosing the NIXL sub-backend:
  <https://github.com/sgl-project/sglang/blob/ac83d8a3392ae3881b645e3b7597aee617d018c3/docs/advanced_features/pd_disaggregation.md>,
  <https://github.com/sgl-project/sglang/blob/ac83d8a3392ae3881b645e3b7597aee617d018c3/python/sglang/srt/disaggregation/nixl/conn.py>.
