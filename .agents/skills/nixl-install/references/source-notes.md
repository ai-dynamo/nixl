# NIXL Install Source Notes

Checked on 2026-05-21. Treat `main` and `latest` links as mutable orientation
unless they are matched to the user's installed release, wheel filename, source
commit, or container image. Keep unresolved install facts in `open-items.md`.

## NIXL Install And Wheel Evidence

- PyPI `nixl` project: <https://pypi.org/project/nixl/>
  - Supports the canonical user-facing package name `nixl`.
  - Supports `Requires-Python >=3.10` and extras `cu12` and `cu13`.
  - Current project description says the meta package installs CUDA 12 and
    CUDA 13 backends and selects the backend from PyTorch's reported CUDA
    version.
- PyPI `nixl` JSON API: <https://pypi.org/pypi/nixl/json>
  - Source for release file metadata, package metadata, and wheel tags.
- PyPI `nixl-cu12` project and JSON API:
  <https://pypi.org/project/nixl-cu12/>,
  <https://pypi.org/pypi/nixl-cu12/json>
  - Source for CUDA-12 package release metadata and wheel tags.
- PyPI `nixl-cu13` project and JSON API:
  <https://pypi.org/project/nixl-cu13/>,
  <https://pypi.org/pypi/nixl-cu13/json>
  - Source for CUDA-13 package release metadata and wheel tags.
- NIXL releases page: <https://github.com/ai-dynamo/nixl/releases>
  - Source for packaging change notes that the `nixl` meta wheel bundles
    CUDA-major backends and selects from `torch.version.cuda`.

## NIXL Plugin Discovery And Python API

- NIXL plugin manager source:
  <https://raw.githubusercontent.com/ai-dynamo/nixl/main/src/core/nixl_plugin_manager.cpp>
  - Supports `NIXL_PLUGIN_DIR` precedence over the default library-relative
    `plugins` directory.
  - Supports backend plugin filenames using the `libplugin_` prefix and `.so`
    suffix, optional plugin-list discovery, registered directories, and static
    plugins.
  - Supports treating plugin file presence as discovery evidence, not proof that
    the plugin can load.
- NIXL Python API source:
  <https://raw.githubusercontent.com/ai-dynamo/nixl/main/src/api/python/_api.py>
  - Supports `nixl_agent.get_plugin_list()`, `get_plugin_params()`, and
    `create_backend()` as source-backed Python probes after importability and
    trusted plugin paths are established.
  - Supports `nixl_agent_config(backends=[])` as a no-backend agent
    configuration. Use this for plugin inventory probes so listing plugins does
    not initialize the default backend first.

## Framework Connector Evidence

- vLLM NixlConnector usage guide:
  <https://docs.vllm.ai/en/latest/features/nixl_connector_usage/>
  - Source for the current developer-preview connector name `NixlConnector`,
    `--kv-transfer-config`, `kv_role`, and
    `kv_connector_extra_config.backends`.
- vLLM NixlConnector API docs:
  <https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector/>
  - Source for the current class path
    `vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector.NixlConnector`.
- vLLM NixlConnector compatibility matrix:
  <https://docs.vllm.ai/en/latest/features/nixl_connector_compatibility/>
  - Source for current developer-preview compatibility guidance. Match against
    the user's installed vLLM version before making version-specific claims.
- Dynamo disaggregated communication docs:
  <https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/deployment-guide/disagg-communication>
  - Source for current Dynamo Kubernetes disaggregated communication examples,
    RDMA resource/capability checks, UCX/libfabric environment variables, and
    vLLM `--kv-transfer-config` examples using `NixlConnector`.
- SGLang PD disaggregation docs:
  <https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md>
  - Source for current SGLang NIXL usage with
    `--disaggregation-transfer-backend nixl` and
    `SGLANG_DISAGGREGATION_NIXL_BACKEND`.
- SGLang NIXL connection source:
  <https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/disaggregation/nixl/conn.py>
  - Source for current use of `SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS` as a
    JSON object with string keys and values, and for plugin availability checks
    through `get_plugin_list()`.

## Checked Source Snapshots

- NIXL HEAD checked on 2026-05-21:
  `b458bf0cdc1d21dd7d3130a14a09441109906569`
- vLLM HEAD checked on 2026-05-21:
  `b730c4635288d75da4788bc28d8d26b5e5c3726c`
- Dynamo HEAD checked on 2026-05-21:
  `5bff35311517b0863549de00e1969810f853c6f5`
- SGLang HEAD checked on 2026-05-21:
  `fbebdd5105dafde32e0cada86184b6c53458e98a`
- SGLang HEAD checked on 2026-05-21 in an additional public-source check:
  `32352f7edfd8b236f2a7d257966f6890d7ae4ecf`
