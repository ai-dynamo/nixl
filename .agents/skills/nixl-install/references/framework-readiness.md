# Framework Readiness

Use this reference only after NIXL importability and basic plugin evidence are
separated from framework connector setup. Match all framework facts to the
user's installed framework version, source commit, image tag, or docs link
before treating them as version-specific.

## vLLM

- Current vLLM NIXL connector docs identify the connector config name
  `NixlConnector`.
- Current vLLM API docs identify the class path
  `vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector.NixlConnector`.
- Current docs describe `kv_connector_extra_config.backends` as the backend
  plugin selector surface.
- Current compatibility docs are latest/developer-preview guidance unless
  matched to the user's installed vLLM version.

Use these as framework connector checks, not as proof that NIXL plugins are
loadable in the user's runtime.

## Dynamo

Current Dynamo Kubernetes disaggregated communication docs cover worker pod
RDMA resources/capabilities, UCX/libfabric environment variables, and a vLLM
disaggregated communication path using `--kv-transfer-config` values such as
`NixlConnector`, `kv_role`, `kv_buffer_device`, and
`kv_connector_extra_config.backends`.

For a real diagnosis, compare the user's config and logs with docs or source
matching the installed Dynamo version. Use latest docs only as fallback and
record `TBD-2` plus the docs channel and installed-version match.

## SGLang

Current SGLang public docs/source cover NIXL for PD disaggregation with
`--disaggregation-transfer-backend nixl`, plus
`SGLANG_DISAGGREGATION_NIXL_BACKEND` and
`SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS` for backend selection.

If the installed SGLang version differs from current public docs/source, record
`TBD-2` before naming concrete flags or environment variables as applicable to
that installation.
