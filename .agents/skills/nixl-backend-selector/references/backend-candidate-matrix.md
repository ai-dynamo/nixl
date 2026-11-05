# Backend Candidate Matrix

Use this reference after the transfer shape, framework/version, and runtime
evidence are known. Treat every row as a candidate direction, not a universal
default.

| Situation | Candidate direction |
| --- | --- |
| Multi-node DRAM/VRAM over InfiniBand or RoCE | Start with `UCX` when the installed framework/source uses UCX and plugin evidence confirms it. Keep fabric, UCX transport selection, GPUDirect RDMA, container, and Kubernetes requirements as `TBD-3` / `TBD-4` until verified. |
| Multi-node DRAM/VRAM on AWS EFA | Treat `LIBFABRIC` as the EFA candidate when the deployment docs or framework selector supports it. Do not generalize EFA guidance to non-EFA fabrics. Keep plugin loadability, `DRAM_SEG` / `VRAM_SEG`, and libfabric provider evidence as `TBD-1` / `TBD-4`. |
| vLLM NIXL connector | Capture installed vLLM/NIXL evidence such as `pip show vllm`, image digest, or source commit first. Current pinned docs say `NixlConnector` defaults to `UCX`; use `kv_connector_extra_config.backends` only when supported by the installed version (`TBD-2`). |
| Dynamo Kubernetes/disaggregated communication | Match the exact documented path, image, and pod spec. Current pinned docs discuss UCX/libfabric and EFA-specific libfabric guidance; config names, defaults, and RDMA requirements remain `TBD-2` / `TBD-4`. |
| SGLang PD disaggregation | Capture installed SGLang source or image first. Current pinned docs/source use `--disaggregation-transfer-backend nixl` to select NIXL as the transfer mechanism, and `SGLANG_DISAGGREGATION_NIXL_BACKEND` or JSON backend parameters to select the NIXL sub-backend. Accepted values remain `TBD-2`. |
| GPU or host memory to local file storage with GPUDirect Storage intent | Prefer `GDS` or `GDS_MT` only after GDS runtime, filesystem/mount/container compatibility, file path, GPU visibility, and plugin loadability are verified (`TBD-1` / `TBD-4`). |
| CPU/file fallback, conventional file I/O, or Docker where GDS is unavailable | Consider `POSIX`; do not present it as a remote network backend. Keep file-system behavior and optional io_uring/container behavior as `TBD-3` / `TBD-4`. |
| S3-compatible object storage | Consider `OBJ`; require bucket/endpoint/credential-source evidence and redact secrets. Treat GPU-direct object storage as vendor/runtime-specific unless the installed implementation proves `VRAM_SEG` (`TBD-1` / `TBD-3` / `TBD-4`). |
| Plugin present but not covered above, such as `MOONCAKE`, `HF3FS`, `GUSLI`, `GPUNETIO`, `UCCL`, or `AZURE_BLOB` | Defer to `TBD-1` and the plugin's upstream README/source for the installed version. Do not infer behavior from analogous backends. |
| Multiple common backends appear valid | Recommend an explicit backend setting for reproducibility and explain that current NIXL source may otherwise select a current-source-dependent first matching backend (`TBD-3`). |
