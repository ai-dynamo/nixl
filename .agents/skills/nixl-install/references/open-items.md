# NIXL Install Open Items

This reference tracks NIXL install facts that remain unresolved or version-specific
after the 2026-05-21 source-verification pass. Reference these IDs from
`SKILL.md`, evals, and diagnosis reports instead of writing unnumbered TBDs.

## TBD-1: Exact CUDA/Wheel Compatibility Matrix

Need a matrix by NIXL release, CUDA minor version, Python ABI, platform, and
wheel tag. Public PyPI metadata resolves released wheel tags, but public sources
have not verified CUDA minor compatibility by release.

Source-backed fact: current NIXL docs and PyPI project metadata say
`pip install nixl` installs CUDA 12 and CUDA 13 backends and selects at runtime
from PyTorch's CUDA version.

Verified public wheel-tag slices:

- `nixl` `1.1.0`: `py3-none-any`, `Requires-Python >=3.10`, extras for `cu12`
  and `cu13`.
- `nixl-cu12` / `nixl-cu13` `1.1.0`: CPython `cp310` through `cp314`,
  `manylinux_2_28_x86_64`, and `manylinux_2_28_aarch64`.
- Historical public summary from PyPI JSON: `0.7.1` has `cp39-cp312`; `0.8.0`
  has `cp39-cp313`; `0.9.0` through `1.1.0` have `cp310-cp314`, for both
  `x86_64` and `aarch64` manylinux_2_28 CUDA-major packages.

Evidence refs checked on 2026-05-21:

- PyPI `nixl`: <https://pypi.org/project/nixl/>
- PyPI `nixl` JSON API: <https://pypi.org/pypi/nixl/json>
- PyPI `nixl-cu12`: <https://pypi.org/project/nixl-cu12/>
- PyPI `nixl-cu12` JSON API: <https://pypi.org/pypi/nixl-cu12/json>
- PyPI `nixl-cu13`: <https://pypi.org/project/nixl-cu13/>
- PyPI `nixl-cu13` JSON API: <https://pypi.org/pypi/nixl-cu13/json>
- NIXL source notes: `source-notes.md`

Gap: no public matrix was verified that maps every release/wheel tag to CUDA
minor compatibility.

## TBD-2: Framework-Version-Specific Connector Behavior

Need connector behavior for the user's exact installed vLLM, Dynamo, or SGLang
version.

Source-backed fact: latest vLLM, Dynamo, and SGLang docs/source contain NIXL
connector guidance. See `source-notes.md` for public URLs.

Gap: older installed versions can have different flags, class names, defaults,
or connector availability. Match against the user's installed framework version
before making concrete claims.

Public evidence notes:

- vLLM publishes a latest/developer-preview NixlConnector compatibility matrix.
- Current SGLang docs use `--disaggregation-transfer-backend nixl`; current
  source defaults `SGLANG_DISAGGREGATION_NIXL_BACKEND` to `UCX` and accepts JSON
  params through `SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS`.

## TBD-3: User-Environment Plugin Availability

Need the actual plugin list and backend availability from the user's failing
environment.

Source-backed fact: NIXL discovers plugins via `NIXL_PLUGIN_DIR`,
library-relative `plugins/`, `libplugin_<BACKEND>.so`, static plugins, and
optional plugin-list entries. See `source-notes.md` for public source links.

Gap: available plugins depend on the installed package/build and runtime
environment. Confirm with `nixl_agent.get_plugin_list()` plus package files and
native dependency inspection from the failing environment.

Treat file presence as advisory, not proof of plugin loadability. If a local
case depends on non-public issue or release evidence, keep that evidence outside
the production skill contract and cite only a redacted, user-provided summary in
the diagnosis report.

## TBD-4: Mutable Source Freshness

Need durable source pinning when a diagnosis must be reproducible later.

Source-backed fact: source URLs in `source-notes.md` were checked on
2026-05-21.

Gap: `main`, `latest`, and developer-preview documentation can drift. Pin source
commits, release branches, wheel filenames, container tags, and checked dates
when diagnosing a specific installed version.

## Source Snapshot

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
