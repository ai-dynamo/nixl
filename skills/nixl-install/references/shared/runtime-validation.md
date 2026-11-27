# Runtime validation

These checks show which NIXL a workload loads and how far it initializes. Run
only checks named by the active solution branch, then use that solution's
diagnosis and fixes.

Run as the workload's user, in its container or pod, interpreter, and
environment. Use a disposable process or authorized reproduction for imports,
native loading, backend creation, or GPU allocation. Do not restart or probe a
production workload only to add diagnostics. Read linked source and
documentation from the deployed release or target ref when it differs from
this repository.

When runtime logs are needed, set `NIXL_LOG_LEVEL=INFO` before NIXL loads. The
native [logger initialization](../../../../src/utils/common/nixl_log.cpp) reads
the setting while the library loads and defaults to `WARN`; when the build or
release emits them, build identity, UCX binding, and plugin `dlopen` failure
details are logged at `INFO`.

## Imported NIXL build

Follow the workload's import order. Record the resolved paths for `nixl` and
`nixl._bindings`, and the process exit status because native failures can occur
during import or shutdown. Import `nixl_ep` only when the workload uses it, and
record its file and loaded extension.

A matching version or discoverable module does not prove that the expected
native build loaded. Validate installed CUDA or ROCm backend distributions—for
example, `nixl-cu12` and `nixl-cu13`—using the deployed meta package's
dependencies and selector. Package count alone does not determine correctness.
Interpret selection with the deployed
[NIXL selector](../../../../src/bindings/python/nixl-meta/nixl/__init__.py),
[selector utilities](../../../../src/bindings/python/nixl-meta/nixl_meta_utils.py),
[`nixl_ep` selector](../../../../src/bindings/python/nixl-meta/nixl_ep/__init__.py),
and
[`nixl_ep` extension loader](../../../../examples/device/ep/nixl_ep/__init__.py).

## PyTorch and CUDA without NIXL

Record `torch.__version__` and `torch.version.cuda`. When authorized and
practical, run one bounded CUDA allocation without importing NIXL. A printed
CUDA version is not a runtime check. Avoid this check on a production GPU when
another CUDA context could interfere with the workload.

A failure here identifies the PyTorch, CUDA, container, or host layer rather
than NIXL.

## Environment and native loading

Inspect the workload process for Python paths, virtual-environment state,
`NIXL_*`, `UCX_*`, `LD_LIBRARY_PATH`, and `LD_PRELOAD`, plus NIXL configuration
files and system linker configuration when relevant. Compare the resolved
binding, `nixl_ep` extension when used, plugin unless statically linked, and
backend-loaded dependencies with the intended installation. Use the
[configuration source](../../../../src/utils/common/configuration.cpp),
[plugin manager](../../../../src/core/nixl_plugin_manager.cpp), and
[environment reference](../../../../fern/docs/pages/resources/environment-variables.md).
A dependency inventory does not prove which library the process loaded; use
the real process's mapped libraries.

## Backend and memory

Start a disposable process with logging configured as above. Create only the
required backend using a bounded timeout, and retain the exit status, log, and
any timeout or termination outcome. Apply the same bound to representative
accelerator-memory registration, and run it only when the workload uses that
memory type and the check is authorized.

A listed plugin or advertised memory type does not prove backend creation or
memory registration. Use the
[Python API](../../../../src/api/python/_api.py),
[plugin manager](../../../../src/core/nixl_plugin_manager.cpp), and selected
[backend documentation](../../../../fern/docs/pages/user-guide/backends/index.md).

## Conflicting UCX copies

Run this check only when backend, registration, crash, or hang evidence points
to UCX. Inspect the real workload process's mapped libraries and, when the
deployed release logs it, the UCX binding reported by NIXL at `INFO`.

A conflict exists when `libucp` or `libucs` resolves from an unintended
installation or from more than one installation, or when transport modules are
duplicated or come from a different installation than those core libraries.
Distinct UCX component libraries from one installation are normal. An
intentionally private or SONAME-suffixed UCX is also not a conflict when NIXL
binds to that intended installation. A separate probe is only supporting
evidence. Use the
[UCX plugin source](../../../../src/plugins/ucx/ucx_plugin.cpp) and installed
UCX documentation to interpret paths.

When the deployed UCX plugin supports `NIXL_PLUGIN_UCX_EXPECTED_SONAME` or its
deprecated `NIXL_UCX_EXPECTED_SONAME` fallback, treat an existing nonempty
value from the environment or NIXL configuration file only as a guard. The
primary name takes precedence when set. The configured substring must appear
in the resolved `libucp` path or UCX plugin creation fails. Do not infer or set
a value from this guidance, and do not use the guard instead of inspecting the
real process's mapped core and transport libraries.

## Framework readiness

Run this check only for a framework integration. Read the running workload's
command line, named configuration files, effective configuration, and connector
logs. Prefer passive evidence over endpoints that can trigger work. Confirm
that the effective configuration enables NIXL before treating an unavailable
message as an installation defect.

Compare readiness with the installed framework release's documentation and
source. An import, open port, or early initialization message does not prove
connector readiness. Peer handshakes and transfers are later layers.

## Evidence boundaries

Each check proves only its own layer. A package version, installation, import,
listed plugin, mapped library, open port, or isolated backend success does not
verify later layers.
