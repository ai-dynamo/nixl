# Framework-supported NIXL

## Investigate if needed

Start with the first observed failure below. Run only the checks named by that
branch, adding a conditional check only when its stated evidence appears. Stop
when the evidence supports one fix or identifies a non-NIXL failing layer. If
that layer passes and the original failure remains, continue with the next
observed layer. The
[runtime validation reference](../shared/runtime-validation.md) defines how
shared checks run and what their evidence proves; this file defines the
framework-supported expectation and fixes.

1. **NIXL is unavailable or import fails:** use the
   [Framework package check](#framework-package-check) and shared [Imported NIXL build](../shared/runtime-validation.md#imported-nixl-build). Add [Environment and native loading](../shared/runtime-validation.md#environment-and-native-loading)
   when the original exception names a native library. If the only evidence is
   a short framework error, inspect its original exception first. If that
   exception points to PyTorch or CUDA, see
   [PyTorch or CUDA fails without NIXL](#pytorch-or-cuda-fails-without-nixl). Stop there only
   when the failure reproduces without NIXL; otherwise return to the original
   NIXL failure. When imports pass but the connector still fails, see
   [The connector did not initialize although imports pass](#the-connector-did-not-initialize-although-imports-pass).
2. ### PyTorch or CUDA fails without NIXL

   use shared [PyTorch and CUDA without
   NIXL](../shared/runtime-validation.md#pytorch-and-cuda-without-nixl) and stop at that layer.
3. **A plugin, backend, or GPU-memory registration fails, or a crash or hang
   names UCX:** use shared [Environment and native loading](../shared/runtime-validation.md#environment-and-native-loading) and [Backend and
   memory](../shared/runtime-validation.md#backend-and-memory). Also use the [Framework package check](#framework-package-check)
   when a loaded binding or UCX
   path comes from an unexpected `nixl_cu*` tree. Continue with [Conflicting
   UCX copies](../shared/runtime-validation.md#conflicting-ucx-copies) only when the evidence points to UCX.
4. ### The connector did not initialize although imports pass

   use shared
   [Framework readiness](../shared/runtime-validation.md#framework-readiness). When the effective configuration does not enable
   NIXL, stop investigating. When the user expects NIXL, follow the remedy
   [The effective framework configuration does not enable expected NIXL, or the
   integration still fails after NIXL imports and its required backend
   initializes](#the-effective-framework-configuration-does-not-enable-expected-nixl-or-the-integration-still-fails-after-nixl-imports-and-its-required-backend-initializes).

### Framework package check

Identify the workload's Python executable, Python version, architecture,
framework release, PyTorch version, and every installed `nixl*` distribution
with its location and requirements. Compare them with what that framework
release declares. Do not infer the workload's state from one package version
string. Use the dependency declaration that applies to this workload's
framework release and installation method, such as its package metadata,
connector requirements, lockfile, or image recipe. Prefer the exact version
selected by its lockfile, supported image, or validation record. If only a
range is declared, do not automatically choose the newest satisfying version.
The
[meta-package selector](../../../../src/bindings/python/nixl-meta/nixl/__init__.py)
and
[selector utilities](../../../../src/bindings/python/nixl-meta/nixl_meta_utils.py)
select the backend variant. Validate installed CUDA or ROCm backend
distributions using the deployed meta package's dependencies and selector;
package count alone does not determine correctness.

## Fix

Apply only the smallest remedy supported by the evidence. Record the current
package origins, image digest, and changed configuration values before making
an authorized change. Do not copy a package recipe or compatibility matrix from
another release.

1. **The NIXL package set is missing or differs from the framework release.**
   Restore the package set declared by that release in the workload's
   interpreter, using its supported installation method or image. When that
   method installs NIXL wheels directly, confirm their declared dependencies
   are already satisfied, then install together every NIXL distribution
   declared directly by that release and every NIXL distribution required by
   its declared meta package, without resolving dependencies. When dependency
   resolution is disabled, name every required NIXL distribution explicitly;
   extras do not install their dependencies. Use force-reinstall only when
   replacing an already-installed set, including a same-version local build
   that would otherwise remain installed. With authorization, remove only
   recorded NIXL distributions proven outside the declared set. Preview changes
   first, and do not let the installation replace PyTorch, CUDA, or another
   shared stack component. If package metadata matches but runtime paths
   redirect loading, use the remedy
   [A loader, plugin, or runtime configuration override redirects or breaks
   NIXL](#a-loader-plugin-or-runtime-configuration-override-redirects-or-breaks-nixl).
2. **PyTorch or CUDA fails without NIXL.** Make no NIXL change. Attribute the
   failure to the observed runtime, container, or host layer. Driver, kernel,
   firmware, BIOS, and privileged host changes require their owner and explicit
   authorization.
3. **The framework's expected NIXL build is incompatible with its Python,
   PyTorch, or CUDA environment.** Use a combination supported by that
   framework release, normally its supported image. Determine published build
   coverage from release-matched package metadata and build records rather than
   a hard-coded matrix. If no compatible published build exists, report that
   limitation instead of guessing a version.
4. ### A loader, plugin, or runtime configuration override redirects or breaks NIXL

   Correct only the value proven to redirect loading or break backend
   or memory initialization, using the
   [configuration source](../../../../src/utils/common/configuration.cpp) and
   [plugin manager](../../../../src/core/nixl_plugin_manager.cpp) for the
   deployed release. When the proven override is a UCX setting, also use the
   [UCX configuration source](../../../../src/plugins/ucx/config.cpp). Remove an
   inherited override only after confirming that no other component needs it.
   Do not use deep binding, preload another UCX, delete system libraries, or
   copy libraries from another image as a general fix. If required components
   need incompatible UCX installations with no removable override, use a
   framework-supported single-UCX image or report that no supported runtime fix
   exists.
5. **A required plugin or native dependency is absent.** If the framework's
   package should contain the plugin, restore that package. If the wheel
   intentionally excludes a userspace dependency, use the selected backend's
   release-matched requirements and add it to the image only with appropriate
   authorization. The deployed release's wheel exclusions are defined by its
   matching [wheel build script](../../../../contrib/build-wheel.sh). Treat host
   drivers and privileged OS packages as host changes, not package repairs.
6. ### The effective framework configuration does not enable expected NIXL, or the integration still fails after NIXL imports and its required backend initializes

   Correct only the effective framework setting, version check,
   or initialization input identified by that framework release's source and
   documentation. If the release exposes no supported control, report an
   integration limitation. Do not treat peer connectivity or transfer fallback
   as an installation fix.

## Verify if possible

Rerun the check that exposed the failure. Then verify only relevant or changed
layers and their practical dependencies with the
[runtime validation reference](../shared/runtime-validation.md). Leave
unavailable evidence unverified; do not install or build solely to verify.

1. The workload interpreter's package set matches the framework release's
   expectation, and the loaded NIXL files come from that set.
2. When no NIXL change was made, rerunning [PyTorch and CUDA without NIXL](../shared/runtime-validation.md#pytorch-and-cuda-without-nixl)
   after the layer owner's change no longer produces the original error. NIXL
   verification does not apply.
3. `nixl` imports there; `nixl_ep` also imports when the framework uses it; and
   the real process maps required native libraries from the intended
   installation.
4. The required backend initializes. When accelerator-memory registration was
   relevant and the check is authorized, representative registration succeeds.
5. The real framework initializes its NIXL integration and reaches its
   release-defined ready state without the original error. When the release
   exposes no observable ready state, leave readiness unverified and report
   whether the original error remains.

Apply the shared [Evidence boundaries](../shared/runtime-validation.md#evidence-boundaries).
