# Chosen or custom NIXL build

## Investigate if needed

Use the workload's interpreter, user, environment, and image, or a disposable
copy of them. Read build and packaging files at the exact NIXL ref being used.
Choose the matching branch below. Follow the checks it names in this file or
in the [runtime validation reference](../shared/runtime-validation.md), under
the conditions it gives. This file
defines the chosen-build expectation and its use-case-specific fixes.

Run only the checks that can change the remedy. Start with the first observed
failure:

1. **A specific release, commit, patch, backend, or UCX build is required:**
   start with [What exact build is required, and does a published build fit?](#what-exact-build-is-required-and-does-a-published-build-fit)
   When NIXL is already installed, use shared [Imported NIXL build](../shared/runtime-validation.md#imported-nixl-build) and [Which
   NIXL is installed and loaded now?](#which-nixl-is-installed-and-loaded-now) Stop if it is already the requested
   build. Consult [Does the candidate build match the target?](#does-the-candidate-build-match-the-target) only when a
   candidate artifact must be evaluated or built.
2. **The build or packaging failed:** use the build-failure instructions under
   [Does the candidate build match the target?](#does-the-candidate-build-match-the-target) and leave the workload
   unchanged. When the build succeeded but the package is absent from the
   target image, see [Did another step or installation replace or shadow the
   chosen build?](#did-another-step-or-installation-replace-or-shadow-the-chosen-build)
3. **Installation succeeded but the workload still uses another NIXL:** use
   shared [Imported NIXL build](../shared/runtime-validation.md#imported-nixl-build) and [Environment and native loading](../shared/runtime-validation.md#environment-and-native-loading), then see
   [Which NIXL is installed and loaded now?](#which-nixl-is-installed-and-loaded-now) and [Did another step or
   installation replace or shadow the chosen build?](#did-another-step-or-installation-replace-or-shadow-the-chosen-build)
4. **The chosen build is installed but `nixl` or `nixl_ep` fails to import:**
   use shared [Imported NIXL build](../shared/runtime-validation.md#imported-nixl-build) and [Environment and native loading](../shared/runtime-validation.md#environment-and-native-loading), then
   see [Which NIXL is installed and loaded now?](#which-nixl-is-installed-and-loaded-now) and [Did another step or
   installation replace or shadow the chosen build?](#did-another-step-or-installation-replace-or-shadow-the-chosen-build) Also consult [Does the
   candidate build match the target?](#does-the-candidate-build-match-the-target) when the error names a missing CUDA
   variant or `nixl_ep` extension. If the error points to PyTorch or CUDA, use
   shared [PyTorch and CUDA without NIXL](../shared/runtime-validation.md#pytorch-and-cuda-without-nixl) before changing NIXL.
5. **The chosen NIXL initializes alone but the framework rejects or fails with
   it:** if not already established, use shared [Imported NIXL build](../shared/runtime-validation.md#imported-nixl-build) and
   [Backend and memory](../shared/runtime-validation.md#backend-and-memory). Then use [Framework readiness](../shared/runtime-validation.md#framework-readiness) and [Does the framework
   release accept this build?](#does-the-framework-release-accept-this-build)
6. **PyTorch or CUDA fails without NIXL:** use shared [PyTorch and CUDA without
   NIXL](../shared/runtime-validation.md#pytorch-and-cuda-without-nixl) and stop at that layer.
7. **The chosen build loads, but a plugin, backend, registration, crash, or
   hang points to native loading or UCX:** use shared [Environment and native
   loading](../shared/runtime-validation.md#environment-and-native-loading) and [Backend and memory](../shared/runtime-validation.md#backend-and-memory). Add [Conflicting UCX copies](../shared/runtime-validation.md#conflicting-ucx-copies) only when
   the evidence points to UCX. Consult [Does the candidate build match the
   target?](#does-the-candidate-build-match-the-target) when a required plugin is absent, and see [Which NIXL is installed
   and loaded now?](#which-nixl-is-installed-and-loaded-now) and [Did another step or installation replace or shadow the
   chosen build?](#did-another-step-or-installation-replace-or-shadow-the-chosen-build) when another NIXL tree may be involved.

### Evidence and sources

1. #### What exact build is required, and does a published build fit?

   Record the
   required version, commit, patch set, backend, UCX revision, and target
   platform. Also record the NIXL requirement or image contents supplied by the
   framework release. For an unmodified published release, inspect its wheel
   set in a disposable copy of the target environment. Check the Python,
   architecture, and platform tags, required plugins, and the `nixl_ep` build
   for the target PyTorch when the framework uses it. Build only when no
   published set satisfies the requirement.
2. #### Which NIXL is installed and loaded now?

   Record every `nixl*`
   distribution, its location, and whether it came from an index, local wheel,
   VCS checkout, or editable install. Use shared [Imported NIXL build](../shared/runtime-validation.md#imported-nixl-build) for
   loaded paths. Set the log level before NIXL loads. The
   `NIXL version: <V> (git: <hash>)` line, when emitted at `INFO` by the
   deployed release's
   [logger initialization](../../../../src/utils/common/nixl_log.cpp) with the
   commit captured by the [common build](../../../../src/utils/common/meson.build),
   identifies the loaded native build; compare it with the intended source
   commit. A manylinux variant wheel may also carry
   `NIXL commit:` and UCX details in its package description, written by the
   [manylinux wheel image](../../../../contrib/Dockerfile.manylinux); do not
   require that record from the meta wheel or other build paths. A version
   string does not identify a custom build. When the runtime reports
   `git: unknown`, preserve loaded-path and artifact-origin evidence but report
   build identity as unverified.
3. #### Does the candidate build match the target?

   Inspect the wheels before
   changing the workload:
   1. Read required variants and versions from the built wheel's `METADATA`
      when available; otherwise use packaging files from the deployed or target
      ref, including the
      [meta-package build](../../../../src/bindings/python/nixl-meta/meson.build)
      when present. All variant wheels must come from the intended source build
      and use a consistent UCX build. Do not expect the meta wheel to contain
      native build identity.
   2. Check `METADATA` for requirements, extras, version, and `Requires-Python`;
      check `WHEEL` `Tag` entries for Python ABI, architecture, and platform
      compatibility. Confirm that required plugins and the required `nixl_ep`
      PyTorch extension are present in the variant selected by the target
      runtime; do not require that target extension from inactive variants.
   3. Interpret the set with that ref's
      [NIXL selector](../../../../src/bindings/python/nixl-meta/nixl/__init__.py)
      and
      [`nixl_ep` selector](../../../../src/bindings/python/nixl-meta/nixl_ep/__init__.py).
      Do not combine wheels from different builds or origins.
   4. For build failures, use the ref's
      [container build script](../../../../contrib/build-container.sh),
      [wheel build script](../../../../contrib/build-wheel.sh),
      [build documentation](../../../../fern/docs/pages/user-guide/building-nixl/index.md),
      and selected
      [backend documentation](../../../../fern/docs/pages/user-guide/backends/index.md).
      Check the script's current `--help`, the effective build options, and
      inherited image, CUDA, PyTorch, UCX, architecture, and platform settings.
      For a host source build, ensure `nvcc` and `ptxas`, plus `CUDA_HOME` when
      set, refer to one toolkit. When a required plugin is absent, confirm
      whether its dependency was missing and the build skipped it. A host build
      that fails on build-tool or dependency-header errors should normally move
      to the container wheel path instead of repairing the host.
4. #### Did another step or installation replace or shadow the chosen build?

   Inspect later image layers, requirements installs, entrypoints, launchers,
   and deployment configuration for another NIXL installation. Compare package
   locations with shared [Imported NIXL build](../shared/runtime-validation.md#imported-nixl-build) and [Environment and native
   loading](../shared/runtime-validation.md#environment-and-native-loading) only when a runtime installation exists. Check `.pth` files,
   editable installs, and user site-packages.
   Libraries such as
   `libibverbs`, `libmlx5`, `libnl-3`, or `libnl-route-3` resolving from
   another NIXL tree show that its loader path still applies. A plugin API
   mismatch outside the chosen installation indicates another plugin tree; one
   inside it indicates a mixed or replaced set.
5. #### Does the framework release accept this build?

   After shared [Imported
   NIXL build](../shared/runtime-validation.md#imported-nixl-build) and [Backend and memory](../shared/runtime-validation.md#backend-and-memory) pass, use [Framework readiness](../shared/runtime-validation.md#framework-readiness) to
   separate configuration from a version or API rejection. Compare a confirmed
   rejection or connector `AttributeError` or `TypeError` with that framework
   release's requirements and connector source.

## Fix

Apply only the remedy supported by the failing check. Record the original
package set and image digest before an authorized change. Commit local NIXL
patches before building so the resulting artifact identifies the source.
Never patch framework source or bypass its NIXL version check.

1. **A published set fits.** Skip the build and continue with
   [Install and integrate the chosen set](#install-and-integrate-the-chosen-set).
2. ### A custom wheel set is required or the build is wrong

   Build in a
   disposable environment using the ref's
   [container build script](../../../../contrib/build-container.sh) with
   `--dockerfile contrib/Dockerfile.manylinux`; the default Dockerfile builds a
   development image, not a wheel set. Use the target ref's build scripts and
   Dockerfile as the source of truth rather than copying a recipe from another
   ref. Target the workload's Python, architecture, platform, CUDA, and PyTorch;
   remove unintended environment overrides; verify the effective base image,
   compiler CUDA major, and wheel platform rather than trusting one requested
   flag; pin a floating `UCX_REF` to one commit when variants must share a UCX
   build; and omit options requiring unavailable credentials or private
   resources. Inspect the generated meta wheel's requirements and build exactly
   the variants it requires. Only the runtime-selected variant needs the
   target PyTorch `nixl_ep` extension. Collect the meta wheel from the path
   produced by that ref; do not assume every variant build copies it into
   `dist/`.

   If the ref cannot produce a fitting complete set, report that limitation
   and offer a compatible ref or framework image instead of changing PyTorch
   or installing a partial set.
3. ### The candidate set is incomplete or inconsistent

   Do not install it.
   Rebuild until the meta wheel and all variants it requires are present, wheel
   tags match the target, build identity and UCX information agree, and every
   required plugin and `nixl_ep` extension is included. Take version-dependent
   plugin requirements from the ref's backend page and plugin build files. Do
   not rename wheels to change their tags.
4. ### Install and integrate the chosen set

   If a later image layer, requirements install, entrypoint, launcher, or
   deployment step replaces or shadows the chosen set, first correct or reorder
   that controlling step; reinstalling before then is temporary.

   For wheel-based installs, use the
   complete meta and variant set from one build or one published release in one
   invocation. Isolate the install from package indexes and disable dependency
   resolution so it cannot fetch another NIXL variant or replace PyTorch,
   CUDA, or unrelated stack components. Use the NIXL repository's
   [vLLM image recipe](../../../../contrib/Dockerfile.vllm) or
   [SGLang image recipe](../../../../contrib/Dockerfile.sglang) as an
   installation pattern, not as a complete image recipe to copy.
   Before installing, confirm that the target environment satisfies the
   applicable non-NIXL `Requires-Dist` entries in the chosen wheels. When
   replacing a different build with the same version, use the force-reinstall
   option shown in the linked recipe.

   For a framework integration, keep corrections in the image recipe, launcher,
   or deployment specification so they can be reverted. Update its NIXL
   requirement or install step, verify that no later step reinstalls the
   original package, and remove only the shadowing path or installation proven
   by the evidence. For a containerized deployment, build a new image from the
   framework image by digest and deploy it by digest after authorization. For a
   non-container framework deployment, install into the exact environment
   identified during investigation and preserve its previous state.
5. **Another NIXL tree or UCX installation supplies native code.** Confirm the
   competing path in a disposable process, then remove its plugin and loader
   path entries together. Ask before removing an explicitly configured
   deployment override. For a UCX conflict, align the chosen set with the one
   UCX installation the workload must use, or remove only the proven competing
   path. Do not preload another UCX or enable runtime deep binding as an ad hoc
   workaround. When the target ref explicitly supports a private or
   SONAME-suffixed UCX build, use that ref's documented build option and verify
   the binding with its
   [UCX binding check](../../../../contrib/check_ucx_binding.py). Do not base a
   framework image on NIXL's development container.
6. **A required plugin or userspace dependency is absent.** If the plugin was
   skipped during the build, follow the two remedies
   [A custom wheel set is required or the build is wrong](#a-custom-wheel-set-is-required-or-the-build-is-wrong) and
   [The candidate set is incomplete or inconsistent](#the-candidate-set-is-incomplete-or-inconsistent). If the plugin is present
   but its wheel intentionally excludes a userspace dependency, add the
   release-matched backend dependency to the target environment or image with
   authorization. Treat host drivers and privileged OS packages as host
   changes.
7. **PyTorch or CUDA fails without NIXL.** Make no NIXL change. Attribute the
   failure to the observed runtime, container, or host layer. Driver, kernel,
   firmware, BIOS, and privileged host changes require their owner and explicit
   authorization.
8. **The framework rejects the chosen build.** Use a framework release that
   supports it when available and authorized. Otherwise report that the
   combination requires a framework change and offer the framework's supported
   NIXL. Do not claim the chosen build can be made compatible through package
   or loader changes.

## Verify if possible

Verify the changed layer and its practical dependencies in the real workload
environment with the
[runtime validation reference](../shared/runtime-validation.md):

1. A wheel-based install contains one complete chosen set; a source install
   contains only the intended build. `nixl`, its bindings, and `nixl_ep` when
   used resolve from that installation.
2. The runtime `git:` value matches the intended full source commit as a
   prefix. For a manylinux variant wheel, its package-description build record
   agrees when present. Required NIXL libraries, plugins, and backend
   dependencies load from the intended locations without an API mismatch. When
   the runtime lacks git metadata, report build identity as unverified rather
   than inferring it from the version.
3. The required backend initializes, and representative accelerator memory
   registers when that layer was relevant and tested.
4. The real framework reaches its release-defined ready state without the
   original error. Processes that use NIXL map only the chosen installation.

Apply the shared [Evidence boundaries](../shared/runtime-validation.md#evidence-boundaries). If build identity or process mappings
cannot be observed, report the chosen build as unverified.
