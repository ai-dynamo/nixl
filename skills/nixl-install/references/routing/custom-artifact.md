# Chosen or custom NIXL build

## Scope

### When to use

1. Use this route when software that uses NIXL, such as vLLM or SGLang, should
   run a specific NIXL build instead of its supported or published package, or
   when no compatible published NIXL build exists.
2. Use this route to choose, build, integrate, or install that build; resolve
   its build, import, native-library or plugin load, backend creation, or
   initialization failure before a transfer starts; or confirm that the
   intended build is loaded instead of another copy.

### When not to use

1. Do not use this route to install or restore a compatible supported or
   published NIXL package when no different build is needed, including after
   an unintended version mismatch or unexpected local build.
2. Do not use this route for runtime transfer problems after the intended
   build is loaded and initialized.
3. Do not use this route to develop, build, or test NIXL itself unless the
   resulting build is intended for software that uses NIXL.

## How to check when unclear

1. Passive evidence in requirements, image or deployment files, package
   metadata, and build or startup logs includes an explicit requirement for a
   different version, source, branch, commit, patch, wheel, dependency such as
   UCX, build option, or backend the supported or published package lacks.
2. It also includes no compatible published build for the target platform,
   architecture, Python, PyTorch, or CUDA; a build or image-integration failure,
   such as the built package being absent from the image; or the framework
   still loading its pinned build.
3. A version string alone does not prove which build is loaded, and the
   presence of a local or custom build alone does not prove it was intended.
4. Required routing facts are whether a different build is intentionally
   required or no compatible published build exists, what identifies the
   target build, and, if a failure was reported, whether it occurred before a
   transfer started.
