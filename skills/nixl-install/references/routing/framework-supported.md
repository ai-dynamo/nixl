# Framework-supported NIXL

## Scope

### When to use

Use this route when software such as vLLM or SGLang expects a supported NIXL
package, the user wants that package, and they need to install it or resolve a
problem before a transfer can start. Relevant problems include a missing or
unintended package or build, failed `import nixl`, native-library or plugin
loading error, backend creation error, or NIXL integration initialization
failure. Also use it when the software's setup or effective configuration does
not enable an expected NIXL integration.

Python, PyTorch, CUDA, UCX, or other native-library errors are in scope when
they block one of those stages; the error alone does not prove the cause.

### When not to use

Do not use this route when the user intentionally wants a different NIXL
release or custom build. Also exclude runtime transfer problems after NIXL and
the software's NIXL integration have initialized, including transfer setup,
execution, data correctness, path, or performance. A running workload whose
symptom is fallback or recomputation remains outside this route even when its
effective configuration does not enable NIXL.

## How to check if uncertain

1. Use existing dependency declarations, configuration, and container setup to
   determine whether the software expects a supported NIXL package. Determine
   whether the user wants that package rather than a custom build.
2. Use the reported symptom, configuration, and startup evidence to determine
   whether the problem occurs during setup or before a transfer can start,
   including an expected integration not enabled in setup or effective
   configuration, rather than as runtime transfer behavior or
   running-workload fallback or recomputation.

The required routing facts are whether the user wants the software-supported
package and whether the problem is during setup or before transfer rather than
runtime transfer behavior. Exact package and loaded-build identity are
investigation data, not prerequisites for selecting this route.
