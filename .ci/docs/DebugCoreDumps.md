# Debugging Core Dumps from CI Failures

When a GPU CI test step crashes, the pipeline automatically exports the container
filesystem to a `.sqsh` file on the shared NFS mount. This guide explains how to
load it and run `gdb` against the crash.

## Step 1: Find the sqsh file

The export path is printed in the Jenkins build log under the step:

```
exportEnroot: export 'nixl-ci-<ucx_version>-<BUILD_NUMBER>' -> <path>
```

Files follow this naming convention (the sqsh is named after the container):

```
/enroot_images/nixl-ci-<ucx_version>-<BUILD_NUMBER>.sqsh
```

Example: `/enroot_images/nixl-ci-v1.21.x-2209.sqsh`

## What the sqsh contains

`enroot export` produces a fully merged image — the PR base image layers and the
writable overlay (runtime-compiled Rust binaries, pip-installed Python packages, any
files written during the test, core dumps) are combined into a single portable file.
You do not need to pull the PR image from Artifactory separately.

## Step 2: Import the container

Run via `scctl` + `srun` from wherever you have cluster access (no direct node login
needed). `enroot create` is non-interactive:

```bash
scctl client connect -- srun -p mizu --ntasks=1 \
  env ENROOT_DATA_PATH=/enroot-data/enroot-data-148069/user-148069 \
  enroot create --name nixl-ci-<ucx_version>-<BUILD_NUMBER> \
  /enroot_images/nixl-ci-<ucx_version>-<BUILD_NUMBER>.sqsh
```

Example:

```bash
scctl client connect -- srun -p mizu --ntasks=1 \
  env ENROOT_DATA_PATH=/enroot-data/enroot-data-148069/user-148069 \
  enroot create --name nixl-ci-v1.21.x-2209 \
  /enroot_images/nixl-ci-v1.21.x-2209.sqsh
```

## Step 3: Start the container

```bash
scctl client connect -- srun -p mizu --ntasks=1 --pty \
  env ENROOT_DATA_PATH=/enroot-data/enroot-data-148069/user-148069 \
  enroot start --rw nixl-ci-v1.21.x-2209
```

You now have an interactive shell inside the exact environment that failed.

## Step 4: Find core dumps

Core dumps are written to the process working directory at crash time and are
preserved in the exported sqsh.

Inside the container:

```bash
find /workspace/nixl /tmp -maxdepth 4 -type f -name "core.*" \
  ! -path "*/subprojects/*" ! -path "*/nixl_build/*"
```

Core file naming follows the host `core_pattern`: `core.<binary>.<pid>.<signal>.<timestamp>`

Expected locations by test type:
- **C++ tests**: `/workspace/nixl/core.<binary>.<pid>.11.<timestamp>`
- **Rust tests**: `/workspace/nixl/target/debug/core.*`
- **Python tests**: `/workspace/nixl/` or test working directory

## Step 5: Run gdb

```bash
# nixl binaries with full debug symbols are in /opt/nixl/bin/
ls /opt/nixl/bin/

gdb /opt/nixl/bin/<binary> /workspace/nixl/core.<binary>.<pid>.11.<timestamp>

# Inside gdb:
(gdb) set solib-search-path /opt/nixl/lib:/opt/nixl/lib/x86_64-linux-gnu/plugins
(gdb) info sharedlibrary
(gdb) bt
```

## Debug symbol coverage

| Component | Symbols | Notes |
|-----------|---------|-------|
| nixl (`/opt/nixl/`) | Full | Built with `--buildtype=debug` |
| UCX (`/opt/ucx/`) | None | Built with `make install-strip`; UCX frames show as `??` in backtraces |
