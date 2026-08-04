# CI Crash Debug — DevOps Reference

This document describes how the automatic container export works on crash and what
operational steps are needed to keep it functioning.

## How it works

When a Slurm run step (`slurmCI` → `slurm.run()` in `swx-jenkins-lib`) exits with a
non-zero code, the pipeline exports the container's full filesystem to a `.sqsh` file on
the shared NFS mount. This lets engineers load the exact environment that failed and run
`gdb` against it without any manual reproduction steps.

The export is driven by a single arg added to each `run` step in `test-matrix.yaml`:

| Arg | Value | Meaning |
|-----|-------|---------|
| `exportEnroot` | `"true"` | On step failure, `enroot export` the container to `${ENROOT_MOUNT_PATH}/<containerName>.sqsh` |

`slurm.run()` in `swx-jenkins-lib/vars/slurm.groovy` runs the test with
`returnStatus: true` (so the export can run after a failure), calls `enroot export` via a
follow-up `srun` on the same job allocation, then re-raises the failure so the build still
fails. The output path is `${ENROOT_MOUNT_PATH}/<containerName>.sqsh` — the container name
is already unique per build, so it doubles as the sqsh filename.

The relevant env vars in `test-matrix.yaml`:

```yaml
ENROOT_MOUNT_PATH: "/enroot_images"                                  # destination dir for the export
ENROOT_DATA_PATH: "/enroot-data/enroot-data-148069/user-148069"     # svc-nixl enroot data dir on mizu
```

`ENROOT_DATA_PATH` must point to the svc-nixl user's enroot data directory on the mizu
nodes — enroot requires it to locate the container's writable layer. It is forwarded to
the export `srun` only when set.

## File naming convention

```
${ENROOT_MOUNT_PATH}/<containerName>.sqsh
```

`containerName` is `nixl-ci-${ucx_version}-${BUILD_NUMBER}`, so:

```
/enroot_images/nixl-ci-<ucx_version>-<BUILD_NUMBER>.sqsh
```

Example: `/enroot_images/nixl-ci-v1.21.x-2209.sqsh`

One file per build per matrix axis. If multiple run steps fail (cpp, python, rust,
nixlbench), they all target the same sqsh path — the last export wins, which is the
container after all test layers have been applied.

## NFS mount on mizu

The path `/enroot_images` must be:
- Mounted on all mizu Slurm compute nodes
- Writable by `svc-nixl` (UID 148069, GID 30 / `hardware` group)

The export command that runs on the mizu node:

```bash
srun --jobid=<JOB_ID> --ntasks=1 --oversubscribe \
  env ENROOT_DATA_PATH=/enroot-data/enroot-data-148069/user-148069 \
  enroot export --output /enroot_images/nixl-ci-<ucx_version>-<BUILD_NUMBER>.sqsh \
  pyxis_nixl-ci-<ucx_version>-<BUILD_NUMBER>
```

## gdb in the container

`gdb` and `gdb-multiarch` are installed in `Dockerfile.gpu-test` so that engineers
can run gdb directly inside the exported container without installing anything.
The install runs as `sudo` (before `COPY`/`build.sh`) because the base image
(`Dockerfile.base`) switches to `USER svc-nixl` before our Dockerfile inherits from it.

## Cleanup

**There is no automated cleanup.** sqsh files accumulate on `/enroot_images` until
manually removed.

Action items:
- Set up a cron job on the mizu cluster (or a Jenkins pipeline_stop hook) to delete
  old files under `/enroot_images/nixl-ci-*.sqsh`
- Coordinate with the cluster admin who manages the NFS mount
