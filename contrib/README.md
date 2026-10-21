## Build options

### Build & install in a conda environment
Refer to [README.md](../README.md) for instructions on how to install prerequisites.
```
$ conda activate <ENV_NAME>
$ export PKG_CONFIG_PATH=<PATH_TO_UCX>/lib/pkgconfig:$PKG_CONFIG_PATH

$ meson setup nixl-build -Ducx_path=<PATH_TO_UCX> --prefix=<PATH_TO_CONDA_ENV>
$ cd nixl-build
$ ninja
$ ninja install

$ cd ..
$ pip install . --config-settings=setup-args="-Ducx_path=<PATH_TO_UCX>"
$ pip list | grep nixl
$ python -c "from nixl._api import nixl_agent; print('Success')"
```

### Inspecting a release wheel set
`contrib/wheel_inspect` compares a candidate wheel set against a reference
release and reports the eight checks from the release audit: SPCX and DDN
INFINIA plugin bundling, the SPCX ref the modules were built from, the UCX
release and commit, files added/deleted/rehashed/changed per matrix cell, the
NIXL and UCX plugin sets, and compressed sizes. Each check is `PASS`/`FAIL`
(code can judge), `REVIEW` (a list for a person to read) or `SKIPPED` (an
input such as the build manifest is missing). It carries no per-release
expectations: whatever differs from the reference is shown.

```bash
# previous release vs the one before it, both from PyPI
python3 contrib/wheel_inspect release-run --candidate pypi:1.5.0 --reference pypi:1.4.1 --out-dir /tmp/inspect
# a release candidate folder in Artifactory vs the latest earlier PyPI release
ARTIFACTORY_USER=... ARTIFACTORY_TOKEN=... \
python3 contrib/wheel_inspect release-run \
    --candidate artifactory:https://artifactory.nvidia.com/artifactory/sw-nbu-swx-nixl-pypi-local/release/1.6.0/<build> \
    --reference auto --out-dir /tmp/inspect --no-publish
```

The result is `wheel-inspection.{json,md,html}` in `--out-dir`. Both sides
must be complete sets (every CUDA x Python x arch cell plus the meta wheel), or
the run stops before making any claim. A `build-manifest.json` next to the
wheels, written by the nightly wheel build, feeds the SPCX and UCX checks;
without it those are reported from what the wheel itself carries. Unit tests
live in `contrib/tests`.
