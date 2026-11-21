# nixl

This is a *meta package*. The PyPI distribution installs both the CUDA 12
and CUDA 13 backends, and the correct one is selected automatically at
runtime based on the CUDA version reported by PyTorch. Source builds install
a single backend unless built with `-Drelease_wheel=true`.

```bash
pip install nixl
```

The `nixl[cu12]` and `nixl[cu13]` extras are accepted for backwards
compatibility but have no additional effect.

Source builds configured with `-Dbuild_nixl_ep=false` omit the `nixl_ep`
dispatcher from this meta wheel, while retaining the ordinary `nixl` API and
its shared helper module. With `-Dbuild_nixl_ep=true`, the dispatcher is
included; using EP also requires a backend built with compatible CUDA and
PyTorch support. Install the meta wheel and its matching backend wheel
together when updating a downstream image.

The existing Meson wheel target uses setuptools' `clean --all` command before
building each wheel. This removes setuptools' cached build output inside the
meta-package directory so that reconfiguring `build_nixl_ep` does not retain
files from an earlier wheel.

The meta wheel requires `uv` to build. Packaging regression tests run without
CUDA or native NIXL dependencies. From the repository root, with `meson`,
`ninja`, `uv`, and `setuptools>=80.9.0` available, run:

```bash
python -m unittest discover -s test/python -p test_meta_wheel.py -v
```
