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
