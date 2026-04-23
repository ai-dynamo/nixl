# nixl

This is a *meta package* that installs both CUDA 12 and CUDA 13 backends.
At runtime, the correct backend is selected automatically based on the CUDA
version reported by PyTorch.

```bash
pip install nixl
```

The `nixl[cu12]` and `nixl[cu13]` extras are accepted for backwards
compatibility but have no additional effect.
