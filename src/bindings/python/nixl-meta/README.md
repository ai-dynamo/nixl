# nixl

This is a *meta package*. The PyPI distribution installs the CUDA backend
package. Source builds install the backend they were built for unless built
with `-Drelease_wheel=true`.

```bash
pip install nixl
```

The `nixl[cu13]` extra is accepted for backwards compatibility but has no
additional effect.
