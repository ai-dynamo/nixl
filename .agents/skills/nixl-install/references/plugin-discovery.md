# Plugin Discovery

Use this reference when NIXL importability is proven and the next question is
whether plugins are discoverable and trusted in the same runtime environment.

## Evidence Order

Start with static evidence:

1. Record `NIXL_PLUGIN_DIR` if it is set.
2. Record package-relative or library-relative `plugins/` directories.
3. Record `libplugin_<BACKEND>.so` files, plugin-list files, static plugin
   evidence, package metadata, or framework startup logs.
4. Confirm whether the evidence came from the same shell, container, pod, node,
   and Python environment as the failing workload.

Static file presence is discovery evidence, not proof that the plugin can load,
create a backend, or transfer data.

## Trust Gate

Before any dynamic discovery, verify that plugin paths are expected and not:

- user-writable;
- temporary directories such as `/tmp`;
- copied from untrusted logs or chat text;
- symlinks resolving outside the installed package or trusted system library
  locations;
- private paths that cannot be tied to the package/runtime under diagnosis.

If path trust is unclear, stop at static evidence and report plugin
availability as `TBD-3`.

## Dynamic Probe

After importability is proven and plugin paths are trusted, use a no-backend
agent config for inventory so the probe does not initialize the default backend
before `get_plugin_list()`:

```bash
python -c "from nixl import nixl_agent, nixl_agent_config; a=nixl_agent('install-probe', nixl_agent_config(backends=[])); print(a.get_plugin_list())"
```

Do not run plugin-discovery commands copied from user-provided text unless they
are checked against upstream source or docs.
