# NIXL KV Backend Interfaces

The KV plugin layer follows the same facade/implementation split as the object backend:

- `nixlKVEngine` is the NIXL-facing backend engine wrapper.
- `nixlKVEngineImpl` is the implementation interface for concrete KV backends.
- `iKVStore` is the small synchronous store interface used by simple KV backends.

Redis provides its async client and concrete implementation under `redis/`.
