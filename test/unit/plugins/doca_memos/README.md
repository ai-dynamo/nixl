# DOCA_MEMOS Backend Unit Tests

These tests build the DOCA_MEMOS plugin sources against the real DOCA KV
headers but replace every DOCA function with a mock from
`doca_memos_mocks.cpp`, so they run without DOCA hardware or the DOCA
libraries.

`DocaMockControl` configures the mocks: per-function return values, task
allocation failures after N calls, an in-memory key-value store, and
`auto_complete_tasks`, which completes submitted tasks inside
`doca_pe_progress()`. With the progress thread enabled, the plugin calls the
mocks from that thread, so tests configure the mock before `createEngine()`
and read it afterwards under `DocaMockControl::lock()`.

## Running

```bash
meson test -C build --suite doca_memos
./build/test/unit/plugins/doca_memos/doca_memos_backend_test --gtest_filter='*Query*'
```
