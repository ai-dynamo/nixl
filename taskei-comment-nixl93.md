# NIXL-93: Update control messages to WRITE instead of SENDRECV

## Status: In Review (Draft PR)

### Summary
Replaced the `fi_senddata`/`fi_recvmsg` (SEND/RECV) notification path in the libfabric backend with `fi_writedata` (RDMA WRITE with immediate data) to a pre-registered remote notification buffer.

### What changed
- Each rail now allocates a notification receive buffer (1024 × 8KB ring buffer slots) registered with `FI_REMOTE_WRITE`
- Notification buffer address + rkey are exchanged during connection metadata serialization
- `postControlMessage` now uses `fi_writedata` to WRITE notification data into a remote slot, with the slot index derived from `xfer_id % NUM_SLOTS`
- `processRemoteWriteCompletion` handles both `MSG_TRANSFER` (data) and `MSG_NOTIFICTION` (notifications) via immediate data dispatch
- Pre-posted recv pool (1024 `fi_recvmsg` per rail) removed — no longer needed

### Testing
| Test | Cluster | Result |
|------|---------|--------|
| `nixl_example LIBFABRIC` | 2× p5en.48xlarge, 16 EFA rails/node | ✅ PASS |
| `agent_example` basic transfer | 1× p5en.48xlarge | ✅ PASS |
| `agent_example` partialMdTest | 1× p5en.48xlarge | ✅ PASS |
| `agent_example` sideXferTest | 1× p5en.48xlarge | ⏳ Hung (root cause identified — see below) |

### sideXferTest hang analysis
The `sideXferTest` creates 2M descriptors (32 mems × 64K descs) with a notification. The hang occurs because:

1. The notification WRITE (`fi_writedata` on rail 0) returns `-FI_EAGAIN` — the send queue is full
2. The EAGAIN retry loop in `postWrite` skips CQ progress when `progress_thread_enabled_ = true` (line 1233 of `libfabric_rail.cpp`)
3. The progress thread is supposed to drain the CQ, but under heavy load (2M data WRITEs + notification WRITE), it can't keep up
4. Result: the main thread spins forever waiting for CQ space that the progress thread isn't draining fast enough

This is a pre-existing design issue in the EAGAIN retry loops — they should progress the CQ even when the progress thread is enabled. The old SEND path didn't trigger this because recv completions provided natural CQ headroom. The WRITE path adds sender-side CQ pressure that wasn't there before.

**Proposed fix (separate PR):** Always call `progressCompletionQueue()` in EAGAIN retry loops, regardless of `progress_thread_enabled_`. This would fix both the notification WRITE case and any future CQ pressure scenarios.

### Artifacts
- Branch: `anshumgo/nixl-93-control-msg-write` on [github.com/anshumang/nixl](https://github.com/anshumang/nixl/tree/anshumgo/nixl-93-control-msg-write)
- Draft PR: pending creation on ai-dynamo/nixl
- Build dir: `/fsx/anshumgo/nixl-93/` on p5en cluster

### Next steps
- Investigate sideXferTest hang (baseline test without changes in progress)
- Run multi-fragment notification test
- Finalize PR after investigation
