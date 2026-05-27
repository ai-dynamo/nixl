# Transfers, Polling, And Cleanup

Use this reference for transfer request creation, bounded polling, prepared
descriptor-list transfers, transfer handle release, and cleanup ordering.

## Quick Index

- `Bounded Wait Helper`: reusable timeout wrapper for transfer polling.
- `One-Off Transfer`: request creation when descriptor lists are known.
- `Prepared Descriptor Lists`: reusable descriptor-list transfer pattern.
- `Reposting And Active Handles`: active-handle and repost safety.
- `Cleanup`: release and ownership rules for same-process and multi-process
  cleanup.

## Bounded Wait Helper

Use bounded polling in examples and patches:

```python
import time

def wait_until(predicate, label, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {label}")
        time.sleep(0.01)

def wait_for_xfer(agent, handle, initial_state, label="transfer completion"):
    state = initial_state

    def complete():
        nonlocal state
        if state != "PROC":
            return True
        state = agent.check_xfer_state(handle)
        return state != "PROC"

    wait_until(complete, label)
    return state
```

## One-Off Transfer

Use this when descriptor lists are known at request time:

```python
handle = None
try:
    handle = initiator.initialize_xfer(
        operation="READ",
        local_descs=local_descs,
        remote_descs=target_descs,
        remote_agent="target",
        notif_msg=b"read-0",
    )

    state = wait_for_xfer(initiator, handle, initiator.transfer(handle))

    if state != "DONE":
        raise RuntimeError(f"NIXL transfer ended in {state}")
finally:
    if handle is not None:
        initiator.release_xfer_handle(handle)
```

The high-level wrapper documents `DONE`, `PROC`, and `ERR`. In the fallback
snapshot, the pybind layer throws typed exceptions for negative statuses before
the wrapper can return for those statuses. Keep normal Python exception
handling around transfer calls and do not rely only on returned `"ERR"`.

## Prepared Descriptor Lists

Use prepared descriptor lists when the same descriptor lists are reused and
indices select which descriptors participate in each transfer:

```python
local_side = None
remote_side = None
handle = None
try:
    local_side = initiator.prep_xfer_dlist("", local_descs)
    remote_side = initiator.prep_xfer_dlist("target", target_descs)

    handle = initiator.make_prepped_xfer(
        operation="READ",
        local_xfer_side=local_side,
        local_indices=[0, 1, 2],
        remote_xfer_side=remote_side,
        remote_indices=[0, 1, 2],
        notif_msg=b"read-blocks-0-2",
    )

    state = wait_for_xfer(initiator, handle, initiator.transfer(handle))
    if state != "DONE":
        raise RuntimeError(f"NIXL transfer ended in {state}")
finally:
    if handle is not None:
        initiator.release_xfer_handle(handle)
    if local_side is not None:
        initiator.release_dlist_handle(local_side)
    if remote_side is not None:
        initiator.release_dlist_handle(remote_side)
```

In the fallback snapshot, an empty agent name denotes the local side for
`prep_xfer_dlist`. Verify this against the user's installed source before
copy-pasting if the version differs.

## Reposting And Active Handles

Only repost a handle after the previous post has completed unless the user's
installed source explicitly supports the desired repost pattern. Active repost,
retry, and cancellation behavior is version- and backend-specific; when not
verified, state the missing installed-source/backend evidence directly.

## Cleanup

Prefer this conservative cleanup order after transfer completion. Verify the
installed source/examples when cleanup ordering is operationally important.

1. Release transfer handles.
2. Release prepared descriptor-list handles.
3. Remove remote agents or invalidate listener metadata for peers owned by the
   current process.
4. Deregister memory owned by the current process.

Same-process direct cleanup where the code owns both agents and registrations:

```python
initiator.remove_remote_agent("target")
initiator.deregister_memory(init_reg)
target.deregister_memory(target_reg)
```

If listener metadata was used, invalidate only metadata owned by the current
process and define the listener endpoint from trusted topology state:

```python
initiator.invalidate_local_metadata(target_ip, target_port)
```

Two-process or two-host example where the initiator only owns its local agent:

```python
initiator.remove_remote_agent("target")
initiator.invalidate_local_metadata(target_ip, target_port)
initiator.deregister_memory(init_reg)
```

The target process should release its own handles and deregister its own memory
after its side observes completion or shutdown. Do not imply that one process
can clean up another process's registered memory unless the user's installed
source and control-plane ownership prove that topology.

If cleanup fails after a partial transfer failure, report the exact exception
and installed source version. Do not mask cleanup failures with broad
`except: pass` blocks.
