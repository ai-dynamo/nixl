# Transfers, Polling, And Cleanup

Use this reference for transfer descriptor lists, request creation, prepared
descriptor handles, posting, bounded polling, telemetry, request release, and
cleanup ownership.

## Quick Index

- `Preconditions`: evidence required before emitting transfer code.
- `Direct Transfer Pattern`: one-off request creation.
- `Prepared Descriptor Pattern`: reusable descriptor-list request creation.
- `Bounded Polling`: timeout-aware status loop.
- `Release And Cleanup`: request and descriptor-handle ownership.
- `Review Hazards`: mistakes that should block or change the answer.

## Source Anchors

Fallback snapshot `b293d9bf2d192b321ee24b1988cf1b6b51875331`:

- `src/api/cpp/nixl.h`: `prepXferDlist()`, `makeXferReq()`,
  `createXferReq()`, `estimateXferCost()`, `postXferReq()`,
  `getXferStatus()`, `getXferTelemetry()`, `queryXferBackend()`,
  `releaseXferReq()`, `releasedDlistH()`.
- `src/api/cpp/nixl_types.h`: `NIXL_READ`, `NIXL_WRITE`, `NIXL_IN_PROG`,
  `NIXL_SUCCESS`, error statuses, cost and telemetry types.
- `docs/nixl.md`: transfer sequence and teardown.
- `docs/BackendGuide.md`: transfer request preparation, async post/status,
  repost caveats, release/cancellation expectations, and no ordering guarantee.
- `examples/cpp/nixl_example.cpp`: basic `createXferReq()` lifecycle.
- `test/nixl/agent_example.cpp`: prepared descriptor-list lifecycle.

Confirm against the user's installed headers before copying code.

## Preconditions

Do not create a transfer recipe until:

- Source/build evidence is known or the answer is explicitly fallback-only.
- Selected backend and required memory types are proven.
- Local and remote descriptors refer to registered memory ranges.
- Remote metadata is loaded on the initiator.
- Buffer and descriptor lifetimes exceed the transfer and cleanup sequence.

## Direct Transfer Pattern

Use `createXferReq()` for simple one-off requests:

```cpp
nixl_xfer_dlist_t src(DRAM_SEG);
nixlBasicDesc src_desc(reinterpret_cast<uintptr_t>(src_buf), bytes, 0);
src.addDesc(src_desc);

nixl_xfer_dlist_t dst(DRAM_SEG);
nixlBasicDesc dst_desc(reinterpret_cast<uintptr_t>(dst_buf), bytes, 0);
dst.addDesc(dst_desc);

nixl_opt_args_t args;
args.backends.push_back(backend);
args.notif = nixl_blob_t("done");

nixlXferReqH *req = nullptr;
nixl_status_t st =
    agent.createXferReq(NIXL_WRITE, src, dst, remote_agent, req, &args);
if (st != NIXL_SUCCESS) {
    return st;
}
```

## Prepared Descriptor Pattern

Use `prepXferDlist()` plus `makeXferReq()` when descriptor preparation is reused
across multiple transfer requests:

- Prepare local descriptors with the overload that omits `agent_name`, or with
  `NIXL_INIT_AGENT` if using the explicit overload.
- Prepare remote descriptors with the remote agent name.
- Pass matching index vectors to `makeXferReq()`.
- Release prepared descriptor handles with `releasedDlistH()` after all request
  handles that use them have been released.

## Bounded Polling

Avoid tight unbounded loops. The fallback status model uses `NIXL_SUCCESS` for
completion and `NIXL_IN_PROG` for active transfers.

```cpp
nixl_status_t st = agent.postXferReq(req);
if (st < NIXL_SUCCESS) {
    agent.releaseXferReq(req);
    return st;
}

const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
while (st == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    st = agent.getXferStatus(req);
}

if (st == NIXL_IN_PROG) {
    // Collect backend logs/status and source evidence before changing retry policy.
}
```

## Cleanup Ownership

- Release each `nixlXferReqH*` with `releaseXferReq()` once the request is no
  longer needed.
- Release prepared descriptor handles with `releasedDlistH()`.
- Deregister only memory registered by the current process/agent.
- Invalidate remote metadata on the initiator when that remote metadata is no
  longer valid or when topology teardown requires it.
- Free application buffers only after transfers are complete, request handles
  are released, and memory is deregistered.

## Repost And Cancellation Caution

The fallback docs state a transfer request can be posted more than once only
after the prior post is complete, and that release/cancellation behavior can be
backend-sensitive. Do not promise production retry, cancellation, abort latency,
or ordering semantics without installed-source/backend evidence.
