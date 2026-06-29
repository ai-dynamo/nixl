# Notifications

Use this reference for transfer notifications, manual notifications,
notification polling, and payload matching.

## Source Anchors

Fallback snapshot `b293d9bf2d192b321ee24b1988cf1b6b51875331`:

- `src/api/cpp/nixl.h`: `getNotifs()`, `genNotif()`, transfer APIs accepting
  notification optional args.
- `src/api/cpp/nixl_types.h`: `nixl_notifs_t`, `nixl_opt_args_t.notif`,
  legacy `notifMsg` and `hasNotif`.
- `docs/BackendGuide.md`: backend notification capabilities, merged
  notification polling, manual notification semantics, and no ordering
  guarantee for standalone notifications.
- `examples/cpp/nixl_example.cpp`: transfer notification example.

Confirm against the user's installed headers before copying code.

## Transfer Notifications

The fallback `nixl_opt_args_t.notif` is the preferred notification field for
new C++ API users. It can be attached when creating or posting a transfer:

```cpp
nixl_opt_args_t args;
args.backends.push_back(backend);
args.notif = nixl_blob_t("request-1234-complete");
```

Use unique, application-generated payloads. Do not rely on notification content
as proof of peer identity or authorization.

## Polling Notifications

`getNotifs()` appends entries into a `nixl_notifs_t`, a map from remote agent
name to notification payload list in the fallback headers.

```cpp
nixl_notifs_t notifs;
nixl_status_t st = agent.getNotifs(notifs, &args);
if (st != NIXL_SUCCESS) {
    return st;
}

auto it = notifs.find(remote_agent);
if (it != notifs.end()) {
    for (const nixl_blob_t &msg : it->second) {
        if (msg == expected_payload) {
            // Matched exactly.
        }
    }
}
```

Prefer exact byte/string equality in application code when exact matching is
required. Do not infer exact matching from backend ordering or prefix behavior
unless installed source proves it.

## Manual Notifications

`genNotif(remote_agent, msg, &args)` generates a notification that is not bound
to a transfer in the fallback API. Remote metadata must already be available.
Backend support is required; if no backend is specified, backend selection is
source/backend-sensitive.

## Review Findings To Raise

- A notification is not authentication.
- A manual notification is not a transfer completion proof.
- Missing notifications may mean backend notification support is absent, remote
  metadata is missing, payload matching is wrong, progress is not being driven,
  or the wrong backend handle was constrained.
- Do not promise notification ordering across requests without installed
  source/backend evidence.
