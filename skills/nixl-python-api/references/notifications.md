# Notifications

Use this reference for transfer completion notifications, manual notifications,
and notification matching semantics.

## Rules

- Notifications are bytes and should be treated as completion hints, not
  authentication.
- Use unique tags per transfer when several transfers may complete close
  together.
- Do not accumulate unbounded notification state from untrusted peers. Consume
  the expected message or apply an application-level bound.
- Verify backend notification support for the selected backend from installed
  source/runtime evidence before relying on notifications.

## Transfer Completion Notification

```python
done = target.check_remote_xfer_done(
    "initiator",
    b"read-0",
    tag_is_prefix=True,
)
```

In the fallback snapshot:

- `tag_is_prefix=True` checks `msg.startswith(lookup_tag)`.
- `tag_is_prefix=False` checks `lookup_tag in msg`.

`tag_is_prefix=False` is substring matching, not exact matching. If exact
matching is required, poll notifications manually and compare bytes exactly:

```python
notif_map = target.get_new_notifs()
for msg in notif_map.get("initiator", []):
    if msg == b"read-0":
        break
```

## Manual Notifications

```python
initiator.send_notif("target", b"control-message")
notif_map = target.get_new_notifs()
```

If notifications do not arrive, verify:

- Remote metadata is loaded.
- Agent names match.
- The selected backend supports notifications.
- The application is polling the same backend path used for the transfer.
- The expected tag is unique and the prefix/substr semantics are intentional.
