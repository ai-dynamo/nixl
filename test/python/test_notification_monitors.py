# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import pickle
import select
import signal
import sys
import threading
import time
import uuid
import weakref

import pytest

import nixl._bindings as bindings
from nixl import (
    nixl_agent,
    nixl_agent_config,
    nixl_notification_monitor,
    nixlNotificationOverflowError,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="requires Linux eventfd"
)

WAIT_MS = 5_000


@pytest.fixture
def ucx_peers():
    config = nixl_agent_config(enable_prog_thread=True, backends=["UCX"])
    source = nixl_agent(f"source-{uuid.uuid4()}", config)
    if "UCX" not in source.backends:
        pytest.skip("UCX is unavailable")
    target = nixl_agent(f"target-{uuid.uuid4()}", config)
    yield source, target


def _legacy_notifications(target, source, count):
    received = []
    deadline = time.monotonic() + WAIT_MS / 1000
    while len(received) < count and time.monotonic() < deadline:
        received.extend(target.get_new_notifs().get(source.name, []))
        if len(received) < count:
            time.sleep(0.001)
    return received


def _ucx_target_without_backends(enable_prog_thread):
    target = nixl_agent(
        f"target-{uuid.uuid4()}",
        nixl_agent_config(enable_prog_thread=enable_prog_thread, backends=[]),
    )
    if "UCX" not in target.get_plugin_list():
        pytest.skip("UCX is unavailable")
    return target


def test_add_notif_monitor_before_backend_creation():
    target = _ucx_target_without_backends(enable_prog_thread=True)
    source = nixl_agent(f"source-{uuid.uuid4()}")

    with nixl_notification_monitor() as monitor:
        target.add_notif_monitor(monitor, b"watch:")
        target.create_backend("UCX")
        source.add_remote_agent(target.get_agent_metadata())
        source.send_notif(target.name, b"watch:ready")

        assert monitor.wait_any(timeout_ms=WAIT_MS) == [
            (target.name, source.name, b"watch:ready")
        ]


def test_add_notif_monitor_with_application_driven_progress():
    target = _ucx_target_without_backends(enable_prog_thread=False)
    target.create_backend("UCX")
    source = nixl_agent(f"source-{uuid.uuid4()}")

    with nixl_notification_monitor() as monitor:
        target.add_notif_monitor(monitor, b"watch:")
        source.add_remote_agent(target.get_agent_metadata())
        source.send_notif(target.name, b"watch:ready")

        # Reading the monitor does not drive UCX. The separate source agent
        # can make send progress, but the target has no progress thread.
        assert monitor.wait_any(timeout_ms=0) == []
        received = []
        deadline = time.monotonic() + WAIT_MS / 1000
        while not received and time.monotonic() < deadline:
            assert target.get_new_notifs() == {}
            received = monitor.wait_any(timeout_ms=0)
            if not received:
                time.sleep(0.001)

        assert received == [(target.name, source.name, b"watch:ready")]


def test_application_driven_progress_returns_redirects_before_overflow_error():
    target = _ucx_target_without_backends(enable_prog_thread=False)
    target.create_backend("UCX")
    source = nixl_agent(f"source-{uuid.uuid4()}")
    capacity, extra = 2, 1
    payloads = [b"flood:%d" % i for i in range(capacity + extra)]

    with nixl_notification_monitor(max_queued=capacity) as monitor:
        target.add_notif_monitor(monitor, b"flood:")
        source.add_remote_agent(target.get_agent_metadata())
        for message in payloads:
            source.send_notif(target.name, message)

        # The get_new_notifs() call that drives progress returns the redirect
        # before wait_any() reports the overflow.
        redirected = _legacy_notifications(target, source, extra)
        assert len(redirected) == extra
        assert monitor.overflow_count() == extra

        with pytest.raises(nixlNotificationOverflowError) as raised:
            monitor.wait_any(timeout_ms=0)
        assert raised.value.redirected == extra
        queued = [message for _, _, message in raised.value.notifications]
        assert sorted(redirected + queued) == sorted(payloads)


def test_notification_fd_and_legacy_fallback(ucx_peers):
    source, target = ucx_peers
    source.add_remote_agent(target.get_agent_metadata())

    with nixl_notification_monitor() as monitor:
        target.add_notif_monitor(monitor, b"watch:\x00")
        fd = monitor.fileno()
        assert monitor.fd() == fd
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        try:
            source.send_notif(target.name, b"watch:\x00\xff")
            assert poller.poll(WAIT_MS) == [(fd, select.POLLIN)]
            assert monitor.wait_any(timeout_ms=0) == [
                (target.name, source.name, b"watch:\x00\xff")
            ]
            assert poller.poll(0) == []

            source.send_notif(target.name, b"other")
            assert _legacy_notifications(target, source, 1) == [b"other"]
            assert monitor.wait_any(timeout_ms=0) == []
        finally:
            poller.unregister(fd)

    source.send_notif(target.name, b"watch:\x00after-close")
    assert _legacy_notifications(target, source, 1) == [b"watch:\x00after-close"]


def test_notification_monitor_collection_unregisters_all_agents(ucx_peers):
    source, target = ucx_peers
    source.add_remote_agent(target.get_agent_metadata())
    monitor = nixl_notification_monitor()
    for agent in (source, target):
        agent.add_notif_monitor(monitor, b"watch:")
        agent.add_notif_monitor(monitor, b"second:")

    del monitor
    gc.collect()

    source.send_notif(target.name, b"watch:after-collection")
    assert _legacy_notifications(target, source, 1) == [b"watch:after-collection"]

    with nixl_notification_monitor() as replacement:
        for agent in (source, target):
            agent.add_notif_monitor(replacement, b"watch:")
            agent.add_notif_monitor(replacement, b"second:")
        source.send_notif(target.name, b"second:replacement")
        assert replacement.wait_any(timeout_ms=WAIT_MS) == [
            (target.name, source.name, b"second:replacement")
        ]


def test_remove_notif_monitor_only_removes_owned_prefix(ucx_peers):
    source, target = ucx_peers
    source.add_remote_agent(target.get_agent_metadata())

    with nixl_notification_monitor() as monitor, nixl_notification_monitor() as other:
        source.add_notif_monitor(monitor, b"watch:")
        target.add_notif_monitor(monitor, b"watch:")
        target.add_notif_monitor(monitor, b"keep:")

        assert target.remove_notif_monitor(other, b"watch:") is False
        assert target.remove_notif_monitor(monitor, b"missing:") is False
        assert target.remove_notif_monitor(monitor, b"watch:") is True
        assert target.remove_notif_monitor(monitor, b"watch:") is False
        assert source.remove_notif_monitor(monitor, b"watch:") is True

        source.send_notif(target.name, b"watch:removed")
        assert _legacy_notifications(target, source, 1) == [b"watch:removed"]
        source.send_notif(target.name, b"keep:registered")
        assert monitor.wait_any(timeout_ms=WAIT_MS) == [
            (target.name, source.name, b"keep:registered")
        ]


def test_conflicting_subscription_requires_removal(ucx_peers):
    source, target = ucx_peers
    source.add_remote_agent(target.get_agent_metadata())

    with nixl_notification_monitor() as first, nixl_notification_monitor() as second:
        target.add_notif_monitor(first, b"")
        target.add_notif_monitor(first, b"kv:")
        for monitor, prefix in ((first, b""), (second, b""), (second, b"kv:layer:")):
            with pytest.raises(bindings.nixlInvalidParamError):
                target.add_notif_monitor(monitor, prefix)

        source.send_notif(target.name, b"default")
        assert first.wait_any(timeout_ms=WAIT_MS) == [
            (target.name, source.name, b"default")
        ]

        assert target.remove_notif_monitor(second, b"") is False
        assert target.remove_notif_monitor(first, b"") is True
        assert target.remove_notif_monitor(first, b"") is False
        target.add_notif_monitor(second, b"")
        source.send_notif(target.name, b"moved")
        assert second.wait_any(timeout_ms=WAIT_MS) == [
            (target.name, source.name, b"moved")
        ]
        assert first.wait_any(timeout_ms=0) == []

        assert target.remove_notif_monitor(second, b"") is True
        source.send_notif(target.name, b"legacy fallback")
        assert _legacy_notifications(target, source, 1) == [b"legacy fallback"]


def test_notification_overflow_error_pickle_round_trip():
    notifications = [("target", "source", b"watch:\x00\xff")]
    error = nixlNotificationOverflowError(notifications, 3)

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is type(error)
    assert restored.notifications == notifications
    assert restored.redirected == 3
    assert str(restored) == str(error)


def test_full_queue_reports_each_overflow_and_preserves_notifications(ucx_peers):
    source, target = ucx_peers
    source.add_remote_agent(target.get_agent_metadata())
    capacity = 16
    total_redirected = 0
    with nixl_notification_monitor(max_queued=capacity) as monitor:
        target.add_notif_monitor(monitor, b"flood:")
        for wave, extra in enumerate((3, 5)):
            payloads = [b"flood:%d:%d" % (wave, i) for i in range(capacity + extra)]
            for message in payloads:
                source.send_notif(target.name, message)

            total_redirected += extra
            deadline = time.monotonic() + WAIT_MS / 1000
            while (
                monitor.overflow_count() < total_redirected
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            assert monitor.overflow_count() == total_redirected

            with pytest.raises(nixlNotificationOverflowError) as raised:
                monitor.wait_any(timeout_ms=0)
            assert raised.value.redirected == extra
            queued = raised.value.notifications
            assert len(queued) == capacity
            assert all(
                local == target.name and remote == source.name
                for local, remote, _ in queued
            )

            received = _legacy_notifications(target, source, extra)
            assert len(received) == extra
            assert sorted(received + [message for _, _, message in queued]) == sorted(
                payloads
            )
            assert monitor.wait_any(timeout_ms=0) == []
            assert monitor.overflow_count() == total_redirected


def test_agent_destruction_keeps_monitor_open_and_quiet():
    agent = nixl_agent(
        f"receiver-{uuid.uuid4()}",
        nixl_agent_config(enable_prog_thread=True, backends=["UCX"]),
    )
    if "UCX" not in agent.backends:
        pytest.skip("UCX is unavailable")

    with nixl_notification_monitor() as monitor:
        agent.add_notif_monitor(monitor, b"watch:")
        fd = monitor.fileno()
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        try:
            agent_ref = weakref.ref(agent)
            del agent
            gc.collect()
            assert agent_ref() is None
            assert poller.poll(0) == []
            assert monitor.wait_any(timeout_ms=20) == []
        finally:
            poller.unregister(fd)


@pytest.mark.parametrize("timeout_ms", [0, 20])
def test_notification_wait_clears_stale_eventfd_wakeup(timeout_ms):
    with nixl_notification_monitor() as monitor:
        fd = monitor.fileno()
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        try:
            os.eventfd_write(fd, 1)
            assert poller.poll(0) == [(fd, select.POLLIN)]
            assert monitor.wait_any(timeout_ms=timeout_ms) == []
            assert poller.poll(0) == []
        finally:
            poller.unregister(fd)


def test_max_queued_must_be_positive():
    with pytest.raises(ValueError):
        nixl_notification_monitor(max_queued=0)


def test_close_releases_eventfd():
    with nixl_notification_monitor() as monitor:
        fd = monitor.fileno()
        os.fstat(fd)

    for operation in (monitor.fileno, monitor.overflow_count, monitor.wait_any):
        with pytest.raises(RuntimeError):
            operation()
    with pytest.raises(OSError):
        os.fstat(fd)
    monitor.close()


def test_notification_monitor_close_wakes_waiter():
    with nixl_notification_monitor() as monitor:
        entering_wait = threading.Event()
        result = []

        def wait():
            entering_wait.set()
            try:
                result.append(monitor.wait_any(timeout_ms=WAIT_MS))
            except Exception as exc:
                result.append(exc)

        waiter = threading.Thread(target=wait, daemon=True)
        waiter.start()
        assert entering_wait.wait(timeout=1)
        # Let the waiter block in poll before checking that close wakes it.
        waiter.join(timeout=0.05)
        assert waiter.is_alive()
        monitor.close()
        waiter.join(timeout=1)
        assert not waiter.is_alive()
        assert len(result) == 1 and isinstance(result[0], RuntimeError)


def test_notification_wait_runs_python_signal_handler():
    def interrupt(signum, frame):
        raise InterruptedError("monitor wait was interrupted")

    with nixl_notification_monitor() as monitor:
        handler = signal.signal(signal.SIGALRM, interrupt)
        start = time.monotonic()
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.02)
        try:
            with pytest.raises(InterruptedError, match="monitor wait was interrupted"):
                monitor.wait_any(timeout_ms=WAIT_MS)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        assert time.monotonic() - start < WAIT_MS / 1000 / 2
