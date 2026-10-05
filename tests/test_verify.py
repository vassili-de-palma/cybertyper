"""Tests for press verification helpers (no actual keyboard listener)."""

import time

from cybertyping.tasks._verify import (
    KeystrokeListener, KeyEvent, _pynput_key_to_label,
)


def test_pynput_key_to_label_letters():
    # We mock pynput key strings (since we don't depend on pynput here)
    assert _pynput_key_to_label("'a'") == "a"
    assert _pynput_key_to_label("'Q'") == "q"
    assert _pynput_key_to_label("' '") == "space"
    assert _pynput_key_to_label("Key.space") == "space"
    assert _pynput_key_to_label("Key.enter") == "enter"


def test_pynput_key_to_label_rejects_unknown():
    # Numbers, punctuation, non-printable specials -> None
    assert _pynput_key_to_label("'1'") is None
    assert _pynput_key_to_label("'.'") is None
    assert _pynput_key_to_label("Key.shift") is None


def test_listener_context_manager_does_not_crash_without_pynput():
    """The listener should silently degrade when pynput is missing or
    blocked by OS permissions. Just exercise the context lifecycle."""
    with KeystrokeListener() as ks:
        # available may be True or False; either way no crash.
        _ = ks.available
        # was_received on an empty queue should not raise.
        assert ks.was_received("a", time.time() - 0.5, window_s=0.1) is False


def test_listener_was_received_window():
    """Inject an event directly into the listener's deque and verify the
    time-window check works."""
    ks = KeystrokeListener()
    t0 = time.time()
    ks._events.append(KeyEvent(timestamp=t0 + 0.1, label="h"))
    ks._available = True
    assert ks.was_received("h", t0, window_s=1.0) is True
    assert ks.was_received("h", t0 + 1.0, window_s=0.05) is False  # too late
    assert ks.was_received("z", t0, window_s=1.0) is False         # wrong label
