"""Intent execution and QR-target state for MentorPi motion."""

from dataclasses import dataclass
import os
import sys
import threading
import time


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "robot"))
from mentorpi_adapter import MentorPiAdapter


@dataclass
class ReactionAction:
    intent: str
    pose_mode: str = "REACTION_POSE"
    valence: float = 0.0
    arousal: float = 0.0


class RobotMotionController:
    """Accept QR target updates and execute gesture intents through the adapter."""

    def __init__(self, mock=True):
        self.adapter = MentorPiAdapter(mock=mock)
        self.adapter.state.person_visible = False
        self.adapter.state.person_distance = 3.0
        self.adapter.state.person_bearing = 0.0
        self.adapter.state.missing_windows = 3
        self.mock = mock
        self.current_intent = "NEUTRAL"
        self._last_target_time = 0.0
        self._motion_lock = threading.Lock()
        self._stop_event = None
        self._thread = None

    def update_target(self, target, now=None):
        """Update adapter target state from QR tracker output, or age it out."""
        now = time.monotonic() if now is None else now
        if target and target.get("visible"):
            self._last_target_time = now
            self.adapter.state.person_visible = True
            self.adapter.state.person_distance = float(target["distance_m"])
            self.adapter.state.person_bearing = float(target["bearing_rad"])
            self.adapter.state.missing_windows = 0
            return

        age = now - self._last_target_time
        if age > 2.0:
            self.adapter.state.person_visible = False
        self.adapter.state.missing_windows = max(0, int(age))

    def dispatch_intent(self, intent, extra=None):
        """Interrupt the previous intent and begin the newest one asynchronously."""
        with self._motion_lock:
            previous_thread = self._thread
            if self._stop_event is not None:
                self._stop_event.set()
            stop_event = threading.Event()
            worker = threading.Thread(
                target=self._run_intent,
                args=(intent, extra, stop_event, previous_thread),
                daemon=True,
                name=f"Motion-{intent}",
            )
            self._stop_event = stop_event
            self._thread = worker
            self.current_intent = intent
            worker.start()

    def _run_intent(self, intent, extra, stop_event, previous_thread):
        # Bounded join lets a currently running scan finish before publishing
        # another command; the scan itself lasts about three seconds.
        if previous_thread is not None:
            previous_thread.join(timeout=5.0)
        if stop_event.is_set():
            return

        if extra == "ACK":
            self.adapter.checkin_ack(stop_event=stop_event)
            if stop_event.is_set():
                return
        elif extra == "SCAN":
            self.adapter.caution_scan()
        elif extra == "PULSE":
            self.adapter.neutral_pulse()
            return

        action = ReactionAction(intent)
        if intent == "NEUTRAL":
            self.adapter.hold_for(action, 0.25, stop_event=stop_event)
            return

        # Recompute motion from the latest QR bearing/range while the intent
        # is active, so the robot can keep correcting toward the marker.
        end = time.monotonic() + 3.0
        while not stop_event.is_set():
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            self.adapter.hold_for(action, min(0.25, remaining), stop_event=stop_event)
        if not stop_event.is_set():
            self.adapter.hold_for(ReactionAction("NEUTRAL"), 0.25)

    def close(self):
        with self._motion_lock:
            if self._stop_event is not None:
                self._stop_event.set()
            worker = self._thread
        if worker is not None:
            worker.join(timeout=5.0)
        self.adapter.shutdown()
