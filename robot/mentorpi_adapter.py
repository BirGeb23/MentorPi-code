"""
============================================================================
mentorpi_adapter.py  —  Translates ReactionAction into MentorPi wheel motion
============================================================================
This adapter follows the same contract as SpotAdapter and the Bittle adapter:

        robot.apply_reaction(ra)

The pipeline hands the adapter a ReactionAction object; the adapter turns
it into wheel-motion commands and publishes them to /cmd_vel. The pipeline
doesn't know or care that the robot is a wheeled MentorPi instead of a Spot
quadruped — that's the point of the adapter pattern.

Design decisions and per-intent motions are documented in:
    MentorPi_Motion_Mapping_Design.md

============================================================================
"""

import json
import threading
import time

from safety import Motion, RobotState, apply_all


# --- Discrete pose table --------------------------------------------------
# Each entry describes the target motion for one intent, relative to the
# person tracker state. These are the design-doc values.
#
#   linear    : desired linear velocity (m/s). + = forward, - = backward
#   angular   : desired angular velocity (rad/s). + = counter-clockwise
#   target_d  : distance from person the intent tries to hold (meters)
#
# The adapter decides what to actually publish by comparing the current
# person distance to target_d and to the safety floor.
INTENT_TABLE = {
    "NEUTRAL":     {"linear":  0.00, "angular": 0.00, "target_d": None},
    "CHECK_IN":    {"linear":  0.08, "angular": 0.30, "target_d": 1.5},
    "ENGAGE":      {"linear":  0.15, "angular": 0.40, "target_d": 1.0},
    "CAUTION":     {"linear":  0.00, "angular": 0.00, "target_d": None},
    "DE_ESCALATE": {"linear": -0.10, "angular": 0.00, "target_d": 2.0},
}

# Face-orient tolerance: if the person's bearing is within this many
# radians of straight ahead, don't bother rotating.
FACE_ALIGN_TOLERANCE = 0.175   # ~10 degrees

# ENGAGE "greet" parameters — a rhythmic forward-back flourish after
# approach. On the MentorPi A1 (Ackermann steering), pure spin-in-place
# is impossible and S-curve wiggles are too subtle to notice. Instead,
# we do a visible forward-back "greet" gesture: robot leans in toward
# the person, pulls back, leans in again, pulls back. Feels like an
# excited "hi!" gesture. Uses pure linear motion so Ackermann runs it.
ENGAGE_GREET_CYCLES      = 2       # forward-back pairs
ENGAGE_GREET_HALF_TIME   = 0.4     # seconds per forward or backward push
ENGAGE_GREET_LINEAR      = 0.15    # m/s during each push
# Net motion per pair: 0 cm (forward 6cm then back 6cm)
# Total duration: cycles × 2 × half_time = 1.6 s

# NEUTRAL "pulse" parameters — subtle back-forward motion after several
# consecutive NEUTRAL cycles to signal the robot is still alive. Slower
# and gentler than the ENGAGE greet. Backward first (gentle pull back),
# then forward return.
NEUTRAL_PULSE_LINEAR     = 0.06    # m/s, slower than GREET for gentler feel
NEUTRAL_PULSE_HALF_TIME  = 0.5     # seconds each direction
# Net motion: 0 cm (back 3cm then forward 3cm). Total: 1.0 s.

# CHECK_IN "acknowledgement" parameters — a single forward-back nod that
# says "I see you". Calmer than the ENGAGE greet (one pair instead of two,
# slower) but faster than the NEUTRAL pulse so it is clearly visible.
# Forward first (leans toward the person), then back.
CHECKIN_ACK_LINEAR       = 0.10    # m/s (MAX_LINEAR_SPEED is 0.25)
CHECKIN_ACK_HALF_TIME    = 0.5     # seconds each direction
# Net motion: 0 cm (forward 5cm then back 5cm). Total: 1.0 s.

# CAUTION "scan" parameters — a watchful evaluation when the robot has
# stayed in CAUTION for multiple cycles. Backs up while turning first
# one way then the other (scanning for changes), then returns forward
# toward original position. More elaborate than the pulse to convey
# "actively watching" vs. NEUTRAL's "passively present".
CAUTION_SCAN_LINEAR      = 0.06    # m/s during motion segments
CAUTION_SCAN_ANGULAR     = 0.30    # rad/s during scan turns
CAUTION_SCAN_SEG_TIME    = 1.0     # seconds per back+turn segment
CAUTION_SCAN_HOLD_TIME   = 0.3     # brief pause between segments
CAUTION_SCAN_RETURN_TIME = 0.4     # seconds for the small forward return
# Total duration: 2×1.0 + 2×0.3 + 0.4 = 3.0 s


class MentorPiAdapter:
    """
    Translates a ReactionAction into a Twist message published to /cmd_vel.
    """

    # ---------------------------------------------------------------
    # 1) CONNECT — set up rosbridge WebSocket (or skip in mock mode)
    # ---------------------------------------------------------------
    def __init__(self, mock=False, rosbridge_host="192.168.149.1",
                 rosbridge_port=9090, topic="/controller/cmd_vel"):
        self.mock  = mock
        self.state = RobotState()
        self._topic = topic

        if mock:
            print("[MENTORPI-MOCK] Mock mode active — no ROS 2 / no robot.")
            self._ws = None
            return

        # Real robot path: talk to rosbridge_websocket via WebSocket.
        # This is a legitimate ROS 2 pattern — the underlying topics
        # ARE ROS 2 topics; rosbridge bridges to them. Same mechanism
        # the mobile app uses.
        try:
            import websocket
        except ImportError:
            raise RuntimeError(
                "websocket-client is required for real motion. "
                "Install with: pip install websocket-client"
            )

        url = f"ws://{rosbridge_host}:{rosbridge_port}"
        print(f"[MENTORPI] Connecting to rosbridge at {url}...")
        self._ws = websocket.create_connection(url, timeout=5)

        # Tell rosbridge we want to publish to this topic
        self._ws.send(json.dumps({
            "op": "advertise",
            "topic": self._topic,
            "type": "geometry_msgs/Twist",
        }))
        print(f"[MENTORPI] Connected. Publishing to {self._topic}.")

    # ---------------------------------------------------------------
    # 2) THE CONTRACT — same signature as SpotAdapter.apply_reaction
    # ---------------------------------------------------------------
    def apply_reaction(self, ra):
        """
        Take a ReactionAction and drive the wheels accordingly.
        `ra` is expected to have fields:  intent, pose_mode, valence, arousal
        """
        # Choose discrete vs continuous motion based on pose_mode
        if ra.pose_mode == "VA_POSE":
            motion = self._va_to_motion(ra.valence, ra.arousal)
        else:
            motion = self._motion_for_intent(ra.intent)

        # Always run the safety layer — never bypass this
        safe = apply_all(motion, self.state)

        # Publish (or print, in mock mode)
        self._publish(safe, ra.intent)

    # ---------------------------------------------------------------
    # 3a) DISCRETE — one motion per intent (uses the table)
    # ---------------------------------------------------------------
    def _motion_for_intent(self, intent: str) -> Motion:
        """
        Build a Motion for one of the 5 discrete intents, adjusting based on
        current person state (distance held, bearing to face, etc.).
        """
        params = INTENT_TABLE.get(intent, INTENT_TABLE["NEUTRAL"])
        linear  = params["linear"]
        angular = params["angular"]
        target  = params["target_d"]

        # Track diagnostic reasons for why motion may be zeroed. Helpful
        # when the robot silently does nothing and it's not obvious why.
        reasons = []

        # Distance-holding logic: only apply linear motion if we're not
        # already at (or past) the target distance for this intent.
        if target is not None and self.state.person_visible:
            d = self.state.person_distance
            if intent == "DE_ESCALATE":
                # Backing away — stop once we've reached target distance.
                if d >= target:
                    linear = 0.0
                    reasons.append(f"at_backoff_distance ({d:.2f}m ≥ "
                                   f"{target:.2f}m)")
            elif intent in ("CHECK_IN", "ENGAGE"):
                # Approaching — stop once we're at or inside target distance.
                if d <= target:
                    linear = 0.0
                    reasons.append(f"at_target_distance ({d:.2f}m ≤ "
                                   f"{target:.2f}m)")

        # Rotation-toward-person logic: only rotate if bearing is outside
        # the alignment cone. We rotate to bring the person TOWARD center.
        #
        # bearing_rad convention (from PersonTracker): positive = person is
        # to the RIGHT of camera center; negative = person is to the LEFT.
        # ROS REP-103: positive angular_z = CCW (turn left); negative = CW.
        # So:
        #   Person on LEFT  (bearing < 0)  → turn LEFT  (angular > 0)
        #   Person on RIGHT (bearing > 0)  → turn RIGHT (angular < 0)
        if angular != 0.0 and self.state.person_visible:
            if abs(self.state.person_bearing) <= FACE_ALIGN_TOLERANCE:
                angular = 0.0
                reasons.append(f"within_face_tolerance "
                               f"({self.state.person_bearing*57.3:+.0f}°)")
            else:
                # Rotate in the direction that brings the person to center.
                # Sign of angular should be OPPOSITE to sign of bearing.
                angular = abs(angular) * (-1 if self.state.person_bearing > 0 else 1)

        # If the person isn't visible, note that too — approach and rotation
        # both require a visible person.
        if not self.state.person_visible and intent != "NEUTRAL" and intent != "CAUTION":
            reasons.append("no_person_visible")

        reason_str = intent + (" | " + " | ".join(reasons) if reasons else "")
        return Motion(linear=linear, angular=angular, reason=reason_str)

    # ---------------------------------------------------------------
    # 3b) CONTINUOUS — VA_POSE maps valence + arousal onto motion
    # ---------------------------------------------------------------
    def _va_to_motion(self, valence: float, arousal: float) -> Motion:
        """
        Continuous mapping used when the pipeline is highly confident.

          valence  →  forward/backward direction and base magnitude
                       +val = approach, -val = retreat
          arousal  →  speed multiplier (higher arousal = faster motion)

        The safety layer still clamps and blocks unsafe commands.
        """
        base_linear = 0.15 * valence
        multiplier  = 0.5 + 0.5 * abs(arousal)
        linear      = base_linear * multiplier

        # Always face the person during VA_POSE
        angular = 0.0
        if self.state.person_visible \
                and abs(self.state.person_bearing) > FACE_ALIGN_TOLERANCE:
            angular = 0.3 if self.state.person_bearing > 0 else -0.3

        return Motion(linear=linear, angular=angular, reason="VA_POSE")

    # ---------------------------------------------------------------
    # 4) PUBLISH — send Twist to rosbridge (or print, in mock)
    # ---------------------------------------------------------------
    def _publish(self, motion: Motion, intent: str):
        if self.mock:
            print(f"[MENTORPI-MOCK] {intent:11s} -> "
                  f"linear={motion.linear:+.3f} m/s  "
                  f"angular={motion.angular:+.3f} rad/s"
                  + (f"   [{motion.reason}]" if motion.reason else ""))
            return

        self._publish_raw(motion)
        print(f"[MENTORPI] {intent:11s} -> "
              f"linear={motion.linear:+.3f} m/s  "
              f"angular={motion.angular:+.3f} rad/s"
              + (f"   [{motion.reason}]" if motion.reason else ""))

    def _publish_raw(self, motion: Motion):
        """Send a Twist to rosbridge with no logging (for hold-rate spamming)."""
        if self.mock or self._ws is None:
            return
        msg = {
            "op": "publish",
            "topic": self._topic,
            "msg": {
                "linear":  {"x": float(motion.linear),  "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": float(motion.angular)},
            },
        }
        self._ws.send(json.dumps(msg))

    # ---------------------------------------------------------------
    # 4b) HOLD — keep publishing the same motion for a duration
    # ---------------------------------------------------------------
    def hold_for(
        self,
        ra,
        duration_s: float,
        rate_hz: float = 20.0,
        stop_event: "threading.Event | None" = None,
    ):
        """
        Apply this reaction and then keep republishing the resulting motion
        at `rate_hz` for `duration_s` total. This is critical for smooth
        motion: the MentorPi motor controller has a watchdog that stops the
        wheels if it doesn't get a fresh Twist every ~200ms. Publishing at
        20Hz keeps the wheels moving continuously for the whole hold window.

        Usage in the runner:
            adapter.hold_for(ra, 2.5)   # move for 2.5 s, then caller sends stop

        stop_event: optional threading.Event. If supplied and set (by another
            thread) before duration_s elapses, this call returns immediately
            instead of finishing out the hold — lets a caller preempt an
            in-progress hold with a fresher decision (e.g. a newer analysis
            window's intent) instead of blocking until the old one finishes.
            Default None preserves the exact original blocking behavior.
        """
        # Compute motion once
        if ra.pose_mode == "VA_POSE":
            motion = self._va_to_motion(ra.valence, ra.arousal)
        else:
            motion = self._motion_for_intent(ra.intent)
        safe = apply_all(motion, self.state)

        # First publish with logging
        self._publish(safe, ra.intent)

        # In mock mode there's nothing to spam-publish — just wait
        if self.mock:
            if stop_event is not None:
                stop_event.wait(duration_s)  # returns early if interrupted
            else:
                time.sleep(duration_s)
            return

        # Keep republishing at rate for the remaining hold duration
        period = 1.0 / rate_hz
        end    = time.time() + duration_s
        while time.time() < end:
            if stop_event is not None and stop_event.is_set():
                return
            self._publish_raw(safe)
            time.sleep(period)

    # ---------------------------------------------------------------
    # 4c) WIGGLE — small side-to-side "hello" flourish (used by ENGAGE)
    # ---------------------------------------------------------------
    def greet(self,
              cycles: int          = ENGAGE_GREET_CYCLES,
              half_time_s: float   = ENGAGE_GREET_HALF_TIME,
              linear_speed: float  = ENGAGE_GREET_LINEAR,
              rate_hz: float       = 20.0):
        """
        Publish a forward-back "greet" gesture for personality/expression.
        The robot leans toward the person (forward), pulls back, repeats.
        Used after ENGAGE to give the robot an excited "hi!" flourish.

        Why this shape: The MentorPi A1 uses Ackermann steering (like a
        car), so it can't spin in place — but it can move forward/backward
        perfectly. This greet uses pure linear motion, making it visibly
        obvious ("robot bounced toward me!") without needing steering.

        Ends with a zero-velocity publish.

        Args:
            cycles: number of forward-back pairs. 2 = "bounce twice".
            half_time_s: how long each forward or back push lasts.
                Longer = bigger gesture, shorter = snappier feel.
            linear_speed: m/s during each push (positive value).
                Combined with half_time_s: each push moves
                linear_speed × half_time_s meters.
            rate_hz: publish rate (20 Hz keeps motor watchdog happy).

        Safety: linear stays under safety.py's max_linear_speed clamp,
        and every command is filtered through apply_all() so E-stop,
        low-battery degradation, and safety-floor blocking all still
        apply. Net displacement per pair is zero (forward = backward).
        """
        total_s = cycles * 2 * half_time_s

        if self.mock:
            print(f"[MENTORPI-MOCK] GREET {cycles}x "
                  f"(±{linear_speed:.2f} m/s, {half_time_s:.1f}s each) "
                  f"total {total_s:.1f}s")
            time.sleep(total_s)
            return

        period = 1.0 / rate_hz
        print(f"[MENTORPI] GREET {cycles}x "
              f"(±{linear_speed:.2f} m/s, {half_time_s:.1f}s each) "
              f"total {total_s:.1f}s")

        for _ in range(cycles):
            # Forward push — "leaning in"
            motion = Motion(linear=+linear_speed, angular=0.0,
                            reason="GREET_FWD")
            safe   = apply_all(motion, self.state)
            end    = time.time() + half_time_s
            while time.time() < end:
                self._publish_raw(safe)
                time.sleep(period)
            # Backward push — "pulling back"
            motion = Motion(linear=-linear_speed, angular=0.0,
                            reason="GREET_BACK")
            safe   = apply_all(motion, self.state)
            end    = time.time() + half_time_s
            while time.time() < end:
                self._publish_raw(safe)
                time.sleep(period)

        # End cleanly with a zero-velocity publish
        self._publish_raw(Motion(linear=0.0, angular=0.0, reason="GREET_END"))

    # ---------------------------------------------------------------
    # 4d) NEUTRAL PULSE — "still alive" motion after several NEUTRAL cycles
    # ---------------------------------------------------------------
    def neutral_pulse(self, rate_hz: float = 20.0):
        """
        Subtle back-forward motion to signal the robot is still present
        and active during long NEUTRAL periods. Called by the runner after
        N consecutive NEUTRAL cycles (threshold configurable there).

        Backward first (gentle pull back), then forward return. Very slow
        and small (~3 cm each way, 1.0 s total) — deliberately less
        expressive than the ENGAGE greet, which stays reserved for
        positive interaction.
        """
        if self.mock:
            print(f"[MENTORPI-MOCK] NEUTRAL PULSE "
                  f"(back+forward @ ±{NEUTRAL_PULSE_LINEAR:.2f} m/s, "
                  f"{2*NEUTRAL_PULSE_HALF_TIME:.1f}s)")
            time.sleep(2 * NEUTRAL_PULSE_HALF_TIME)
            return

        period = 1.0 / rate_hz
        print(f"[MENTORPI] NEUTRAL PULSE "
              f"(back+forward @ ±{NEUTRAL_PULSE_LINEAR:.2f} m/s, "
              f"{2*NEUTRAL_PULSE_HALF_TIME:.1f}s)")

        # Backward push (gentle pull back)
        motion = Motion(linear=-NEUTRAL_PULSE_LINEAR, angular=0.0,
                        reason="PULSE_BACK")
        safe   = apply_all(motion, self.state)
        end    = time.time() + NEUTRAL_PULSE_HALF_TIME
        while time.time() < end:
            self._publish_raw(safe)
            time.sleep(period)
        # Forward return
        motion = Motion(linear=+NEUTRAL_PULSE_LINEAR, angular=0.0,
                        reason="PULSE_FWD")
        safe   = apply_all(motion, self.state)
        end    = time.time() + NEUTRAL_PULSE_HALF_TIME
        while time.time() < end:
            self._publish_raw(safe)
            time.sleep(period)
        # End cleanly
        self._publish_raw(Motion(linear=0.0, angular=0.0, reason="PULSE_END"))

    # ---------------------------------------------------------------
    # 4d-2) CHECK_IN ACKNOWLEDGEMENT — "I see you" forward-back nod
    # ---------------------------------------------------------------
    def checkin_ack(self, stop_event=None, rate_hz: float = 20.0):
        """
        One forward-back nod for CHECK_IN: forward CHECKIN_ACK_HALF_TIME
        seconds, then back the same, at CHECKIN_ACK_LINEAR m/s. Each half
        goes through apply_all, so the safety layer still applies (for
        example, the forward half is zeroed inside MIN_DISTANCE_FLOOR).

        stop_event: optional threading.Event; if set, the nod stops early
        and the wheels are zeroed, so a newer command can take over.
        """
        def stopped():
            return stop_event is not None and stop_event.is_set()

        if self.mock:
            print(f"[MENTORPI-MOCK] CHECK_IN ACK "
                  f"(forward+back @ ±{CHECKIN_ACK_LINEAR:.2f} m/s, "
                  f"{2*CHECKIN_ACK_HALF_TIME:.1f}s)")
            if stop_event is not None:
                stop_event.wait(2 * CHECKIN_ACK_HALF_TIME)
            else:
                time.sleep(2 * CHECKIN_ACK_HALF_TIME)
            return

        period = 1.0 / rate_hz
        print(f"[MENTORPI] CHECK_IN ACK "
              f"(forward+back @ ±{CHECKIN_ACK_LINEAR:.2f} m/s, "
              f"{2*CHECKIN_ACK_HALF_TIME:.1f}s)")
        for linear, reason in ((+CHECKIN_ACK_LINEAR, "ACK_FWD"),
                               (-CHECKIN_ACK_LINEAR, "ACK_BACK")):
            safe = apply_all(Motion(linear=linear, angular=0.0, reason=reason),
                             self.state)
            end = time.time() + CHECKIN_ACK_HALF_TIME
            while time.time() < end and not stopped():
                self._publish_raw(safe)
                time.sleep(period)
            if stopped():
                break
        self._publish_raw(Motion(linear=0.0, angular=0.0, reason="ACK_END"))

    # ---------------------------------------------------------------
    # 4e) CAUTION SCAN — "watchful evaluation" after several CAUTION cycles
    # ---------------------------------------------------------------
    def caution_scan(self, rate_hz: float = 20.0):
        """
        Backward scan gesture for CAUTION when held for multiple cycles.
        Robot backs up while turning first one way, holds, then the other
        way, holds, then makes a small forward return. Conveys "backing up
        to evaluate the situation" — visibly active but non-threatening.

        More elaborate than the NEUTRAL pulse (different intent = different
        motion). Never approaches the person (always backward or
        stationary), so it's safe by construction.
        """
        total_s = (2 * CAUTION_SCAN_SEG_TIME
                   + 2 * CAUTION_SCAN_HOLD_TIME
                   + CAUTION_SCAN_RETURN_TIME)

        if self.mock:
            print(f"[MENTORPI-MOCK] CAUTION SCAN "
                  f"(back-right, hold, back-left, hold, return, "
                  f"~{total_s:.1f}s)")
            time.sleep(total_s)
            return

        period = 1.0 / rate_hz
        print(f"[MENTORPI] CAUTION SCAN "
              f"(back-scan-return, ~{total_s:.1f}s total)")

        def publish_for(linear, angular, duration, reason):
            motion = Motion(linear=linear, angular=angular, reason=reason)
            safe   = apply_all(motion, self.state)
            end    = time.time() + duration
            while time.time() < end:
                self._publish_raw(safe)
                time.sleep(period)

        # Segment 1: backward + turn one way
        publish_for(-CAUTION_SCAN_LINEAR, -CAUTION_SCAN_ANGULAR,
                    CAUTION_SCAN_SEG_TIME, "SCAN_BACK_R")
        # Brief hold
        publish_for(0.0, 0.0, CAUTION_SCAN_HOLD_TIME, "SCAN_HOLD_1")
        # Segment 2: backward + turn the other way
        publish_for(-CAUTION_SCAN_LINEAR, +CAUTION_SCAN_ANGULAR,
                    CAUTION_SCAN_SEG_TIME, "SCAN_BACK_L")
        # Brief hold
        publish_for(0.0, 0.0, CAUTION_SCAN_HOLD_TIME, "SCAN_HOLD_2")
        # Small forward return (partly recover the ground given up)
        publish_for(+CAUTION_SCAN_LINEAR, 0.0,
                    CAUTION_SCAN_RETURN_TIME, "SCAN_RETURN")
        # End cleanly
        self._publish_raw(Motion(linear=0.0, angular=0.0, reason="SCAN_END"))

    # ---------------------------------------------------------------
    # 5) SHUTDOWN — stop the wheels cleanly
    # ---------------------------------------------------------------
    def shutdown(self):
        """Stop motion and close the rosbridge connection."""
        if self.mock:
            print("[MENTORPI-MOCK] Shutdown — wheels would be stopped.")
            return

        zero_msg = {
            "op": "publish",
            "topic": self._topic,
            "msg": {
                "linear":  {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        }
        for _ in range(5):
            try:
                self._ws.send(json.dumps(zero_msg))
            except Exception:
                break

        try:
            self._ws.send(json.dumps({"op": "unadvertise", "topic": self._topic}))
            self._ws.close()
        except Exception:
            pass
        print("[MENTORPI] Wheels stopped, rosbridge disconnected.")
