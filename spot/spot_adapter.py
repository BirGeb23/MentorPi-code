#!/usr/bin/env python3
"""
============================================================================
spot_adapter.py  —  Turns intents into Spot movement (Spot SDK)
============================================================================
Implements the movements in "Proposed Spot Movements for Gesture-Driven
Intents" (team Google Doc). Same interface as MentorPiAdapter, so the
gesture logic can drive either robot:

    adapter.state                          person_visible / distance / bearing
    adapter.hold_for(ra, seconds, stop_event=...)
    adapter.checkin_ack(stop_event=...)    one nod
    adapter.caution_scan(stop_event=...)   body turns left/right, no steps
    adapter.neutral_pulse(stop_event=...)  small body bob
    adapter.stop()                         stop walking, hold current posture
    adapter.shutdown()                     sit, power off, release lease

Per intent (doc values):
    NEUTRAL      normal stand, no walking
    CHECK_IN     +2 cm, ~3 deg forward tilt; turn in place to face the person;
                 no walking (distance kept)
    ENGAGE       +8 cm, ~6 deg forward lean; turn to face, then walk forward
                 <= 0.3 m/s to 80% of the streak's starting distance, never
                 closer than 1.5 m
    CAUTION      neutral still stance, no walking (optional 0.2 m step back
                 is OFF)
    DE_ESCALATE  -8 cm, ~6 deg lean back; walk backward <= 0.2 m/s for
                 0.45 m per streak

Safety (doc table): forward 0.3 m/s, backward 0.2 m/s, turn 0.3 rad/s,
never walk forward inside 1.2 m, body height within +/-0.12 m, tilt within
+/-0.20 rad. No walking or turning unless a person is currently visible.
Walking/turning use short velocity commands (0.6 s) re-sent ~10x per
second: if this program stops, Spot stops. Spot's obstacle avoidance stays on.

Testing switches:
    mock=True            print the commands, never connect to Spot
    allow_walking=False  no forward/backward walking (postures + turning only)
    allow_turning=False  no turning in place
    speed_scale=0.5      half of every speed limit (stage 3 of the test plan)

E-stop: this adapter does NOT register its own E-stop. The E-stop operator
runs a separate E-stop (the Spot tablet, or the SDK's estop example on
another laptop) so it stays independent of this program. start() refuses
to power on unless an E-stop is configured and not triggered.

UNVERIFIED ON THE ROBOT (check in Stage 2 before anything else):
    PITCH_FORWARD_SIGN: whether positive pitch tilts Spot's nose DOWN. Do a
    single nod first; if Spot tilts its nose UP, set it to -1.
============================================================================
"""

import contextlib
import math
import threading
import time
from dataclasses import dataclass

# ---------- Speeds and distances (doc "Safety limits") ----------
MAX_FORWARD_MPS   = 0.30
MAX_BACKWARD_MPS  = 0.20
MAX_TURN_RADPS    = 0.30
TURN_GAIN         = 1.0     # v_rot = TURN_GAIN * bearing error, then clamped
FACE_TOLERANCE    = math.radians(10)
ENGAGE_FRACTION   = 0.8     # stop at 80% of the starting distance
ENGAGE_MIN_DIST   = 1.5     # never closer than this for ENGAGE
FORWARD_FLOOR     = 1.2     # never walk forward inside this
RETREAT_M         = 0.45    # DE_ESCALATE retreat per streak
MAX_HEIGHT        = 0.12
MAX_TILT          = 0.20
MAX_YAW           = 0.30

VEL_CMD_DURATION_S = 0.6    # each velocity command expires after this
CONTROL_RATE_HZ    = 10.0

# +1: positive pitch = nose down (lean forward). VERIFY in Stage 2.
PITCH_FORWARD_SIGN = +1

# ---------- Postures: (body height offset m, forward lean rad) ----------
POSTURE = {
    "NEUTRAL":     (0.00,  0.00),
    "CHECK_IN":    (0.02,  0.05),
    "ENGAGE":      (0.08,  0.10),
    "CAUTION":     (0.00,  0.00),
    "DE_ESCALATE": (-0.08, -0.10),   # negative lean = lean back
}

# ---------- Extra gestures ----------
NOD_LEAN, NOD_HOLD_S, NOD_MOVE_S = 0.15, 0.3, 0.35          # ~1 s total
SCAN_YAW, SCAN_HOLD_S, SCAN_MOVE_S = math.radians(15), 0.3, 0.6  # ~3 s total
BOB_DROP, BOB_HALF_S = 0.05, 0.5                             # ~1 s total


def _clamp(v, lim):
    return max(-lim, min(lim, v))


@dataclass
class ReactionAction:
    intent: str
    pose_mode: str = "REACTION_POSE"


class MotionDispatcher:
    """Runs one movement at a time on its own thread, "interrupt & replace":
    a new movement stops the current one first (same as the MentorPi runner).
    Each movement function must accept stop_event=..."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = None
        self._thread = None

    def dispatch(self, fn, *args, **kwargs):
        with self._lock:
            if self._stop is not None:
                self._stop.set()
            stop_event = threading.Event()
            prev = self._thread

            def run():
                if prev is not None:
                    prev.join(timeout=5.0)
                if not stop_event.is_set():
                    try:
                        fn(*args, stop_event=stop_event, **kwargs)
                    except Exception as e:
                        print(f"[MOTION] error: {e}")

            self._stop, self._thread = stop_event, threading.Thread(
                target=run, daemon=True, name="SpotMotion")
            self._thread.start()

    def stop_all(self, timeout=5.0):
        with self._lock:
            if self._stop is not None:
                self._stop.set()
            t = self._thread
        if t is not None:
            t.join(timeout=timeout)


@dataclass
class PersonState:
    """Same fields the MentorPi adapter's state uses."""
    person_visible:  bool  = False
    person_distance: float = 3.0     # m
    person_bearing:  float = 0.0     # rad, positive = person to Spot's RIGHT
    missing_windows: int   = 0


class SpotAdapter:
    def __init__(self, hostname=None, mock=True, allow_walking=True,
                 allow_turning=True, speed_scale=1.0, robot=None):
        """robot: an already-authenticated bosdyn Robot to reuse (so the live
        gesture script only asks for the login once). Optional."""
        self.mock = mock
        self.hostname = hostname
        self._given_robot = robot
        self.allow_walking = allow_walking
        self.allow_turning = allow_turning
        self.speed_scale = max(0.0, min(1.0, speed_scale))
        self.state = PersonState()
        self._lock = threading.Lock()      # one command sender at a time
        self._intent = None                # for per-streak targets
        self._engage_target = None
        self._engage_start = 0.0
        self._walked = 0.0
        self._posture = (0.0, 0.0, 0.0)    # height, lean, yaw last sent
        self._last_vel = (None, 0.0)
        self._stack = contextlib.ExitStack()
        self._robot = self._cmd = None
        self._started = False

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------
    def start(self):
        if self.mock:
            print("[SPOT-MOCK] Mock mode: no connection, commands are printed only.")
            self._started = True
            return
        import bosdyn.client
        import bosdyn.client.util
        from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
        from bosdyn.client.robot_command import RobotCommandClient, blocking_stand

        if self._given_robot is not None:
            robot = self._given_robot
        else:
            sdk = bosdyn.client.create_standard_sdk("SpotGestureAdapter")
            robot = sdk.create_robot(self.hostname)
            bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()
        if robot.is_estopped():
            raise RuntimeError(
                "Spot is E-stopped or has no E-stop configured. The E-stop operator "
                "must start the E-stop (tablet or SDK estop example) and release it first.")
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        try:
            self._stack.enter_context(
                LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True))
        except Exception as e:
            raise RuntimeError(
                f"Could not take the lease ({e}). If the tablet is driving Spot, "
                "release control on the tablet first.") from e
        print("[SPOT] Lease acquired. Powering on...")
        robot.power_on(timeout_sec=20)
        if not robot.is_powered_on():
            raise RuntimeError("Spot did not power on.")
        self._robot = robot
        self._cmd = robot.ensure_client(RobotCommandClient.default_service_name)
        print("[SPOT] Standing up...")
        blocking_stand(self._cmd, timeout_sec=10)
        self._started = True
        print("[SPOT] Ready.")

    def shutdown(self):
        if not self._started:
            self._stack.close()
            return
        if self.mock:
            print("[SPOT-MOCK] Shutdown: would stop, sit, power off, release lease.")
            return
        from bosdyn.client.robot_command import RobotCommandBuilder
        try:
            print("[SPOT] Sitting down...")
            with self._lock:
                self._cmd.robot_command(RobotCommandBuilder.synchro_sit_command())
            time.sleep(3.0)
            print("[SPOT] Powering off...")
            self._robot.power_off(cut_immediately=False, timeout_sec=20)
        except Exception as e:
            print(f"[SPOT] Shutdown problem: {e}  (use the E-stop / tablet if needed)")
        finally:
            self._stack.close()   # returns the lease
            print("[SPOT] Lease released.")

    # ------------------------------------------------------------------
    # Low-level senders
    # ------------------------------------------------------------------
    def _send_posture(self, height, lean, yaw=0.0, note=""):
        height = _clamp(height, MAX_HEIGHT)
        lean = _clamp(lean, MAX_TILT)
        yaw = _clamp(yaw, MAX_YAW)
        self._posture = (height, lean, yaw)
        if self.mock:
            print(f"[SPOT-MOCK] stand  height={height:+.2f} m  lean={lean:+.2f} rad  "
                  f"yaw={yaw:+.2f} rad  {note}")
            return
        from bosdyn.client.robot_command import RobotCommandBuilder
        from bosdyn.geometry import EulerZXY
        cmd = RobotCommandBuilder.synchro_stand_command(
            body_height=height,
            footprint_R_body=EulerZXY(yaw=yaw, roll=0.0, pitch=PITCH_FORWARD_SIGN * lean))
        with self._lock:
            self._cmd.robot_command(cmd)

    def _send_velocity(self, v_x, v_rot, note=""):
        height, lean, _ = self._posture
        # Commands go out ~10x per second; only print when they change or
        # once a second, so the log stays readable (mock and real mode).
        key = (round(v_x, 2), round(v_rot, 2))
        if key != self._last_vel[0] or time.time() - self._last_vel[1] >= 1.0:
            tag = "[SPOT-MOCK]" if self.mock else "[SPOT]"
            print(f"{tag} walk   v_x={v_x:+.2f} m/s  v_rot={v_rot:+.2f} rad/s  {note}")
            self._last_vel = (key, time.time())
        if self.mock:
            return
        from bosdyn.client.robot_command import RobotCommandBuilder
        from bosdyn.geometry import EulerZXY
        params = RobotCommandBuilder.mobility_params(
            body_height=height,
            footprint_R_body=EulerZXY(yaw=0.0, roll=0.0, pitch=PITCH_FORWARD_SIGN * lean))
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=v_x, v_y=0.0, v_rot=v_rot, params=params)
        try:
            with self._lock:
                self._cmd.robot_command(cmd, end_time_secs=time.time() + VEL_CMD_DURATION_S)
        except Exception as e:
            print(f"[SPOT] Spot REJECTED the walk command: {e}")
            raise

    def new_streak(self):
        """Forget the current intent, so the next hold_for starts a fresh
        streak (new ENGAGE target, retreat counter back to zero)."""
        self._intent = None

    def stop(self):
        """Stop walking/turning and hold the current posture."""
        h, lean, _ = self._posture
        self._last_vel = (None, 0.0)
        self._send_posture(h, lean, 0.0, "(stop)")

    @staticmethod
    def _wait(stop_event, seconds):
        """Sleep, but return True early if interrupted."""
        if stop_event is None:
            time.sleep(seconds)
            return False
        return stop_event.wait(seconds)

    # ------------------------------------------------------------------
    # Intents
    # ------------------------------------------------------------------
    def hold_for(self, ra, duration_s, rate_hz=CONTROL_RATE_HZ, stop_event=None):
        intent = ra.intent if ra.intent in POSTURE else "NEUTRAL"
        s = self.state

        # New streak: reset per-streak targets and set the posture.
        if intent != self._intent:
            self._intent = intent
            self._walked = 0.0
            self._engage_target = None
            if intent == "ENGAGE" and s.person_visible:
                self._engage_start = s.person_distance
                self._engage_target = max(ENGAGE_MIN_DIST,
                                          ENGAGE_FRACTION * s.person_distance)
                print(f"[SPOT] ENGAGE target: {self._engage_target:.2f} m "
                      f"(starting at {s.person_distance:.2f} m)")
            h, lean = POSTURE[intent]
            self._send_posture(h, lean, 0.0, f"({intent} posture)")

        if intent in ("NEUTRAL", "CAUTION"):
            self._wait(stop_event, duration_s)      # posture already set
            return

        period = 1.0 / rate_hz
        fwd_max = MAX_FORWARD_MPS * self.speed_scale
        back_max = MAX_BACKWARD_MPS * self.speed_scale
        turn_max = MAX_TURN_RADPS * self.speed_scale
        end = time.time() + duration_s
        was_moving = False
        last_reason = None

        while time.time() < end:
            if stop_event is not None and stop_event.is_set():
                break
            v_x, v_rot, why = 0.0, 0.0, ""
            visible = s.person_visible

            if not visible:
                why = "no person visible"
            else:
                off = s.person_bearing
                # Turn to face first (CHECK_IN, ENGAGE). bearing > 0 = person
                # on the right -> turn right (negative v_rot).
                if intent in ("CHECK_IN", "ENGAGE") and abs(off) > FACE_TOLERANCE:
                    if self.allow_turning:
                        v_rot = _clamp(-TURN_GAIN * off, turn_max)
                        why = f"turning to face ({math.degrees(off):+.0f} deg)"
                    else:
                        why = "turning disabled"
                elif intent == "ENGAGE" and self._engage_target is not None:
                    d = s.person_distance
                    # Backup limit in case the face-size distance is noisy:
                    # never walk much further than the planned gap.
                    max_walk = max(0.0, self._engage_start - self._engage_target)
                    if d <= self._engage_target or d <= FORWARD_FLOOR:
                        why = f"at ENGAGE distance ({d:.2f} m)"
                    elif self._walked >= max_walk:
                        why = "walked the planned distance"
                    elif not self.allow_walking:
                        why = "walking disabled"
                    else:
                        v_x = fwd_max
                        why = f"approaching ({d:.2f} m -> {self._engage_target:.2f} m)"
                elif intent == "DE_ESCALATE":
                    if self._walked >= RETREAT_M:
                        why = f"retreat done ({RETREAT_M:.2f} m)"
                    elif not self.allow_walking:
                        why = "walking disabled"
                    else:
                        v_x = -back_max
                        why = f"backing away ({self._walked:.2f}/{RETREAT_M:.2f} m)"

            reason = why.split(" (")[0]          # ignore the changing numbers
            if reason != last_reason:
                print(f"[SPOT] {intent}: {why or 'holding posture'}")
                last_reason = reason
            if v_x != 0.0 or v_rot != 0.0:
                self._send_velocity(v_x, v_rot, why)
                self._walked += abs(v_x) * period    # commanded-speed estimate
                was_moving = True
            elif was_moving:
                self.stop()
                was_moving = False
            if self._wait(stop_event, period):
                break

        if was_moving:
            self.stop()

    # ------------------------------------------------------------------
    # Extra gestures (postures only, Spot does not step)
    # ------------------------------------------------------------------
    def checkin_ack(self, stop_event=None):
        """One nod: lean forward ~9 deg, hold, return to the CHECK_IN posture."""
        h, lean = POSTURE["CHECK_IN"]
        print("[SPOT] CHECK_IN nod")
        self._send_posture(h, NOD_LEAN, 0.0, "(nod down)")
        interrupted = self._wait(stop_event, NOD_MOVE_S + NOD_HOLD_S)
        self._send_posture(h, lean, 0.0, "(nod up)")
        if not interrupted:
            self._wait(stop_event, NOD_MOVE_S)

    def caution_scan(self, stop_event=None):
        """Body turns ~15 deg one way, then the other, then back. No steps."""
        h, lean = POSTURE["CAUTION"]
        print("[SPOT] CAUTION scan")
        for yaw, wait in ((+SCAN_YAW, SCAN_MOVE_S + SCAN_HOLD_S),
                          (-SCAN_YAW, 2 * SCAN_MOVE_S + SCAN_HOLD_S)):
            self._send_posture(h, lean, yaw, "(scan)")
            if self._wait(stop_event, wait):
                break
        self._send_posture(h, lean, 0.0, "(scan centre)")

    def neutral_pulse(self, stop_event=None):
        """Small body bob: lower ~5 cm and back."""
        h, lean = POSTURE["NEUTRAL"]
        print("[SPOT] idle bob")
        self._send_posture(h - BOB_DROP, lean, 0.0, "(bob down)")
        interrupted = self._wait(stop_event, BOB_HALF_S)
        self._send_posture(h, lean, 0.0, "(bob up)")
        if not interrupted:
            self._wait(stop_event, BOB_HALF_S)
