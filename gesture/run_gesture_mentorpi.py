#!/usr/bin/env python3
"""
============================================================================
run_gesture_mentorpi.py  —  Gesture-driven intents for the MentorPi
============================================================================
Phase 1 + 2 of the gesture plan. Reads the MentorPi's HTTP camera stream,
recognizes hand gestures with MediaPipe's Gesture Recognizer, turns a held
gesture into one of the five intents, and drives the robot through the
existing MentorPiAdapter + safety layer.

    Thumb up      -> ENGAGE       (forward approach to ~1.0 m, no extra gesture)
    Pointing up   -> CHECK_IN     (forward-back "I see you" nod, then approach
                                   to ~1.5 m and turn to face; the nod repeats
                                   every time the intent is re-sent, see
                                   CHECKIN_ACK_EVERY_DISPATCH)
    Open palm     -> CAUTION      (freeze; SCAN once per streak if held)
    Thumb down    -> DE_ESCALATE  (back away to ~2.0 m)
    anything else -> NEUTRAL      (stop; gentle PULSE every 20 s if idle)

How a gesture becomes motion:
  1. Face tracking (PersonTracker, every few frames) finds the person and
     updates adapter.state (visible, distance, bearing), same as the other
     runners.
  2. The gesture recognizer runs on a crop that FOLLOWS the person: centred
     on their face and sized from the face's height, so a hand stays a
     similar size to the model at 1 m, 1.5 m and 2 m. Phase 0 showed a
     fixed crop only works at one distance, and no crop fails at 1.5 m.
  3. Hold check: a gesture only counts once it has been seen in at least
     MIN_VOTES frames within the last VOTE_WINDOW_S seconds and makes up
     most of the gesture frames in that window. Frames with no hand or an
     unmapped gesture ("None", fist, victory...) are ignored, not counted
     against it. One misread frame can never move the robot.
  4. While the gesture is held, its intent is re-sent every REDISPATCH_S
     seconds, so the robot keeps acting (the adapter stops approaching at
     the intent's target distance). After RELEASE_S seconds without the
     gesture, the robot returns to NEUTRAL and stops.
  5. Gestures are only acted on while a face has been seen recently, so a
     hand with no person attached (or a stale detection) can't drive it.

Motion runs on its own thread with "interrupt & replace": a new decision
stops the current hold immediately. The CHECK_IN nod (1 s) can be cut short
too; SCAN (about 3 s) finishes before the next command starts.

MUST run on the Pi, from gesture/ in the repo. The robot modules are
imported from robot/; gesture_recognizer.task must be downloaded into
this folder (see README.md).

Usage:
    source ~/va_pipeline_venv/bin/activate
    cd ~/MentorPi-code/gesture

    # Mock mode (default): prints what the robot WOULD do, no wheel motion
    python3 run_gesture_mentorpi.py --duration 60

    # Real wheel motion: open space, hand near the power switch
    python3 run_gesture_mentorpi.py --duration 60 --real-motion

    # Live dashboard works too (separate terminal): python3 dashboard_server.py
============================================================================
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from mentorpi_adapter import MentorPiAdapter
from person_tracker import PersonTracker
from lidar_reader import LidarDistanceReader
from safety import apply_all

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"ERROR: MediaPipe Tasks API not available: {e}")
    sys.exit(1)


# ---------- Configuration ----------
DEFAULT_STREAM_URL = (
    "http://127.0.0.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image"
)
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gesture_recognizer.task")

GESTURE_TO_INTENT = {
    "Thumb_Up":    "ENGAGE",
    "Pointing_Up": "CHECK_IN",
    "Open_Palm":   "CAUTION",
    "Thumb_Down":  "DE_ESCALATE",
}

MIN_SCORE      = 0.6    # ignore gesture frames below this confidence
VOTE_WINDOW_S  = 1.0    # hold check looks at this much recent history
MIN_VOTES      = 4      # frames of the same gesture needed in the window
MIN_SHARE      = 0.75   # ...and it must be this share of gesture frames
RELEASE_S      = 1.5    # no gesture for this long -> back to NEUTRAL
REDISPATCH_S   = 2.5    # re-send a held gesture's intent this often
HOLD_S         = 3.0    # each intent's motion lasts this long unless replaced
HOLD_CHUNK_S   = 0.5    # re-check person distance this often during a hold

FACE_STRIDE        = 3     # run face detection every Nth frame
FACE_RECENT_S      = 2.0   # gestures only count if a face was seen this recently
MISSING_STEP_S     = 1.0   # each second without a face = 1 "missing window"

# Person-following crop, in units of the face's height (s):
CROP_WIDTH_FACES = 7.0     # crop width = 7 s, centred on the face
CROP_UP_FACES    = 1.5     # crop top = face top - 1.5 s (hands raised high)
CROP_DOWN_FACES  = 5.0     # crop bottom = face top + 5 s (chest-height hands)
CROP_MIN_PX      = 160     # never shrink the crop below this
CROP_SMOOTH      = 0.3     # 0..1, how fast the crop follows the face

# CHECK_IN nod: True = on every send (every REDISPATCH_S while pointing up),
# False = once, when a CHECK_IN streak starts.
CHECKIN_ACK_EVERY_DISPATCH = True
NEUTRAL_PULSE_EVERY_S   = 20.0  # gentle "still here" pulse while idle

STATUS_FILE = "/tmp/spot_dashboard_status.json"   # read by dashboard_server.py
HISTORY_MAX = 5


@dataclass
class ReactionAction:
    intent:    str
    pose_mode: str
    valence:   float = 0.0
    arousal:   float = 0.0


# ============================================================================
# Hold check
# ============================================================================
class GestureVoter:
    """Keeps the last VOTE_WINDOW_S seconds of mapped gestures and decides
    which one (if any) is being held."""

    def __init__(self):
        self.votes = deque()   # (time, intent)

    def add(self, t, intent):
        if intent is not None:
            self.votes.append((t, intent))

    def decide(self, now):
        while self.votes and now - self.votes[0][0] > VOTE_WINDOW_S:
            self.votes.popleft()
        if not self.votes:
            return None
        counts = Counter(i for _, i in self.votes)
        intent, n = counts.most_common(1)[0]
        if n >= MIN_VOTES and n / len(self.votes) >= MIN_SHARE:
            return intent
        return None


# ============================================================================
# Person-following crop
# ============================================================================
class CropFollower:
    def __init__(self, w, h):
        self.w, self.h = w, h
        # Default before any face is seen: the window that worked at
        # 1.5-2 m in Phase 0 (middle 50% wide, bottom half of the frame).
        self.box = np.array([w * 0.25, h * 0.5, w * 0.75, h * 1.0])

    def update(self, bbox):
        fx, fy, fw, fh = bbox
        s = max(fw, fh, 1)
        cx = fx + fw / 2.0
        half_w = max(CROP_WIDTH_FACES * s, CROP_MIN_PX) / 2.0
        x0, x1 = cx - half_w, cx + half_w
        y0 = fy - CROP_UP_FACES * s
        y1 = max(fy + CROP_DOWN_FACES * s, y0 + CROP_MIN_PX)
        target = np.array([max(0, x0), max(0, y0),
                           min(self.w, x1), min(self.h, y1)])
        self.box = (1 - CROP_SMOOTH) * self.box + CROP_SMOOTH * target

    def crop(self, frame):
        x0, y0, x1, y1 = (int(round(v)) for v in self.box)
        return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


# ============================================================================
# Main
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Gesture-driven MentorPi runner")
    p.add_argument("--duration", type=float, default=0,
                   help="Seconds to run (default 0 = until Ctrl+C)")
    p.add_argument("--real-motion", action="store_true",
                   help="Enable real wheel motion. Default is MOCK.")
    p.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-lidar", action="store_true",
                   help="Skip the LiDAR (face-size distance only)")
    p.add_argument("--debug", action="store_true",
                   help="Print every per-frame gesture change")
    p.add_argument("--face-conf", type=float, default=0.5,
                   help="Face detection confidence threshold (default 0.5). "
                        "Try 0.3 if your face isn't found in dim or backlit spots.")
    p.add_argument("--snapshot", default=None, metavar="PATH",
                   help="Save the full camera frame 10 s into the run, with the "
                        "detected face (green) and gesture crop (blue) drawn on it")
    return p.parse_args()


def main():
    args = parse_args()
    is_mock = not args.real_motion

    if not os.path.exists(args.model) or os.path.getsize(args.model) == 0:
        print(f"ERROR: gesture model missing or empty: {args.model}")
        sys.exit(1)

    print("=" * 62)
    print("  MENTORPI GESTURE RUNNER")
    if is_mock:
        print("  Mode: MOCK (no motion). Ctrl+C to stop.")
    else:
        print("  Mode: *** REAL MOTION *** -- robot wheels will actually turn!")
        print("  Keep ~2.5 m clear in front and behind. Hand near the power switch.")
    print("  Thumb up=ENGAGE  Pointing up=CHECK_IN  Open palm=CAUTION  "
          "Thumb down=DE_ESCALATE")
    print("=" * 62)

    # ---- Robot + sensors ----
    adapter = MentorPiAdapter(mock=is_mock)
    adapter.state.person_visible  = False
    adapter.state.person_distance = 3.0
    adapter.state.person_bearing  = 0.0
    adapter.state.missing_windows = 0

    lidar_reader = None
    if not args.no_lidar:
        try:
            lidar_reader = LidarDistanceReader(window_deg=10.0, aggregation="min")
            lidar_reader.start()
            print("[LIDAR] Using LiDAR for distance.")
        except Exception as e:
            print(f"[LIDAR] Could not start LiDAR reader ({e}); "
                  "using face-size distance.")
            lidar_reader = None

    tracker = PersonTracker(camera_hfov_deg=90.0, distance_k=60.0,
                            min_confidence=args.face_conf, model_selection=1,
                            lidar_reader=lidar_reader)

    recognizer = vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        ))

    cap = cv2.VideoCapture(args.stream_url)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        print(f"ERROR: could not read from camera stream {args.stream_url}")
        adapter.shutdown()
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"[CAM] Stream opened: {w}x{h}")
    follower = CropFollower(w, h)

    # ---- Dashboard status (same file/shape as continuous_runner_http.py) ----
    status = {
        "cycle": 0, "phase": "gesture",
        "person": {"visible": False, "distance_m": None, "bearing_rad": None,
                   "bearing_deg": None, "distance_source": None},
        "emotion": {"valence": None, "arousal": None},
        "intent": {"current": "NEUTRAL", "history": []},
        "motion": {"linear_mps": 0.0, "angular_radps": 0.0, "reason": None},
        "safety": {"estop": "OK", "distance_floor": "OK", "battery": "OK",
                   "missing_windows": 0},
        "camera": {"stream_url": "/camera"},
        "runner": {"mode": "MOCK" if is_mock else "REAL"},
        "gesture": {"label": None, "score": None},
        "history": [],
    }

    def write_status():
        try:
            status["timestamp"] = time.time()
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f)
        except Exception:
            pass

    # ---- Motion thread: interrupt & replace ----
    motion_lock = threading.Lock()
    current = {"stop": None, "thread": None}

    def hold_interruptible(ra, duration, stop_event):
        end = time.time() + duration
        while not stop_event.is_set():
            left = end - time.time()
            if left <= 0:
                break
            adapter.hold_for(ra, min(HOLD_CHUNK_S, left), stop_event=stop_event)

    def run_motion(intent, extra, stop_event, prev_thread):
        # Let the previous command finish stopping (SCAN can't be cut short)
        # so two threads never publish at the same time.
        if prev_thread is not None:
            prev_thread.join(timeout=5.0)
        if stop_event.is_set():
            return
        if extra == "ACK":
            adapter.checkin_ack(stop_event=stop_event)
            if stop_event.is_set():
                return
        elif extra == "SCAN":
            adapter.caution_scan()
        elif extra == "PULSE":
            adapter.neutral_pulse()
            return
        ra = ReactionAction(intent=intent, pose_mode="REACTION_POSE")
        if intent == "NEUTRAL":
            hold_interruptible(ra, 0.5, stop_event)
            return
        hold_interruptible(ra, HOLD_S, stop_event)
        if not stop_event.is_set():
            stop_ra = ReactionAction(intent="NEUTRAL", pose_mode="REACTION_POSE")
            hold_interruptible(stop_ra, 0.5, stop_event)

    def dispatch(intent, extra=None):
        with motion_lock:
            if current["stop"] is not None:
                current["stop"].set()
            stop_event = threading.Event()
            t = threading.Thread(target=run_motion,
                                 args=(intent, extra, stop_event, current["thread"]),
                                 daemon=True, name="MotionDispatch")
            current["stop"], current["thread"] = stop_event, t
            t.start()

        preview = apply_all(adapter._motion_for_intent(intent), adapter.state)
        status["intent"]["current"] = intent
        status["intent"]["history"] = (status["intent"]["history"] + [intent])[-5:]
        status["motion"] = {"linear_mps": round(preview.linear, 3),
                            "angular_radps": round(preview.angular, 3),
                            "reason": preview.reason}
        status["cycle"] += 1
        status["history"].insert(0, {
            "cycle": status["cycle"], "timestamp": time.time(),
            "valence": None, "arousal": None, "intent": intent,
            "distance_m": status["person"]["distance_m"],
            "bearing_deg": status["person"]["bearing_deg"],
            "linear_mps": round(preview.linear, 3),
            "angular_radps": round(preview.angular, 3),
        })
        status["history"] = status["history"][:HISTORY_MAX]
        write_status()

    # ---- Real-motion countdown ----
    # Keep reading (and discarding) frames during the countdown. A plain
    # sleep here let ~5 s of frames pile up in OpenCV's stream buffer, so the
    # first several seconds of the run were processing stale images.
    if not is_mock:
        print("\n  Starting real motion in 5 seconds -- Ctrl+C to abort.")
        t_wait = time.time()
        while time.time() - t_wait < 5.0:
            cap.read()

    voter = GestureVoter()
    active_intent = "NEUTRAL"
    last_dispatch_t = 0.0
    last_gesture_t = 0.0
    last_face_t = 0.0
    streak_count = 0
    caution_scanned = False
    neutral_since = time.time()
    last_pulse_t = time.time()
    dispatch_counts = Counter()

    frames = 0
    face_checks = face_hits = 0
    last_bbox = None
    snapshot_saved = False
    recog_total = 0.0
    last_label = None
    last_report_t = time.time()
    t_start = time.time()
    t0_ms = int(t_start * 1000)

    print("\n[RUN] Watching for gestures...\n")
    try:
        while args.duration <= 0 or time.time() - t_start < args.duration:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frames += 1
            now = time.time()

            # -- 1. Face tracking: person state + crop position --
            if frames % FACE_STRIDE == 0:
                det = tracker.process_frame(frame)
                face_checks += 1
                if det["visible"]:
                    face_hits += 1
                    last_bbox = det["bbox"]
                    last_face_t = now
                    adapter.state.person_visible  = True
                    adapter.state.person_distance = det["distance_m"]
                    adapter.state.person_bearing  = det["bearing_rad"]
                    follower.update(det["bbox"])
                    status["person"] = {
                        "visible": True,
                        "distance_m": round(det["distance_m"], 2),
                        "bearing_rad": round(det["bearing_rad"], 3),
                        "bearing_deg": round(det["bearing_rad"] * 57.2958, 1),
                        "distance_source": det["distance_source"],
                    }
                elif now - last_face_t > FACE_RECENT_S:
                    adapter.state.person_visible = False
                    status["person"]["visible"] = False
                missing = 0 if adapter.state.person_visible else \
                    int((now - last_face_t) / MISSING_STEP_S)
                adapter.state.missing_windows = missing
                status["safety"]["missing_windows"] = missing

            # -- 2. Gesture on the person-following crop --
            roi, crop_box = follower.crop(frame)
            if args.snapshot and not snapshot_saved and now - t_start >= 10.0:
                snap = frame.copy()
                x0, y0, x1, y1 = crop_box
                cv2.rectangle(snap, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 2)
                if last_bbox is not None and now - last_face_t <= 1.0:
                    bx, by, bw, bh = last_bbox
                    cv2.rectangle(snap, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.imwrite(args.snapshot, snap)
                snapshot_saved = True
                print(f"[{now - t_start:6.1f}s] snapshot saved to {args.snapshot}")
            if roi.size == 0:
                continue
            rgb = np.ascontiguousarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            t_r = time.time()
            result = recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(t_r * 1000) - t0_ms + frames)
            recog_total += time.time() - t_r

            label, score = None, None
            tops = [g[0] for g in result.gestures if g]
            if tops:
                best = max(tops, key=lambda c: c.score)
                label, score = best.category_name, best.score
            intent_vote = None
            if label in GESTURE_TO_INTENT and score >= MIN_SCORE:
                intent_vote = GESTURE_TO_INTENT[label]
            face_recent = now - last_face_t <= FACE_RECENT_S
            voter.add(now, intent_vote if face_recent else None)
            status["gesture"] = {"label": label,
                                 "score": round(score, 2) if score else None}

            if args.debug and label != last_label:
                print(f"[{now - t_start:6.1f}s] frame gesture: {label} "
                      f"{'' if score is None else f'({score:.2f})'}")
                last_label = label

            # -- 3. Hold check -> intent decisions --
            held = voter.decide(now)
            if held is not None:
                last_gesture_t = now
                new_streak = held != active_intent
                if new_streak or now - last_dispatch_t >= REDISPATCH_S:
                    streak_count = 1 if new_streak else streak_count + 1
                    if new_streak:
                        caution_scanned = False
                    extra = None
                    if held == "CHECK_IN" and (new_streak or CHECKIN_ACK_EVERY_DISPATCH):
                        extra = "ACK"
                    elif held == "CAUTION" and streak_count >= 2 and not caution_scanned:
                        extra, caution_scanned = "SCAN", True
                    d = adapter.state.person_distance
                    print(f"[{now - t_start:6.1f}s] GESTURE -> {held}"
                          f"{' (new)' if new_streak else ' (held)'}"
                          f"{' + ' + extra if extra else ''}"
                          f"   person {d:.2f} m")
                    active_intent = held
                    last_dispatch_t = now
                    dispatch_counts[held] += 1
                    dispatch(held, extra)
            elif active_intent != "NEUTRAL" and now - last_gesture_t > RELEASE_S:
                print(f"[{now - t_start:6.1f}s] gesture released -> NEUTRAL")
                active_intent = "NEUTRAL"
                last_dispatch_t = now
                neutral_since = last_pulse_t = now
                streak_count = 0
                dispatch_counts["NEUTRAL"] += 1
                dispatch("NEUTRAL")
            elif (active_intent == "NEUTRAL" and adapter.state.person_visible
                  and now - last_pulse_t >= NEUTRAL_PULSE_EVERY_S
                  and now - neutral_since >= NEUTRAL_PULSE_EVERY_S):
                print(f"[{now - t_start:6.1f}s] idle {now - neutral_since:.0f}s -> PULSE")
                last_pulse_t = now
                dispatch("NEUTRAL", "PULSE")

            # -- 4. Periodic status line + dashboard refresh --
            if now - last_report_t >= 5.0:
                el = now - t_start
                p = status["person"]
                person_txt = "visible" if p["visible"] else "not visible"
                if p["visible"] and p["distance_m"] is not None:
                    person_txt += " at %.2f m" % p["distance_m"]
                print(f"[{el:6.1f}s] status: {frames / el:.1f} fps, "
                      f"gesture {1000 * recog_total / max(frames, 1):.0f} ms/frame, "
                      f"face found in {face_hits}/{face_checks} checks, "
                      f"person {person_txt}, intent {active_intent}")
                face_hits = face_checks = 0
                last_report_t = now
                write_status()

    except KeyboardInterrupt:
        print("\n  Stopped (Ctrl+C).")
    finally:
        with motion_lock:
            if current["stop"] is not None:
                current["stop"].set()
        if current["thread"] is not None:
            current["thread"].join(timeout=5.0)
        adapter.shutdown()
        cap.release()
        recognizer.close()
        tracker.close()
        if lidar_reader is not None:
            try:
                lidar_reader.stop()
            except Exception:
                pass

    el = max(time.time() - t_start, 1e-6)
    print("\n================ SUMMARY ================")
    print(f"Ran {el:.0f}s, {frames} frames ({frames / el:.1f} fps), "
          f"gesture recognizer {1000 * recog_total / max(frames, 1):.0f} ms/frame")
    print("Intent dispatches:")
    for name, n in dispatch_counts.most_common():
        print(f"    {name:<12} {n}")
    if not dispatch_counts:
        print("    none")


if __name__ == "__main__":
    main()
