#!/usr/bin/env python3
"""
============================================================================
spot_gestures.py  —  The GESTURE side (camera -> person -> gesture -> intent)
============================================================================
Everything about seeing and deciding, nothing about how Spot moves (that is
spot_adapter.py). Shared by the two runnable programs:

    spot_gesture_basic.py   follows the person nearest the centre of the view
    spot_gesture_tag.py     only the person wearing the AprilTag can control Spot

What this file does each frame:
  1. Gets a frame from the Spot CAM PTZ (live WebRTC stream) or a recording.
  2. Finds every face; each face keeps its own ID while it stays in view
     (FaceTracker). A "controller" object (CentreLock here, TagControl in
     spot_tag_control.py) decides which face is the person in control.
  3. Runs MediaPipe's Gesture Recognizer on a crop that follows that person,
     and keeps only hands that belong to them.
  4. Turns a held gesture into an intent with the hold check, and hands the
     intent to spot_adapter.py (when --drive mock/real):
        Thumb up -> ENGAGE        Pointing up -> CHECK_IN (+ one nod per streak)
        Open palm -> CAUTION (+ one scan per streak)   Thumb down -> DE_ESCALATE
        anything else -> NEUTRAL (+ small bob after 30 s idle)

Needs (in the same folder, the Spot SDK's examples/spot_cam):
    gesture_recognizer.task, blaze_face_full_range.tflite, spot_adapter.py,
    and the SDK example's webrtc.py / webrtc_client.py.
============================================================================
"""

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
import types
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CALIBRATION_FILE = os.path.join(HERE, "spot_calibration.json")

# ---------- Gesture -> intent and decision rules ----------
GESTURE_TO_INTENT = {
    "Thumb_Up":    "ENGAGE",
    "Pointing_Up": "CHECK_IN",
    "Open_Palm":   "CAUTION",
    "Thumb_Down":  "DE_ESCALATE",
}
MIN_SCORE, VOTE_WINDOW_S, MIN_VOTES, MIN_SHARE = 0.6, 1.0, 4, 0.75
RELEASE_S, REDISPATCH_S, HOLD_S = 1.5, 2.5, 3.0
FACE_STRIDE = 3
CROP_WIDTH_FACES, CROP_UP_FACES, CROP_DOWN_FACES = 7.0, 1.5, 5.0
CROP_MIN_PX, CROP_SMOOTH = 160, 0.3
CHECKIN_ACK_EVERY_DISPATCH = False   # Spot: one nod per CHECK_IN streak
NEUTRAL_PULSE_EVERY_S = 30.0         # Spot: idle bob every 30 s

# ---------- Which face is a person, and which hands are theirs ----------
TRACK_GATE_FACES = 1.0       # a face may move this many face widths between checks
TRACK_SIZE_RATIO = (0.6, 1.6)
TRACK_FORGET_S = 1.0         # drop a face track unseen this long
PERSON_LOST_S = 2.0          # controller unseen this long -> nobody in control
HAND_X_FACES = 3.0           # hand within this many face widths left/right of the face
HAND_Y_UP_FACES = 1.5        # ...from this far above the face
HAND_Y_DOWN_FACES = 5.0      # ...to this far below it (waist height)
HAND_SIZE_RATIO = (0.4, 2.2) # hand size / face size


def centre(bb):
    x, y, w, h = bb
    return x + w / 2.0, y + h / 2.0


def size(bb):
    return max(bb[2], bb[3], 1)


def hand_belongs(hand, face_bb, other_faces=()):
    """hand = (centre_x, centre_y, size) in full-frame pixels. True if the hand
    is in this person's zone, about the right size for their face, and not
    closer to someone else's face."""
    fx, fy, fw, fh = face_bb
    s = size(face_bb)
    if abs(hand[0] - (fx + fw / 2.0)) > HAND_X_FACES * s:
        return False
    if not (fy - HAND_Y_UP_FACES * s <= hand[1] <= fy + HAND_Y_DOWN_FACES * s):
        return False
    if not (HAND_SIZE_RATIO[0] <= hand[2] / s <= HAND_SIZE_RATIO[1]):
        return False
    cx, cy = centre(face_bb)
    mine = math.hypot(hand[0] - cx, hand[1] - cy) / s
    for bb in other_faces:
        if bb == face_bb:
            continue
        ox, oy = centre(bb)
        if math.hypot(hand[0] - ox, hand[1] - oy) / size(bb) < mine:
            return False
    return True


class FaceTracker:
    """Every face keeps its own ID while it stays in view, so the person in
    control can't be swapped for someone else with a similar face size."""

    def __init__(self):
        self.tracks = {}          # id -> {"bb": box, "seen": time}
        self._next_id = 1

    def update(self, faces, now):
        unmatched = list(faces)
        for tid, tr in sorted(self.tracks.items(), key=lambda kv: -kv[1]["seen"]):
            if not unmatched:
                break
            tcx, tcy = centre(tr["bb"])
            ts = size(tr["bb"])
            best, best_d = None, None
            for bb in unmatched:
                d = math.hypot(centre(bb)[0] - tcx, centre(bb)[1] - tcy)
                r = size(bb) / ts
                if d <= TRACK_GATE_FACES * ts and TRACK_SIZE_RATIO[0] <= r <= TRACK_SIZE_RATIO[1]:
                    if best_d is None or d < best_d:
                        best, best_d = bb, d
            if best is not None:
                tr["bb"], tr["seen"] = best, now
                unmatched.remove(best)
        for bb in unmatched:
            self.tracks[self._next_id] = {"bb": bb, "seen": now}
            self._next_id += 1
        for tid in [t for t, tr in self.tracks.items() if now - tr["seen"] > TRACK_FORGET_S]:
            del self.tracks[tid]

    def visible(self, now):
        return {tid: tr["bb"] for tid, tr in self.tracks.items() if tr["seen"] == now}


class CentreLock:
    """Controller for spot_gesture_basic.py: locks onto the person nearest the
    centre of the view and keeps following that person's face ID. If they're
    gone for PERSON_LOST_S, the person nearest the centre is picked again."""

    intro = "[CTRL] Following the person nearest the centre of the view."

    def __init__(self):
        self.faces = FaceTracker()
        self.controller_id = None
        self.target = None
        self.last_seen = -1e9
        self.events = []

    def update(self, frame, faces, now, w, h):
        self.faces.update(faces, now)
        visible = self.faces.visible(now)
        if self.controller_id is not None and self.controller_id not in self.faces.tracks \
                and now - self.last_seen > PERSON_LOST_S:
            self.events.append("Person lost")
            self.controller_id = None
        if self.controller_id is None and visible:
            self.controller_id = min(visible, key=lambda t: math.hypot(
                centre(visible[t])[0] - w / 2.0, centre(visible[t])[1] - h / 2.0))
            self.events.append(f"Following face #{self.controller_id} (nearest the centre)")
        if self.controller_id in visible:
            self.target, self.last_seen = visible[self.controller_id], now
            return self.target
        return None

    def present(self, now):
        return self.target is not None and now - self.last_seen <= PERSON_LOST_S

    def hand_ok(self, hand, faces):
        return self.target is not None and hand_belongs(hand, self.target, faces)

    def status(self, now):
        return f"FOLLOWING face #{self.controller_id}" if self.present(now) else "NO PERSON"

    def draw(self, view):
        pass

    def reset(self, now):
        self.controller_id, self.target = None, None
        self.events.append("Reset (r): picking the person nearest the centre again")


class GestureVoter:
    def __init__(self):
        self.votes = deque()

    def add(self, t, intent):
        if intent is not None:
            self.votes.append((t, intent))

    def decide(self, now):
        while self.votes and now - self.votes[0][0] > VOTE_WINDOW_S:
            self.votes.popleft()
        if not self.votes:
            return None
        intent, n = Counter(i for _, i in self.votes).most_common(1)[0]
        if n >= MIN_VOTES and n / len(self.votes) >= MIN_SHARE:
            return intent
        return None


class CropFollower:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.box = np.array([0.0, 0.0, float(w), float(h)])

    def target_box(self, bbox):
        """The gesture area for a face box, without smoothing."""
        fx, fy, fw, fh = bbox
        s = max(fw, fh, 1)
        cx = fx + fw / 2.0
        half_w = max(CROP_WIDTH_FACES * s, CROP_MIN_PX) / 2.0
        y0 = fy - CROP_UP_FACES * s
        y1 = max(fy + CROP_DOWN_FACES * s, y0 + CROP_MIN_PX)
        return np.array([max(0, cx - half_w), max(0, y0),
                         min(self.w, cx + half_w), min(self.h, y1)])

    def update(self, bbox):
        self.box = (1 - CROP_SMOOTH) * self.box + CROP_SMOOTH * self.target_box(bbox)

    def reset(self, bbox):
        """Jump straight to a face (e.g. a new controller), no smoothing."""
        self.box = self.target_box(bbox)

    def crop_at(self, frame, bbox):
        x0, y0, x1, y1 = (int(round(v)) for v in self.target_box(bbox))
        return frame[y0:y1, x0:x1], (x0, y0, x1, y1)

    def crop(self, frame):
        x0, y0, x1, y1 = (int(round(v)) for v in self.box)
        return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


class LatestFrame:
    """Holds only the newest frame, so processing never falls behind live video."""

    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._frame = None

    def put(self, frame):
        with self._lock:
            self._frame = frame
        self._event.set()

    def get(self, timeout):
        if not self._event.wait(timeout):
            return None
        with self._lock:
            frame, self._frame = self._frame, None
            self._event.clear()
        return frame


def connect_robot(hostname):
    """Log in once; the stream and the adapter share this Robot."""
    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client import spot_cam
    sdk = bosdyn.client.create_standard_sdk("SpotGestureLive")
    spot_cam.register_all_service_clients(sdk)
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    return robot


def set_ptz_only_view(robot, screen):
    """Switch the Spot CAM stream to one camera view (default mech_full =
    the PTZ alone, full frame) instead of the PTZ + 360 panorama layout."""
    from bosdyn.client.spot_cam.compositor import CompositorClient
    comp = robot.ensure_client(CompositorClient.default_service_name)
    comp.set_screen(screen)
    time.sleep(1.0)                       # give the stream a moment to switch
    current = comp.get_screen()
    print(f"[CAM] Stream view: {current}")
    if current != screen:
        print(f"[CAM] WARNING: asked for '{screen}' but the camera reports "
              f"'{current}'. Use --crop-ptz if the preview still shows both cameras.")


# Where the PTZ view sits inside the stacked PTZ + 360 layout ("mech"):
# top-middle, about 22-78% of the width and the top 55% of the height.
PTZ_PANE = (0.222, 0.0, 0.778, 0.555)    # x0, y0, x1, y1 as fractions


def start_spot_stream(robot, hostname, latest, shutdown_flag):
    """Receive the Spot CAM WebRTC video in a background thread (reuses the
    SDK example's webrtc.py), pushing frames into `latest`."""
    from webrtc import start_webrtc

    async def grab_frames(client, options, flag):
        while not flag.is_set():
            try:
                frame = await asyncio.wait_for(client.video_frame_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[STREAM] frame error: {e}")
                continue
            latest.put(frame.to_ndarray(format="bgr24"))
            while not client.audio_frame_queue.empty():
                await client.audio_frame_queue.get()

    options = types.SimpleNamespace(hostname=hostname, sdp_port=31102,
                                    sdp_filename="h264.sdp", cam_ssl_cert=False)
    t = threading.Thread(target=start_webrtc,
                         args=(shutdown_flag, options, robot.user_token, grab_frames),
                         daemon=True, name="WebRTC")
    t.start()
    return t


def load_calibration():
    """Returns (distance_k, hfov_deg, centre_offset, bearing_sign) or Nones."""
    if not os.path.exists(CALIBRATION_FILE):
        print(f"[CAL] Not found: {CALIBRATION_FILE}")
        return None, None, 0.0, 1
    try:
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
        return (float(cal["distance_k"]), float(cal["hfov_deg"]),
                float(cal.get("centre_offset", 0.0)), int(cal.get("bearing_sign", 1)))
    except Exception as e:
        print(f"[CAL] Could not read {CALIBRATION_FILE}: {e}")
        return None, None, 0.0, 1


def detect_face(detector, frame):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.detections:
        return None
    b = max((d.bounding_box for d in res.detections), key=lambda bb: bb.width * bb.height)
    return (b.origin_x, b.origin_y, b.width, b.height)


def detect_faces_all(detector, frame, max_faces=6):
    """Every face in the frame, biggest first."""
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    boxes = [(d.bounding_box.origin_x, d.bounding_box.origin_y,
              d.bounding_box.width, d.bounding_box.height) for d in res.detections]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_faces]


def hands_in(result, crop_box):
    """[(gesture, score, (centre_x, centre_y, size, wrist_x, wrist_y))] for each
    hand found in a crop, with positions converted to full-frame pixels."""
    x0, y0, x1, y1 = crop_box
    cw, ch = x1 - x0, y1 - y0
    out = []
    for i, gestures in enumerate(result.gestures):
        if not gestures or i >= len(result.hand_landmarks):
            continue
        xs = [x0 + lm.x * cw for lm in result.hand_landmarks[i]]
        ys = [y0 + lm.y * ch for lm in result.hand_landmarks[i]]
        hand = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                max(max(xs) - min(xs), max(ys) - min(ys)), xs[0], ys[0])  # [0] = wrist
        out.append((gestures[0].category_name, gestures[0].score, hand))
    return out


def run_calibration(next_frame, face_detector, w):
    """Interactive: person stands at measured spots, we record face size and
    position, then save distance_k (m * px) and the camera's horizontal view."""

    def measure(seconds=2.0):
        heights, centres = [], []
        end = time.time() + seconds
        while time.time() < end:
            frame, _ = next_frame()
            if frame is None:
                break
            bb = detect_face(face_detector, frame)
            if bb:
                heights.append(bb[3])
                centres.append((bb[0] + bb[2] / 2.0) / w - 0.5)
        if not heights:
            return None, None
        return float(np.median(heights)), float(np.median(centres))

    print("\nCALIBRATION. Measure distances on the floor from Spot's front with a tape.")
    ks, centres = [], []
    for d in (2.0, 3.0):
        input(f"\nStand straight in front of Spot at {d:.1f} m, facing it, then press Enter...")
        h, c = measure()
        if h is None:
            print("  No face found. Check the framing and try again later.")
            continue
        ks.append(d * h)
        centres.append(c)
        print(f"  face height {h:.0f} px, centre offset {c:+.3f} -> k = {d * h:.0f}")
    lateral, forward = 1.0, 2.0
    input(f"\nStand {forward:.1f} m in front of Spot and {lateral:.1f} m to Spot's RIGHT, "
          "then press Enter...")
    h, c = measure()
    if not ks or h is None:
        print("Calibration incomplete; nothing saved.")
        return
    c0 = float(np.mean(centres))      # where "straight ahead" lands in the image
    if abs(c - c0) < 0.02:
        print("The side position barely moved in the image; nothing saved. "
              "Check you moved 1 m to the side.")
        return
    angle = math.degrees(math.atan2(lateral, forward))
    hfov = angle / abs(c - c0)
    sign = 1 if c > c0 else -1        # +1: person on Spot's right appears on image right
    cal = {"distance_k": float(np.mean(ks)), "hfov_deg": hfov,
           "centre_offset": c0, "bearing_sign": sign,
           "measured": time.strftime("%Y-%m-%d %H:%M")}
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\nSaved {CALIBRATION_FILE}: distance_k={cal['distance_k']:.0f}, "
          f"hfov={hfov:.1f} deg, straight ahead at image offset {c0:+.3f}, "
          f"bearing sign {sign:+d}.")
    if abs(c0) > 0.1:
        print("Note: 'straight ahead' is well off the image centre. The PTZ is probably "
              "not pointing straight forward; consider re-aiming it and re-running this.")


def add_common_args(p):
    p.add_argument("hostname", nargs="?", help="Robot IP (omit with --video)")
    p.add_argument("--video", help="Run on a recorded .mp4 instead of the live stream")
    p.add_argument("--drive", choices=["none", "mock", "real"], default="none",
                   help="none: print intents only; mock: print Spot commands; real: move Spot")
    p.add_argument("--no-walk", action="store_true", help="Disable walking")
    p.add_argument("--no-turn", action="store_true", help="Disable turning in place")
    p.add_argument("--speed-scale", type=float, default=0.5,
                   help="Fraction of the speed limits (default 0.5)")
    p.add_argument("--calibrate", action="store_true",
                   help="Measure distance/direction and save spot_calibration.json")
    p.add_argument("--duration", type=float, default=0, help="Seconds (0 = until stopped)")
    p.add_argument("--show", action="store_true", help="Preview window")
    p.add_argument("--model", default=os.path.join(HERE, "gesture_recognizer.task"))
    p.add_argument("--face-model", default=os.path.join(HERE, "blaze_face_full_range.tflite"))
    p.add_argument("--face-conf", type=float, default=0.5)
    p.add_argument("--screen", default="mech_full",
                   help="Spot CAM view to stream (default mech_full = PTZ only)")
    p.add_argument("--crop-ptz", action="store_true",
                   help="Fallback: cut the PTZ view out of the stacked PTZ + 360 layout")
    p.add_argument("--debug", action="store_true")
    return p


def run(args, control, skeleton=None):
    """Main loop shared by spot_gesture_basic.py and spot_gesture_tag.py.
    `control` decides who the person in control is (CentreLock or TagControl).
    `skeleton` (optional, spot_skeleton.SkeletonCheck): a hand only counts if
    it sits at the wrist of the controller's own body skeleton."""
    if not args.video and not args.hostname:
        sys.exit("Give the robot IP, or --video FILE.mp4")
    if args.video and args.drive == "real":
        sys.exit("--drive real needs the live stream, not --video")
    for path in (args.model, args.face_model):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            sys.exit(f"Missing or empty model file: {path}")

    recognizer = vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=args.model),
            # IMAGE mode: find hands from scratch every frame (VIDEO mode got
            # stuck tracking the resting hand). Two hands so the resting hand
            # can't hide the gesturing one.
            running_mode=vision.RunningMode.IMAGE,
            num_hands=getattr(args, "num_hands", 2)))
    face_detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=args.face_model),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=args.face_conf))

    # ---- Frame source ----
    shutdown_flag = threading.Event()
    cap, robot = None, None
    latest = LatestFrame()
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Could not open {args.video}")
        print(f"[SRC] Recording: {args.video}")
    else:
        print(f"[SRC] Connecting to Spot at {args.hostname} ...")
        robot = connect_robot(args.hostname)
        try:
            set_ptz_only_view(robot, args.screen)
        except Exception as e:
            print(f"[CAM] Could not switch the view to '{args.screen}': {e}")
        start_spot_stream(robot, args.hostname, latest, shutdown_flag)

    def ptz_only(frame):
        if not args.crop_ptz:
            return frame
        fh, fw = frame.shape[:2]
        x0, y0, x1, y1 = PTZ_PANE
        return np.ascontiguousarray(frame[int(y0 * fh):int(y1 * fh),
                                          int(x0 * fw):int(x1 * fw)])

    def next_frame():
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                return None, None
            return ptz_only(frame), cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame = latest.get(timeout=20.0)
        return (ptz_only(frame), time.time()) if frame is not None else (None, None)

    frame, now = next_frame()
    if frame is None:
        print("ERROR: no video frame received. Check the compositor (mech_full), "
              "the bitrate (2 Mbps) and Spot's Wi-Fi.")
        shutdown_flag.set()
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"[SRC] Frame size: {w}x{h}")

    if args.calibrate:
        run_calibration(next_frame, face_detector, w)
        shutdown_flag.set()
        return

    # ---- Distance / direction calibration ----
    distance_k, hfov_deg, centre_offset, bearing_sign = load_calibration()
    calibrated = distance_k is not None
    if calibrated:
        print(f"[CAL] distance_k={distance_k:.0f}, hfov={hfov_deg:.1f} deg")
    else:
        print("[CAL] No spot_calibration.json: distance/direction unknown, so "
              "walking and turning are DISABLED. Run --calibrate first.")

    # ---- Movement side (spot_adapter.py) ----
    adapter, motion = None, None
    if args.drive != "none":
        from spot_adapter import SpotAdapter, MotionDispatcher, ReactionAction
        adapter = SpotAdapter(
            hostname=args.hostname, mock=(args.drive == "mock"),
            allow_walking=calibrated and not args.no_walk,
            allow_turning=calibrated and not args.no_turn,
            speed_scale=args.speed_scale, robot=robot)
        motion = MotionDispatcher()

        def why_off(flag_off, flag_name):
            if not calibrated:
                return "OFF (no spot_calibration.json in this folder; run --calibrate)"
            if flag_off:
                return f"OFF ({flag_name} was given)"
            return "ON"
        print(f"[DRIVE] Turning to face: {why_off(args.no_turn, '--no-turn')}")
        print(f"[DRIVE] Walking:         {why_off(args.no_walk, '--no-walk')}")
        print(f"[DRIVE] Speed scale:     {args.speed_scale} "
              f"(forward {0.3 * args.speed_scale:.2f} m/s, backward {0.2 * args.speed_scale:.2f} m/s)")
        if args.drive == "real":
            print("\n*** REAL MOTION: E-stop operator ready, area clear. "
                  "Starting in 5 s (Ctrl+C to abort) ***")
            t_wait = time.time()
            while time.time() - t_wait < 5.0:
                next_frame()           # keep the stream fresh
        adapter.start()

        def run_intent(intent, extra, stop_event):
            if extra == "ACK":
                adapter.checkin_ack(stop_event=stop_event)
            elif extra == "SCAN":
                adapter.caution_scan(stop_event=stop_event)
            if stop_event.is_set():
                return
            adapter.hold_for(ReactionAction(intent), HOLD_S, stop_event=stop_event)
            if not stop_event.is_set():
                adapter.stop()

    print("Thumb up=ENGAGE  Pointing up=CHECK_IN  Open palm=CAUTION  Thumb down=DE_ESCALATE")
    print(control.intro)
    print({"none": "Spot will NOT move (intents printed only).",
           "mock": "Mock drive: SDK commands printed, Spot will NOT move.",
           "real": "REAL drive: Spot moves."}[args.drive] + "\n")

    follower = CropFollower(w, h)
    voter = GestureVoter()
    all_faces = []
    t_start = now
    active, last_dispatch, last_gesture = "NEUTRAL", -1e9, -1e9
    streak, caution_scanned = 0, False
    neutral_since = last_pulse = now
    counts = Counter()
    frames = face_checks = face_hits = 0
    recog_total = 0.0
    last_report = now
    last_label = None

    def decide(intent, extra=None, note=""):
        counts[intent] += 1
        d = adapter.state.person_distance if adapter else None
        dist_txt = f"   person {d:.2f} m" if (d is not None and calibrated) else ""
        print(f"[{now - t_start:6.1f}s] INTENT -> {intent}{note}"
              f"{' + ' + extra if extra else ''}{dist_txt}")
        if motion is not None:
            motion.dispatch(run_intent, intent, extra)

    def recognize(crop):
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    try:
        while frame is not None:
            frames += 1

            # 1. Faces -> who is in control, crop position, distance, direction
            if frames % FACE_STRIDE == 1:
                face_checks += 1
                all_faces = detect_faces_all(face_detector, frame)
                ctrl_bb = control.update(frame, all_faces, now, w, h)
                if ctrl_bb is not None:
                    follower.update(ctrl_bb)
                    face_hits += 1
                    if adapter is not None and calibrated:
                        adapter.state.person_distance = distance_k / max(ctrl_bb[3], 1)
                        offset = (ctrl_bb[0] + ctrl_bb[2] / 2.0) / w - 0.5 - centre_offset
                        adapter.state.person_bearing = math.radians(
                            bearing_sign * offset * hfov_deg)
            in_control = control.present(now)
            if adapter is not None:
                adapter.state.person_visible = in_control

            # 2. Gesture: only the person in control, only their own hands
            roi, box = follower.crop(frame)
            label = score = None
            if skeleton is not None:
                skeleton.update(frame, control, now, in_control)
            if in_control and roi.size:
                t_r = time.time()
                result = recognize(roi)
                recog_total += time.time() - t_r
                mine = []
                for g, sc, hand in hands_in(result, box):
                    if not control.hand_ok(hand, all_faces):
                        continue
                    if skeleton is not None and not skeleton.hand_ok(hand, now, g):
                        continue
                    mine.append((g, sc))
                if mine:
                    # Prefer one of our gestures over "None" (e.g. the resting hand)
                    mapped = [m for m in mine if m[0] in GESTURE_TO_INTENT]
                    label, score = max(mapped or mine, key=lambda m: m[1])
            for msg in control.events:
                print(f"[{now - t_start:6.1f}s] [CTRL] {msg}")
            control.events.clear()
            vote = GESTURE_TO_INTENT.get(label) if (score or 0) >= MIN_SCORE else None
            voter.add(now, vote if in_control else None)
            if args.debug and label != last_label:
                print(f"[{now - t_start:6.1f}s] frame gesture: {label} "
                      f"{'' if score is None else f'({score:.2f})'}")
                last_label = label

            # 3. Decision rules
            held = voter.decide(now)
            if held is not None:
                last_gesture = now
                new = held != active
                if new or now - last_dispatch >= REDISPATCH_S:
                    streak = 1 if new else streak + 1
                    if new:
                        caution_scanned = False
                    extra = None
                    if held == "CHECK_IN" and (new or CHECKIN_ACK_EVERY_DISPATCH):
                        extra = "ACK"
                    elif held == "CAUTION" and streak >= 2 and not caution_scanned:
                        extra, caution_scanned = "SCAN", True
                    decide(held, extra, " (new)" if new else " (held)")
                    active, last_dispatch = held, now
            elif active != "NEUTRAL" and now - last_gesture > RELEASE_S:
                decide("NEUTRAL", note=" (gesture released)")
                active, last_dispatch = "NEUTRAL", now
                neutral_since = last_pulse = now
                streak = 0
            elif (active == "NEUTRAL" and in_control
                  and now - last_pulse >= NEUTRAL_PULSE_EVERY_S
                  and now - neutral_since >= NEUTRAL_PULSE_EVERY_S):
                print(f"[{now - t_start:6.1f}s] idle -> NEUTRAL + BOB")
                last_pulse = now
                if motion is not None:
                    motion.dispatch(adapter.neutral_pulse)

            # 4. Status line
            if now - last_report >= 5.0:
                extra_txt = ""
                if adapter is not None and calibrated and in_control:
                    extra_txt = (f", person {adapter.state.person_distance:.2f} m at "
                                 f"{math.degrees(adapter.state.person_bearing):+.0f} deg")
                if skeleton is not None:
                    extra_txt += ", " + skeleton.status(now)
                print(f"[{now - t_start:6.1f}s] status: gesture model "
                      f"{1000 * recog_total / max(frames, 1):.0f} ms/frame, person in control "
                      f"seen in {face_hits}/{face_checks} checks, {len(all_faces)} face(s) in view, "
                      f"{control.status(now)}, intent {active}{extra_txt}")
                face_hits = face_checks = 0
                last_report = now

            # 5. Optional preview
            if args.show:
                view = frame.copy()
                for bx, by, bw, bh in all_faces:              # everyone: grey
                    cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (160, 160, 160), 1)
                control.draw(view)
                if skeleton is not None:
                    skeleton.draw(view)
                if in_control:
                    x0, y0, x1, y1 = box                     # gesture area: blue
                    cv2.rectangle(view, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 2)
                    bx, by, bw, bh = control.target          # person in control: green
                    cv2.rectangle(view, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
                cv2.putText(view, f"{label or '-'}  ->  {active}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(view, control.status(now), (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0) if in_control else (0, 165, 255), 2)
                cv2.imshow("Spot gestures (q quit, r reset)", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    control.reset(now)

            if args.duration > 0 and now - t_start >= args.duration:
                break
            frame, now = next_frame()

    except KeyboardInterrupt:
        print("\n  Stopped (Ctrl+C).")
    finally:
        if motion is not None:
            motion.stop_all()
        if adapter is not None:
            adapter.shutdown()
        shutdown_flag.set()
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()
        recognizer.close()
        face_detector.close()
        if skeleton is not None:
            skeleton.close()

    print("\n================ SUMMARY ================")
    print(f"Frames processed: {frames}, gesture model "
          f"{1000 * recog_total / max(frames, 1):.0f} ms/frame")
    if skeleton is not None:
        print(skeleton.summary())
    print("Intent decisions:")
    for name, n in counts.most_common():
        print(f"    {name:<12} {n}")
    if not counts:
        print("    none")
