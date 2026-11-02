#!/usr/bin/env python3
"""
============================================================================
qr_priority_live.py  —  Live webcam check for QR-badge gesture priority
============================================================================
Opens a window on a webcam (or the MentorPi stream) and draws every frame:

    - every face (grey box); the priority person's face (orange, PRIORITY)
    - every allowed QR code (green outline if linked to a face, red if not)
    - the imaginary line from a face's chin to its QR code
    - each hand: green if it belongs to the priority person, red if not,
      grey while nobody has priority
    - top left: the gesture and its intent. QR first: while a QR is linked
      to a face, only that person's gesture shows and its intent works.
      With no QR, any hand's gesture still shows but the intent reads
      "locked (no QR)".

Nothing here moves the robot. The hold check and motion from
gesture/run_gesture_mentorpi.py are not included: this is for checking
and tuning the rules in robot/qr_priority.py.

Usage (in a venv with mediapipe 0.10.21 + opencv-python):
    cd ~/MentorPi-code/tools
    python3 qr_priority_live.py --make-qr /tmp/priority_qr.png  # test badge
    python3 qr_priority_live.py                 # webcam 0
    python3 qr_priority_live.py --camera 1      # another camera
    python3 qr_priority_live.py --any-qr        # accept any QR payload
    python3 qr_priority_live.py --show-zones    # draw where a QR must be

Show the QR on a phone or print it, and hold it at chest height below your
face. The image is not mirrored (QR codes don't read when flipped), so
moving left moves you right on screen. Press q to quit.
============================================================================
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"ERROR: MediaPipe Tasks API not available: {e}")
    sys.exit(1)

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from qr_priority import (LINE_MAX_DOWN_FACES, LINE_MAX_SIDEWAYS_FACES,
                         PRIORITY_PAYLOADS, Face, PriorityState, QRDetector,
                         assign_hand_to_face, gesture_allowed)

DEFAULT_MODEL = os.path.join(_MENTORPI_ROOT, "gesture", "gesture_recognizer.task")

# Same mapping and threshold as run_gesture_mentorpi.py
GESTURE_TO_INTENT = {
    "Thumb_Up":    "ENGAGE",
    "Pointing_Up": "CHECK_IN",
    "Open_Palm":   "CAUTION",
    "Thumb_Down":  "DE_ESCALATE",
}
MIN_SCORE = 0.6

# BGR colours
GREY   = (160, 160, 160)
ORANGE = (0, 165, 255)
GREEN  = (0, 200, 0)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)
FONT   = cv2.FONT_HERSHEY_SIMPLEX
# HERSHEY_SIMPLEX is ~22 px tall at scale 1.0, so this gives 12 px text
HUD_SCALE = 12 / 22


def parse_args():
    p = argparse.ArgumentParser(description="Live QR-priority check")
    p.add_argument("--camera", default="0",
                   help="Webcam index (0, 1, ...) or a camera stream URL")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--payload", action="append", default=None,
                   help="Allowed QR text (repeat for more). "
                        f"Default: {', '.join(sorted(PRIORITY_PAYLOADS))}")
    p.add_argument("--any-qr", action="store_true",
                   help="Accept any QR code, whatever it says")
    p.add_argument("--qr-stride", type=int, default=2,
                   help="Look for QR codes every Nth frame (default 2)")
    p.add_argument("--qr-hold", type=float, default=0.5,
                   help="Seconds priority lasts after the QR was last seen "
                        "(default 0.5, just enough to ride out a missed "
                        "read). 0 = only while the QR is read; needs "
                        "--qr-stride 1.")
    p.add_argument("--face-conf", type=float, default=0.5,
                   help="Face detection confidence threshold (default 0.5)")
    p.add_argument("--num-hands", type=int, default=4,
                   help="Max hands the recognizer looks for (default 4)")
    p.add_argument("--show-zones", action="store_true",
                   help="Draw the area under each face where a QR must be")
    p.add_argument("--make-qr", default=None, metavar="PATH",
                   help="Write a QR image with the (first) allowed payload "
                        "to PATH and exit")
    return p.parse_args()


def make_qr(path, text):
    img = cv2.QRCodeEncoder.create().encode(text)
    scale = max(1, 600 // img.shape[0])
    img = cv2.resize(img, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    border = 4 * scale   # white quiet zone so detectors can find it
    img = cv2.copyMakeBorder(img, border, border, border, border,
                             cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, img)
    print(f"Wrote QR for '{text}' to {path} ({img.shape[1]}x{img.shape[0]} px)")


def detect_faces(detector, frame_rgb):
    h, w = frame_rgb.shape[:2]
    results = detector.process(frame_rgb)
    faces = []
    for det in results.detections or []:
        b = det.location_data.relative_bounding_box
        faces.append(Face(b.xmin * w, b.ymin * h, b.width * w, b.height * h,
                          det.score[0] if det.score else 0.0))
    return faces


def pt(xy):
    return (int(round(xy[0])), int(round(xy[1])))


def draw_label(img, text, org, colour, scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, scale, colour, 1, cv2.LINE_AA)


def draw_faces(img, faces, priority, show_zones):
    for f in faces:
        is_p = f is priority.face
        colour = ORANGE if is_p else GREY
        cv2.rectangle(img, pt((f.x, f.y)), pt((f.x + f.w, f.y + f.h)),
                      colour, 3 if is_p else 1)
        if is_p:
            draw_label(img, "PRIORITY", pt((f.x, f.y - 4)), ORANGE)
        if show_zones:
            cx, cy = f.chin
            dx = LINE_MAX_SIDEWAYS_FACES * f.w
            cv2.rectangle(img, pt((cx - dx, cy)),
                          pt((cx + dx, cy + LINE_MAX_DOWN_FACES * f.h)),
                          colour, 1)


def draw_links(img, links):
    for link in links:
        colour = GREEN if link.face is not None else RED
        if link.qr.corners is not None:
            cv2.polylines(img, [link.qr.corners.astype(np.int32).reshape(-1, 1, 2)],
                          True, colour, 2)
        q = pt(link.qr.center)
        if link.face is not None:
            c = pt(link.face.chin)
            cv2.line(img, c, q, ORANGE, 2, cv2.LINE_AA)
            cv2.circle(img, c, 4, ORANGE, -1)
            cv2.circle(img, q, 4, ORANGE, -1)
        draw_label(img, link.qr.text if link.face else f"{link.qr.text} (no face)",
                   (q[0] + 8, q[1]), colour, 0.45)


def draw_hands(img, result, faces, priority):
    """Draw every hand. With a priority person: green = theirs, red =
    ignored. Without one: grey (gesture seen, no intent).
    Returns (gesture, score) of the most confident hand that counts: the
    priority person's hands if there is one, otherwise any hand.
    (None, None) if no counted hand shows a gesture."""
    h, w = img.shape[:2]
    best = (None, None)
    for i, lms in enumerate(result.hand_landmarks):
        xs = np.array([lm.x * w for lm in lms])
        ys = np.array([lm.y * h for lm in lms])
        center = (float(xs.mean()), float(ys.mean()))
        face = assign_hand_to_face(center, faces)
        allowed = gesture_allowed(face, priority)
        if not priority.active:
            colour = GREY
        else:
            colour = GREEN if allowed else RED

        for x, y in zip(xs, ys):
            cv2.circle(img, pt((x, y)), 2, colour, -1)
        cv2.rectangle(img, pt((xs.min(), ys.min())), pt((xs.max(), ys.max())),
                      colour, 1)
        if face is not None:
            cv2.line(img, pt(center), pt(face.center), colour, 1, cv2.LINE_AA)

        if allowed and i < len(result.gestures) and result.gestures[i]:
            g = result.gestures[i][0]
            if g.category_name != "None" and \
                    (best[1] is None or g.score > best[1]):
                best = (g.category_name, g.score)
    return best


def draw_hud(img, gesture, score, priority):
    """Top-left: the gesture, and the intent it maps to. The intent only
    works while a QR is linked to a face; without one it reads "locked".
    Unmapped or low-confidence gestures are NEUTRAL, as in
    run_gesture_mentorpi.py."""
    g_text = "none" if gesture is None else f"{gesture} ({score:.2f})"
    draw_label(img, f"Gesture: {g_text}", (8, 20), WHITE, HUD_SCALE)
    if not priority.active:
        draw_label(img, "Intent:  locked (no QR)", (8, 42), RED, HUD_SCALE)
        return
    intent = "NEUTRAL"
    if gesture is not None and score >= MIN_SCORE:
        intent = GESTURE_TO_INTENT.get(gesture, "NEUTRAL")
    draw_label(img, f"Intent:  {intent}", (8, 42),
               WHITE if intent == "NEUTRAL" else GREEN, HUD_SCALE)


def main():
    args = parse_args()
    if args.any_qr:
        payloads = None
    else:
        payloads = set(args.payload) if args.payload else PRIORITY_PAYLOADS

    if args.make_qr:
        make_qr(args.make_qr, sorted(payloads)[0] if payloads else "MENTORPI-PRIORITY")
        return

    if not os.path.exists(args.model) or os.path.getsize(args.model) == 0:
        print(f"ERROR: gesture model missing or empty: {args.model}")
        sys.exit(1)

    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,   # full range, for people 1-2 m+ away
        min_detection_confidence=args.face_conf)
    recognizer = vision.GestureRecognizer.create_from_options(
        vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=args.num_hands,
        ))
    qr_detector = QRDetector(payloads)
    priority = PriorityState(hold_s=args.qr_hold)

    # A bare number means a local webcam index (e.g. 0 on a laptop).
    src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.camera}")
        print("  On macOS, allow camera access for your terminal in System "
              "Settings > Privacy & Security > Camera, then restart it.")
        sys.exit(1)

    print("Allowed QR payloads:",
          "any" if payloads is None else ", ".join(sorted(payloads)))
    t0 = time.time()
    last_ts = -1
    frame_i = 0
    last_links = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: no frame from camera")
                break
            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = detect_faces(face_detector, rgb)
            codes = qr_detector.detect(frame) \
                if frame_i % args.qr_stride == 0 else None
            links, events = priority.update(now, faces, codes)
            if codes is not None:
                last_links = links   # keep drawing them between QR checks
            for ev in events:
                if ev == "granted":
                    print(f"[PRIORITY] granted ({priority.text})")
                else:
                    print("[PRIORITY] released")

            ts = max(int((now - t0) * 1000), last_ts + 1)
            last_ts = ts
            result = recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            draw_faces(frame, faces, priority, args.show_zones)
            draw_links(frame, last_links)
            gesture, score = draw_hands(frame, result, faces, priority)
            draw_hud(frame, gesture, score, priority)
            cv2.imshow("QR priority (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_i += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        face_detector.close()


if __name__ == "__main__":
    main()
