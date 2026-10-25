#!/usr/bin/env python3
"""
============================================================================
gesture_check.py  —  Phase 0 feasibility check for gesture-driven intents
============================================================================
Reads the MentorPi's existing HTTP camera stream, runs MediaPipe's Gesture
Recognizer on every frame (or every Nth with --stride), and prints:

    - the camera frame size actually received
    - each recognized gesture with its confidence, which hand, and the
      hand's width in pixels (a stand-in for distance in the range test)
    - at the end: frames processed, average FPS, average recognizer time
      per frame, and how often each gesture was seen

Nothing here moves the robot. It only reads the camera.

Usage (on the Pi, in the venv the pipeline already uses):
    source ~/va_pipeline_venv/bin/activate
    cd ~/MentorPi-code/gesture
    python3 gesture_check.py                  # 30 s, every frame
    python3 gesture_check.py --duration 60
    python3 gesture_check.py --stride 2       # every other frame (less CPU)

Needs gesture_recognizer.task next to this script (or --model PATH).
============================================================================
"""

import argparse
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"ERROR: MediaPipe Tasks API not available: {e}")
    print("  The Gesture Recognizer needs mediapipe 0.10 or newer.")
    print("  Check with: python3 -c \"import mediapipe; print(mediapipe.__version__)\"")
    sys.exit(1)

DEFAULT_STREAM_URL = (
    "http://127.0.0.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image"
)
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gesture_recognizer.task")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 gesture recognizer check")
    p.add_argument("--stream-url", default=DEFAULT_STREAM_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--duration", type=float, default=30.0,
                   help="Seconds to run (default 30)")
    p.add_argument("--stride", type=int, default=1,
                   help="Run the recognizer on every Nth frame (default 1)")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="Only print gestures at or above this confidence")
    p.add_argument("--crop", type=float, default=1.0,
                   help="Run the recognizer on the central part of the frame "
                        "only, e.g. 0.5 = the middle 50%% of width and height. "
                        "Makes hands bigger relative to what the model sees. "
                        "Stand in the middle of the camera view when using it.")
    p.add_argument("--crop-y", type=float, default=0.5,
                   help="Where the crop window sits vertically: 0 = top of the "
                        "frame, 0.5 = middle (default), 1 = bottom. The "
                        "MentorPi camera looks up, so people appear low in "
                        "the frame; try 0.8 to 1.0.")
    p.add_argument("--expect", default=None,
                   help="Accuracy test: the one gesture you will hold for the "
                        "whole run (Thumb_Up, Thumb_Down, Open_Palm, "
                        "Pointing_Up, Closed_Fist, Victory, ILoveYou). The "
                        "summary then reports how often it was recognized.")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to wait before measuring, to get into position")
    p.add_argument("--snapshot", default=None, metavar="PATH",
                   help="Save one image of what the recognizer saw (after "
                        "cropping), halfway through the run, with any hand "
                        "points drawn on it. Use to check framing and hand size.")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"MediaPipe version: {mp.__version__}")

    if not os.path.exists(args.model):
        print(f"ERROR: model file not found: {args.model}")
        print("  Download it first (see the instructions you were given), "
              "or pass --model PATH.")
        sys.exit(1)

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(args.stream_url)
    if not cap.isOpened():
        print(f"ERROR: could not open camera stream: {args.stream_url}")
        print("  Is web_video_server running in the MentorPi container?")
        sys.exit(1)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: stream opened but no frame arrived.")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"Camera frame size: {w}x{h} (w x h)")
    crop = min(max(args.crop, 0.1), 1.0)
    cw, ch = int(w * crop), int(h * crop)
    crop_y = min(max(args.crop_y, 0.0), 1.0)
    cx0, cy0 = (w - cw) // 2, int((h - ch) * crop_y)
    if crop < 1.0:
        print(f"Cropping to a {cw}x{ch} region (top-left at {cx0},{cy0}) "
              f"before recognition.")
    print(f"Running for {args.duration:.0f}s, recognizer on every "
          f"{args.stride} frame(s). Show gestures to the camera.\n")

    if args.delay > 0:
        print(f"Get into position: starting in {args.delay:.0f}s...")
        t_wait = time.time()
        while time.time() - t_wait < args.delay:
            cap.read()   # keep draining the stream so frames stay fresh
        print("Go.\n")

    counts = Counter()
    frame_labels = Counter()   # one label per frame in which a hand was found
    snapshot_saved = False
    frames_read = 0
    frames_run = 0
    infer_total = 0.0
    last_printed = None
    t_start = time.time()
    t0_ms = int(t_start * 1000)

    while time.time() - t_start < args.duration:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        frames_read += 1
        if frames_read % args.stride != 0:
            continue

        if crop < 1.0:
            frame = frame[cy0:cy0 + ch, cx0:cx0 + cw]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs strictly increasing timestamps in ms
        ts = int(time.time() * 1000) - t0_ms + frames_run
        t_inf = time.time()
        result = recognizer.recognize_for_video(mp_image, ts)
        infer_total += time.time() - t_inf
        frames_run += 1

        # Per-frame label for the accuracy test: the top label of the most
        # confident hand, counted whenever any hand was found (no threshold).
        tops = [g[0] for g in result.gestures if g]
        if tops:
            frame_labels[max(tops, key=lambda c: c.score).category_name] += 1

        seen = []
        for i, gestures in enumerate(result.gestures):
            if not gestures:
                continue
            top = gestures[0]
            if top.score < args.min_score:
                continue
            hand = result.handedness[i][0].category_name if result.handedness else "?"
            xs = [lm.x for lm in result.hand_landmarks[i]]
            hand_px = (max(xs) - min(xs)) * cw   # px in the original frame
            counts[top.category_name] += 1
            seen.append(f"{top.category_name} ({top.score:.2f}, {hand} hand, "
                        f"{hand_px:.0f}px wide)")

        if args.snapshot and not snapshot_saved and \
                time.time() - t_start >= args.duration / 2:
            snap = frame.copy()
            for hand_lms in result.hand_landmarks:
                for lm in hand_lms:
                    cv2.circle(snap, (int(lm.x * cw), int(lm.y * ch)), 2,
                               (0, 255, 0), -1)
            cv2.imwrite(args.snapshot, snap)
            snapshot_saved = True
            print(f"[{time.time() - t_start:5.1f}s] snapshot saved to {args.snapshot}")

        summary = "; ".join(seen) if seen else "no gesture"
        # Print only when what we see changes, so the output stays readable
        if summary != last_printed:
            print(f"[{time.time() - t_start:5.1f}s] {summary}")
            last_printed = summary

    elapsed = time.time() - t_start
    cap.release()
    recognizer.close()

    print("\n================ SUMMARY ================")
    print(f"Camera frame size:        {w}x{h}")
    print(f"Frames read from stream:  {frames_read}  "
          f"({frames_read / elapsed:.1f} fps)")
    print(f"Frames run through model: {frames_run}  "
          f"({frames_run / elapsed:.1f} fps)")
    if frames_run:
        print(f"Avg recognizer time:      {1000 * infer_total / frames_run:.0f} ms/frame")
    print("Gesture detections (score >= "
          f"{args.min_score}):")
    if counts:
        for name, n in counts.most_common():
            print(f"    {name:<14} {n}")
    else:
        print("    none")

    if args.expect:
        hand_frames = sum(frame_labels.values())
        print(f"\nAccuracy test, holding {args.expect}:")
        print(f"    Frames with a hand found: {hand_frames} of {frames_run} "
              f"({100 * hand_frames / max(frames_run, 1):.0f}%)")
        if hand_frames:
            for name, n in frame_labels.most_common():
                mark = "  <- correct" if name == args.expect else ""
                print(f"    {name:<14} {100 * n / hand_frames:5.1f}% of hand frames{mark}")


if __name__ == "__main__":
    main()
