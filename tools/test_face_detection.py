#!/usr/bin/env python3
"""
============================================================================
test_face_detection.py  —  Diagnose why the person tracker isn't finding faces
============================================================================
Captures ONE frame from the MentorPi camera and runs MediaPipe on it with
several different settings so we can figure out which combination works.

Also saves the raw frame + an annotated frame with any detections drawn,
so you can visually verify what the camera actually sees.

Usage (on the Pi, with venv active):
    python3 test_face_detection.py
============================================================================
"""

import cv2
import mediapipe as mp

STREAM_URL = "http://127.0.0.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image"
RAW_OUT   = "/tmp/test_frame.jpg"
ANNOT_OUT = "/tmp/test_frame_annotated.jpg"


def try_detection(rgb_frame, model_selection, min_confidence):
    """Try face detection with specific settings, return number of faces found."""
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_confidence,
    ) as detector:
        result = detector.process(rgb_frame)
        if result.detections:
            return len(result.detections), result.detections
        return 0, []


def main():
    print(f"[TEST] Opening {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"[ERROR] Could not open HTTP stream.")
        return

    # Grab a fresh frame (throw away the first couple, they might be stale)
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Could not read a frame.")
        return

    h, w = frame.shape[:2]
    print(f"[TEST] Frame size: {w}x{h}")

    # Save raw frame for visual inspection
    cv2.imwrite(RAW_OUT, frame)
    print(f"[TEST] Saved raw frame to {RAW_OUT}")

    # Run detection with different settings
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(f"\n[TEST] Trying different MediaPipe settings on this single frame:\n")

    settings_to_try = [
        (0, 0.5, "Short-range, confidence 0.5 (current defaults)"),
        (0, 0.3, "Short-range, confidence 0.3 (more permissive)"),
        (0, 0.1, "Short-range, confidence 0.1 (very permissive)"),
        (1, 0.5, "Full-range, confidence 0.5"),
        (1, 0.3, "Full-range, confidence 0.3"),
        (1, 0.1, "Full-range, confidence 0.1"),
    ]

    best_result = None
    best_settings = None
    for model_sel, conf, label in settings_to_try:
        num, detections = try_detection(rgb, model_sel, conf)
        print(f"  {label}: {num} face(s)")
        if num > 0 and best_result is None:
            best_result = detections
            best_settings = (model_sel, conf, label)

    # Annotate the best result if we found any faces
    if best_result:
        print(f"\n[TEST] Best result: {best_settings[2]} — {len(best_result)} face(s)")
        annotated = frame.copy()
        for det in best_result:
            b = det.location_data.relative_bounding_box
            x = int(b.xmin * w)
            y = int(b.ymin * h)
            bw = int(b.width * w)
            bh = int(b.height * h)
            conf = det.score[0] if det.score else 0.0

            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            cv2.putText(annotated, f"{conf:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            print(f"       Face at pixel ({x}, {y}) size {bw}x{bh}, confidence {conf:.3f}")

        cv2.imwrite(ANNOT_OUT, annotated)
        print(f"[TEST] Saved annotated frame to {ANNOT_OUT}")
    else:
        print(f"\n[TEST] NO settings detected a face.")
        print(f"       Likely causes:")
        print(f"       - Face is not in the frame (check camera angle)")
        print(f"       - Face is too small/blurry")
        print(f"       - Lighting is very poor")
        print(f"       - Face is at an extreme angle (side profile, looking down)")

    print(f"\n[TEST] View the saved images:")
    print(f"       scp pi@<pi-ip>:{RAW_OUT} ~/Downloads/")
    print(f"       scp pi@<pi-ip>:{ANNOT_OUT} ~/Downloads/")


if __name__ == "__main__":
    main()
