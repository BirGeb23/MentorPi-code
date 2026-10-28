"""Run the modular camera, QR tracking, gesture, and robot-control pipeline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time

import cv2

from camera_feed import CameraFeed, DEFAULT_ROBOT_STREAM
from gesture_recognition import GestureIntentVoter, GestureRecognizer
from qr_tracker import QRTracker
from robot_motion import RobotMotionController


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "gesture_recognizer.task")
REDISPATCH_S = 2.5
RELEASE_S = 1.5


def parse_args():
    parser = argparse.ArgumentParser(description="Modular MentorPi QR and gesture pipeline")
    parser.add_argument("--source", choices=("robot", "webcam"), default="robot",
                        help="Camera source (default: robot stream).")
    parser.add_argument("--stream-url", default=DEFAULT_ROBOT_STREAM,
                        help="Robot camera HTTP stream URL.")
    parser.add_argument("--webcam-index", type=int, default=0,
                        help="Local webcam device index when --source webcam.")
    parser.add_argument("--qr-size", type=float, default=0.15, metavar="METERS",
                        help="Printed QR code width for fallback range estimation.")
    parser.add_argument("--camera-hfov", type=float, default=90.0, metavar="DEGREES",
                        help="Camera horizontal field of view for bearing/range estimates.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="MediaPipe Gesture Recognizer model path.")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Do not use LiDAR; estimate range from the QR code size.")
    parser.add_argument("--real-motion", action="store_true",
                        help="Enable wheel motion. Default is mock/safe mode.")
    parser.add_argument("--duration", type=float, default=0,
                        help="Seconds to run; 0 runs until Ctrl+C.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame gesture detections.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        camera = CameraFeed(args.source, args.stream_url, args.webcam_index)
    except RuntimeError as exc:
        print(f"[CAMERA] ERROR: {exc}", file=sys.stderr)
        return 1

    lidar_reader = None
    robot_path = os.path.join(REPO_ROOT, "robot")
    sys.path.insert(0, robot_path)
    if not args.no_lidar:
        try:
            from lidar_reader import LidarDistanceReader

            lidar_reader = LidarDistanceReader(window_deg=10.0, aggregation="min")
            lidar_reader.start()
            print("[LIDAR] LiDAR range enabled.")
        except Exception as exc:
            print(f"[LIDAR] Unavailable ({exc}); using QR-size range estimate.")
            lidar_reader = None

    motion = RobotMotionController(mock=not args.real_motion)
    if args.real_motion:
        print("[MOTION] REAL MOTION enabled. Keep clear space around the robot.")
        print("[MOTION] Starting in 5 seconds; Ctrl+C to abort.")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            camera.close()
            motion.close()
            if lidar_reader:
                lidar_reader.stop()
            return 130

    try:
        recognizer = GestureRecognizer(args.model)
    except RuntimeError as exc:
        print(f"[GESTURE] ERROR: {exc}", file=sys.stderr)
        camera.close()
        motion.close()
        if lidar_reader:
            lidar_reader.stop()
        return 1

    qr_tracker = QRTracker(
        camera_hfov_deg=args.camera_hfov,
        qr_size_m=args.qr_size,
        lidar_reader=lidar_reader,
    )
    processing_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision")
    cv2.namedWindow("MentorPi Camera Preview", cv2.WINDOW_NORMAL)
    print("[PREVIEW] OpenCV window started. Press q or Esc to quit.")
    print(f"[CAMERA] Source: {args.source}")
    print("[GESTURE] Recognizer reads the original camera frame; tracking does not gate gestures.")

    voter = GestureIntentVoter()
    active_intent = "NEUTRAL"
    last_gesture_time = 0.0
    last_dispatch_time = 0.0
    frame_number = 0
    started = time.monotonic()
    last_log = started
    try:
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            now = time.monotonic()
            ok, original_frame = camera.read()
            if not ok:
                motion.update_target(None, now)
                voter.update(now, None, None)
                if active_intent != "NEUTRAL" and now - last_gesture_time > RELEASE_S:
                    active_intent = "NEUTRAL"
                    motion.dispatch_intent("NEUTRAL")
                time.sleep(0.03)
                continue

            frame_number += 1
            # Both processors read the same unmodified frame in parallel.
            # QR tracking returns an untouched copy; only the gesture
            # recognizer adds overlays before the main thread displays it.
            qr_future = processing_pool.submit(qr_tracker.process_frame, original_frame)
            gesture_future = processing_pool.submit(recognizer.process_frame, original_frame)
            target, annotated_frame = qr_future.result()
            motion.update_target(target, now)
            label, score, hands = gesture_future.result()
            intent = voter.update(now, label, score)

            if args.debug and label is not None:
                print(f"[GESTURE] {label} ({score:.2f})")
            if intent is not None:
                last_gesture_time = now
                if intent != active_intent or now - last_dispatch_time >= REDISPATCH_S:
                    extra = "ACK" if intent == "CHECK_IN" else None
                    motion.dispatch_intent(intent, extra=extra)
                    active_intent = intent
                    last_dispatch_time = now
                    print(f"[INTENT] {intent} | QR target: "
                          f"{'visible' if target['visible'] else 'not visible'}")
            elif active_intent != "NEUTRAL" and now - last_gesture_time > RELEASE_S:
                active_intent = "NEUTRAL"
                motion.dispatch_intent("NEUTRAL")
                print("[INTENT] Gesture released -> NEUTRAL")

            recognizer.draw_hand_skeletons(annotated_frame, hands)
            cv2.imshow("MentorPi Camera Preview", annotated_frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("\n[RUN] Preview closed by user.")
                break

            if now - last_log >= 5.0:
                print(f"[STATUS] {frame_number} frames | gesture={label or 'none'} | "
                      f"QR={'visible' if target['visible'] else 'not visible'} | "
                      f"intent={active_intent}")
                last_log = now
    except KeyboardInterrupt:
        print("\n[RUN] Stopping.")
    finally:
        processing_pool.shutdown(wait=True)
        cv2.destroyAllWindows()
        camera.close()
        recognizer.close()
        motion.close()
        if lidar_reader is not None:
            lidar_reader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
