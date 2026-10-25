#!/usr/bin/env python3
"""
============================================================================
test_rotation_direction.py  —  Verify the tracker→adapter rotation sign
============================================================================
This bypasses the emotion pipeline entirely so we can isolate ONE question:
    "When the tracker sees a person off-center, does the robot rotate
     TOWARD them or AWAY from them?"

For each of 3 cycles, we:
  1. Capture 4 seconds of camera, run the tracker to get bearing/distance
  2. FORCE the intent to CHECK_IN (so angular motion is guaranteed to fire
     if the bearing is outside the alignment tolerance)
  3. Ask the adapter to publish that motion for 2.5 seconds, then stop

No pipeline, no VA values, no chance of NEUTRAL clobbering the test.

Positions to test:
    Cycle 1: stand directly ahead (should not rotate — inside tolerance)
    Cycle 2: step ONE STEP TO YOUR LEFT and stay there
    Cycle 3: step ONE STEP TO YOUR RIGHT and stay there

Usage:
    source ~/va_pipeline_venv/bin/activate
    cd ~/MentorPi-code/tools
    python3 test_rotation_direction.py
============================================================================
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import cv2

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from mentorpi_adapter import MentorPiAdapter
from person_tracker  import PersonTracker
from lidar_reader    import LidarDistanceReader

STREAM_URL       = "http://127.0.0.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image"
CAPTURE_SECONDS  = 4.0
MOTION_HOLD_S    = 2.5


@dataclass
class ReactionAction:
    intent:    str
    pose_mode: str
    valence:   float = 0.0
    arousal:   float = 0.0


def capture_and_track(tracker):
    """
    Open the HTTP stream, run tracker on frames for CAPTURE_SECONDS,
    return the freshest detection (or None if no face seen).
    """
    print(f"[CAP] Capturing {CAPTURE_SECONDS}s and tracking...")
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("[CAP] Failed to open stream")
        return None

    latest = None
    frames_with_face = 0
    frame_idx = 0
    start = time.time()
    while time.time() - start < CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue
        frame_idx += 1
        # Run tracker on every 3rd frame
        if frame_idx % 3 == 0:
            det = tracker.process_frame(frame)
            if det['visible']:
                latest = det
                frames_with_face += 1

    cap.release()
    if latest:
        src = latest.get('distance_source', '?')
        print(f"[TRK] Face seen in {frames_with_face} sampled frames | "
              f"distance={latest['distance_m']:.2f}m ({src})  "
              f"bearing={latest['bearing_rad']:+.2f}rad "
              f"({latest['bearing_rad']*57.3:+.0f}°)")
    else:
        print("[TRK] NO face detected this cycle")
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Force CHECK_IN each cycle to isolate rotation direction test."
    )
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Stop after N cycles. 0 = infinite until Ctrl+C.")
    args = parser.parse_args()

    print("=" * 72)
    print("  ROTATION DIRECTION TEST  —  bypass pipeline, force CHECK_IN")
    print("  Robot in open space. Ctrl+C to abort.")
    print("=" * 72)
    print("  Move around freely between cycles — each capture is only 4s.")
    print("  Suggested positions:")
    print("    Straight ahead → robot should NOT rotate (within tolerance)")
    print("    Step LEFT      → robot should rotate to follow you left")
    print("    Step RIGHT     → robot should rotate to follow you right")
    if args.max_cycles > 0:
        print(f"  Will stop after {args.max_cycles} cycles.")
    else:
        print("  Running indefinitely — Ctrl+C to stop.")
    print("=" * 72)
    print("\n  Starting in 5 seconds — Ctrl+C to abort.")
    time.sleep(5)

    adapter = MentorPiAdapter(mock=False)
    adapter.state.person_visible  = False
    adapter.state.person_distance = 3.0
    adapter.state.person_bearing  = 0.0
    adapter.state.missing_windows = 0

    lidar_reader = LidarDistanceReader(window_deg=10.0, aggregation="min")
    try:
        lidar_reader.start()
        print("[LIDAR] Using LiDAR for distance.")
    except Exception as e:
        print(f"[LIDAR] LiDAR unavailable ({e}); using face-bbox fallback.")
        lidar_reader = None

    tracker = PersonTracker(
        camera_hfov_deg=90.0,
        distance_k=60.0,
        min_confidence=0.5,
        model_selection=1,
        lidar_reader=lidar_reader,
    )
    print("[TRK] Tracker ready.\n")

    cycle = 0
    try:
        while True:
            cycle += 1
            if args.max_cycles > 0 and cycle > args.max_cycles:
                print(f"\nReached max cycles ({args.max_cycles}). Stopping.")
                break
            print(f"----- Cycle {cycle} @ {time.strftime('%H:%M:%S')} -----")

            det = capture_and_track(tracker)
            if det:
                adapter.state.person_visible  = True
                adapter.state.person_distance = det['distance_m']
                adapter.state.person_bearing  = det['bearing_rad']
                adapter.state.missing_windows = 0

                # Force CHECK_IN — pushes distance target to 1.5m, angular=0.30
                # regardless of what the emotion pipeline would have said.
                # Also set target_d far so linear motion doesn't cut it off.
                ra = ReactionAction(intent="CHECK_IN", pose_mode="REACTION_POSE")
                print(f"[DISPATCH] Forcing CHECK_IN (bearing "
                      f"{det['bearing_rad']*57.3:+.0f}°) → adapter:")
                print("           ", end="")
                # Publish at 20Hz for MOTION_HOLD_S — beats the motor
                # controller's ~200ms watchdog and gives smooth motion.
                adapter.hold_for(ra, MOTION_HOLD_S)

                # Then stop
                stop_ra = ReactionAction(intent="NEUTRAL", pose_mode="REACTION_POSE")
                print("           [STOP] ", end="")
                adapter.hold_for(stop_ra, 0.5)
            else:
                print("[DISPATCH] No face — skipping motion this cycle.")

            print(f"----- Cycle {cycle} complete -----\n")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        adapter.shutdown()
        try:
            tracker.close()
        except Exception:
            pass
        if lidar_reader is not None:
            try:
                lidar_reader.stop()
            except Exception:
                pass
        print("Test complete.")


if __name__ == "__main__":
    main()
