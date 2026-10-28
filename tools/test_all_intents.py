#!/usr/bin/env python3
"""
============================================================================
test_all_intents.py  —  Force each of the 5 intents, one per cycle
============================================================================
Runs one cycle per intent so you can visually verify what each behavior
looks like on the real robot:

    Cycle 1: NEUTRAL      → should do nothing (baseline)
    Cycle 2: CHECK_IN     → approach + face person, stop at 1.5 m
    Cycle 3: ENGAGE       → approach faster to closer distance (1.0 m)
    Cycle 4: DE_ESCALATE  → back straight away to 2.0 m (no rotation)
    Cycle 5: CAUTION      → freeze (same motion as NEUTRAL but different intent)

Between each cycle it captures 4 s of camera, runs the tracker to update
distance and bearing, then dispatches the intent with the 2.5 s continuous
20 Hz publish and a stop. Bypasses the emotion pipeline so we can isolate
each intent's motion behavior.

SUGGESTED STARTING POSITION:
    Stand about 2 m directly in front of the robot.

    The robot will approach on cycles 2 and 3, back away on cycle 4, then
    freeze on cycle 5. Move with it if needed to keep your face visible.
    Have room BEHIND the robot for cycle 4's back-away.

Usage:
    source ~/va_pipeline_venv/bin/activate
    cd ~/MentorPi-code/tools
    python3 test_all_intents.py

Optional: --hold N   set the motion hold duration per cycle (default 2.5s).
                     Use --hold 4 to see longer motion per intent.
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

STREAM_URL      = "http://192.168.149.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image"
CAPTURE_SECONDS = 4.0


# Sequence of intents to test, with human-readable explanations
INTENT_SEQUENCE = [
    ("NEUTRAL",
     "Should do NOTHING. Robot holds still (baseline)."),
    ("CHECK_IN",
     "Should approach slowly (0.08 m/s) until 1.5 m away and turn to face you."),
    ("ENGAGE",
     "Should approach FASTER (0.15 m/s) to a CLOSER distance (1.0 m), then "
     "do a forward-back GREET (bounces toward you twice for personality)."),
    ("DE_ESCALATE",
     "Should back STRAIGHT away (0.10 m/s reverse, no rotation) until 2.0 m."),
    ("CAUTION",
     "Should FREEZE (same motion as NEUTRAL, but the intent means 'be watchful')."),
]


@dataclass
class ReactionAction:
    intent:    str
    pose_mode: str
    valence:   float = 0.0
    arousal:   float = 0.0


def capture_and_track(tracker):
    """Capture CAPTURE_SECONDS of frames, return the freshest face detection."""
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
        description="Force each of the 5 intents, one per cycle, on the real robot."
    )
    parser.add_argument("--hold", type=float, default=2.5,
                        help="How long each intent's motion runs, in seconds "
                             "(default 2.5). Try --hold 4 for longer motion.")
    args = parser.parse_args()

    print("=" * 72)
    print("  ALL-INTENTS TEST  —  bypass pipeline, force each intent")
    print("  Robot in open space with room behind it (for DE_ESCALATE).")
    print("  Have hand on power switch. Ctrl+C to abort.")
    print("=" * 72)
    print(f"  {len(INTENT_SEQUENCE)} cycles, {args.hold}s of motion each.")
    print(f"  Recommended start position: ~2 m directly in front of robot.")
    print("=" * 72)
    print("\n  Starting in 5 seconds — Ctrl+C to abort.")
    time.sleep(5)

    adapter = MentorPiAdapter(mock=False)
    adapter.state.person_visible  = False
    adapter.state.person_distance = 2.0
    adapter.state.person_bearing  = 0.0
    adapter.state.missing_windows = 0

    # LiDAR reader for accurate distance (±3 cm). Falls back to face-bbox
    # estimation inside PersonTracker if start() fails.
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

    try:
        for cycle_num, (intent, explanation) in enumerate(INTENT_SEQUENCE, start=1):
            print("=" * 72)
            print(f"  CYCLE {cycle_num}/{len(INTENT_SEQUENCE)}: forcing {intent}")
            print(f"  Expected: {explanation}")
            print("=" * 72)
            print(f"[Cycle @ {time.strftime('%H:%M:%S')}]")

            det = capture_and_track(tracker)
            if det:
                adapter.state.person_visible  = True
                adapter.state.person_distance = det['distance_m']
                adapter.state.person_bearing  = det['bearing_rad']
                adapter.state.missing_windows = 0
            else:
                # Even without a detection, still dispatch — this lets
                # NEUTRAL/CAUTION show correct behavior (both stay still).
                adapter.state.person_visible  = False
                adapter.state.missing_windows += 1

            ra = ReactionAction(intent=intent, pose_mode="REACTION_POSE")
            print(f"[DISPATCH] Forcing {intent} → adapter:")
            print("           ", end="")
            adapter.hold_for(ra, args.hold)

            # ENGAGE personality: forward-back GREET after approaching.
            if intent == "ENGAGE":
                print("           ", end="")
                adapter.greet()

            # Explicit stop between intents so the robot doesn't drift into
            # the next intent's capture window.
            stop_ra = ReactionAction(intent="NEUTRAL", pose_mode="REACTION_POSE")
            print("           [STOP] ", end="")
            adapter.hold_for(stop_ra, 0.5)

            # Brief pause between intents so it's easy to see what happened
            print(f"----- {intent} cycle complete -----\n")
            time.sleep(1.5)

        print("=" * 72)
        print("  All 5 intents tested. Watch the summary above.")
        print("=" * 72)

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
