#!/usr/bin/env python3
"""
============================================================================
test_new_motions.py  —  Try the NEUTRAL pulse and CAUTION scan gestures
============================================================================
Runs the three "personality" motions in isolation so you can see how each
one looks on the actual robot before integrating them into the runner.

Motions:
    - ENGAGE GREET    → forward-back gesture for positive interaction
    - NEUTRAL PULSE   → subtle "still alive" back-forward motion
    - CAUTION SCAN    → back-turn-turn-return watchful motion

Usage:
    source ~/va_pipeline_venv/bin/activate
    cd ~/MentorPi-code/tools

    # See all three back-to-back (default) — good for comparison
    python3 test_new_motions.py

    # Just one at a time
    python3 test_new_motions.py --motion neutral_pulse
    python3 test_new_motions.py --motion caution_scan
    python3 test_new_motions.py --motion engage_greet

    # Adjust pause between motions
    python3 test_new_motions.py --pause 5

    # Repeat each motion N times so it's easier to watch
    python3 test_new_motions.py --repeat 3

SAFETY: Robot in open space (~2 m clear ahead AND behind — CAUTION scan
backs up). Hand on power switch. Ctrl+C to abort.
============================================================================
"""

import argparse
import os
import sys
import time

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from mentorpi_adapter import MentorPiAdapter


MOTIONS = ["engage_greet", "neutral_pulse", "caution_scan"]

MOTION_DESCRIPTIONS = {
    "engage_greet": (
        "ENGAGE GREET — quick forward-back bounces ×2, ~1.6s. "
        "Positive interaction gesture."
    ),
    "neutral_pulse": (
        "NEUTRAL PULSE — gentle back-then-forward, ~1.0s. "
        "Subtle 'still here, no action needed' signal."
    ),
    "caution_scan": (
        "CAUTION SCAN — back+turn right, hold, back+turn left, hold, "
        "small forward return, ~3.0s. 'Let me back up and check things out.'"
    ),
}


def run_motion(adapter, name):
    print()
    print("=" * 72)
    print(f"  {MOTION_DESCRIPTIONS[name]}")
    print("=" * 72)
    if name == "engage_greet":
        adapter.greet()
    elif name == "neutral_pulse":
        adapter.neutral_pulse()
    elif name == "caution_scan":
        adapter.caution_scan()


def main():
    parser = argparse.ArgumentParser(
        description="Test proposed NEUTRAL/CAUTION motions in isolation."
    )
    parser.add_argument("--motion", choices=MOTIONS + ["all"], default="all",
                        help="Which motion to run. Default: all three in sequence.")
    parser.add_argument("--pause", type=float, default=3.0,
                        help="Seconds to pause between motions. Default 3.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="How many times to repeat each motion. Default 1.")
    args = parser.parse_args()

    to_run = MOTIONS if args.motion == "all" else [args.motion]

    print("=" * 72)
    print("  NEW MOTIONS TEST — visual check on the real robot")
    print("=" * 72)
    print(f"  Motions:    {', '.join(to_run)}")
    print(f"  Repeat:     {args.repeat}x each")
    print(f"  Pause:      {args.pause}s between motions")
    print("=" * 72)
    print("  Robot in open space (~2 m clear front and back).")
    print("  CAUTION SCAN backs up ~12 cm — make sure there's room behind.")
    print("  Hand on power switch. Ctrl+C to abort.")
    print()
    print("  Starting in 5 seconds...")
    time.sleep(5)

    adapter = MentorPiAdapter(mock=False)

    try:
        for name in to_run:
            for i in range(args.repeat):
                if args.repeat > 1:
                    print(f"\n  [repeat {i+1}/{args.repeat}]")
                run_motion(adapter, name)
                if args.pause > 0:
                    print(f"\n  ...pausing {args.pause}s...")
                    time.sleep(args.pause)
        print("\n" + "=" * 72)
        print("  Test complete.")
        print("=" * 72)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        adapter.shutdown()


if __name__ == "__main__":
    main()
