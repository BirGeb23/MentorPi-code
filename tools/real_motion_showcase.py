#!/usr/bin/env python3
"""
============================================================================
real_motion_showcase.py  —  Test all 5 intents with real motion
============================================================================
Bypasses the pipeline. Sends each intent directly to the adapter in real
mode so you can verify every motion type works. Same as mentorpi_demo.py's
Phase 1 showcase, but with mock=False so wheels actually turn.

Each intent runs for MOTION_DURATION_S seconds, then a full stop is sent,
then a pause before the next intent. Total: about 40 seconds.

SAFETY:
  - Robot must be in open space (~3m clear in front and behind)
  - Hand on power switch
  - Ctrl+C aborts and sends stop
  - Speeds are the adapter's designed values (0.10-0.15 m/s max)

Usage:
    python3 real_motion_showcase.py
============================================================================
"""

import os
import sys
import time
from dataclasses import dataclass

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from mentorpi_adapter import MentorPiAdapter

# ---- How long to hold each intent before moving to the next ----
MOTION_DURATION_S = 4.0    # each intent gets this many seconds of motion
PAUSE_S           = 2.0    # pause with stop commands between intents


@dataclass
class ReactionAction:
    intent:    str
    pose_mode: str = "REACTION_POSE"
    valence:   float = 0.0
    arousal:   float = 0.0


def hold_intent(adapter, intent: str, duration_s: float, hz: float = 5.0):
    """Send the intent repeatedly for the duration so the motor keeps moving."""
    ra = ReactionAction(intent=intent)
    period = 1.0 / hz
    end = time.time() + duration_s
    while time.time() < end:
        adapter.apply_reaction(ra)
        time.sleep(period)


def stop_hold(adapter, duration_s: float, hz: float = 10.0):
    """Send NEUTRAL (stop) repeatedly to guarantee the robot stops."""
    ra = ReactionAction(intent="NEUTRAL")
    period = 1.0 / hz
    end = time.time() + duration_s
    while time.time() < end:
        adapter.apply_reaction(ra)
        time.sleep(period)


def phase(adapter, title: str, intent: str, person_dist: float, hold_s: float):
    """One phase: set person distance, hold intent, then stop and pause."""
    print(f"\n" + "=" * 66)
    print(f"  {title}")
    print(f"  intent={intent}   person_dist={person_dist}m   hold={hold_s}s")
    print("=" * 66)

    adapter.state.person_distance = person_dist

    print(f"  Starting motion in 2 seconds — Ctrl+C aborts")
    time.sleep(2)

    print(f"  Sending {intent} for {hold_s}s...")
    hold_intent(adapter, intent, hold_s)

    print(f"  Sending NEUTRAL stop for {PAUSE_S}s...")
    stop_hold(adapter, PAUSE_S)


def main():
    print("\n" + "#" * 66)
    print("#" + " " * 64 + "#")
    print("#       REAL MOTION SHOWCASE — all 5 intents with wheels     #")
    print("#" + " " * 64 + "#")
    print("#  Robot MUST be in open space. Hand on power switch.        #")
    print("#" + " " * 64 + "#")
    print("#" * 66)

    print("\nStarting in 5 seconds — Ctrl+C to abort.")
    time.sleep(5)

    adapter = MentorPiAdapter(mock=False)
    adapter.state.person_visible  = True
    adapter.state.person_bearing  = 0.0

    try:
        # Person is far — CHECK_IN and ENGAGE should approach
        phase(adapter, "1. NEUTRAL — should NOT move",
              "NEUTRAL",  person_dist=2.0, hold_s=MOTION_DURATION_S)

        phase(adapter, "2. CHECK_IN — should approach slowly toward 1.5m",
              "CHECK_IN", person_dist=3.0, hold_s=MOTION_DURATION_S)

        phase(adapter, "3. ENGAGE — should approach at moderate speed toward 1.0m",
              "ENGAGE",   person_dist=3.0, hold_s=MOTION_DURATION_S)

        phase(adapter, "4. CAUTION — should hold still (regardless of distance)",
              "CAUTION",  person_dist=3.0, hold_s=MOTION_DURATION_S)

        phase(adapter, "5. DE_ESCALATE — should back away toward 2.0m",
              "DE_ESCALATE", person_dist=1.0, hold_s=MOTION_DURATION_S)

        print(f"\n" + "=" * 66)
        print(f"  Showcase complete. Sending final stop...")
        stop_hold(adapter, 1.0)
        print("=" * 66)

    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPTED] Sending emergency stop...")
        stop_hold(adapter, 2.0)

    finally:
        adapter.shutdown()
        print("\nDone.")


if __name__ == "__main__":
    main()
