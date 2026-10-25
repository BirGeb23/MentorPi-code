#!/usr/bin/env python3
"""
============================================================================
offline_full_loop.py  —  Full pipeline+adapter loop on a saved video
============================================================================
Runs the entire emotion-recognition pipeline on a video file and dispatches
each window's ReactionAction to the MentorPiAdapter in mock mode. Same
architecture as continuous_runner.py, but uses a saved video as input
instead of the live camera. Safe to run anywhere — no motion.

What this proves:
  - Pipeline output cleanly feeds into MentorPiAdapter without manual steps
  - The full loop is automatable in one command
  - When the live camera comes back, this pattern extends to live operation
    (only the input source changes)

Usage:
    python3 offline_full_loop.py <video_path> [--pacing SECONDS]

Example:
    python3 offline_full_loop.py samples/mentorpi_capture_run2.mp4
    python3 offline_full_loop.py samples/angry_man.mp4 --pacing 1.0

The --pacing flag controls how long to wait between dispatching each
window's ReactionAction (default 2.0 s — a compromise between simulating
real-time behavior and keeping the demo watchable).
============================================================================
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass

# Make sure we can import MentorPiAdapter from the same folder
# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from mentorpi_adapter import MentorPiAdapter

# ============================================================================
# Configuration (adjust these paths if your setup is different)
# ============================================================================
# On the Pi, the pipeline lives at ~/chaicenter-spot-basics/va-driven-behaviors
# On your laptop, adjust to wherever you cloned it.
# The team's VA pipeline (chaicenter-spot-basics/va-driven-behaviors): looked for
# next to this repo first, then at ~/chaicenter-spot-basics/va-driven-behaviors.
# Set MENTORPI_VA_DIR to use another location.
_REPO_VA_DIR = os.path.join(os.path.dirname(_MENTORPI_ROOT), "va-driven-behaviors")
PIPELINE_DIR = os.environ.get("MENTORPI_VA_DIR", _REPO_VA_DIR if os.path.isdir(_REPO_VA_DIR)
                              else os.path.expanduser("~/chaicenter-spot-basics/va-driven-behaviors"))
SAMPLES_DIR  = os.path.join(PIPELINE_DIR, "samples")
CSV_DIR      = os.path.join(PIPELINE_DIR, "evaluation/session_csvs")
# Python used to run the pipeline: the Pi's venv if present, else this one.
_PI_VENV = os.path.expanduser("~/va_pipeline_venv/bin/python")
VENV_PYTHON = _PI_VENV if os.path.exists(_PI_VENV) else sys.executable


# ============================================================================
# Minimal ReactionAction — matches what the pipeline emits
# ============================================================================
@dataclass
class ReactionAction:
    intent:    str
    pose_mode: str
    valence:   float = 0.0
    arousal:   float = 0.0


# ============================================================================
# Helper: pretty section separator
# ============================================================================
def section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ============================================================================
# Step 1: Run the pipeline on the video via subprocess
# ============================================================================
def run_pipeline_on(video_path: str) -> bool:
    """
    Invoke run_offline.py on the given video. Uses sparse mode so it's
    fast enough for interactive demos. Returns True on success.
    """
    # The pipeline expects the video to be inside samples/ relative to its dir
    video_name = os.path.basename(video_path)
    if not os.path.exists(os.path.join(SAMPLES_DIR, video_name)):
        # Copy the video into the samples folder so the pipeline can find it
        print(f"[SETUP] Copying {video_path} to pipeline samples folder...")
        import shutil
        shutil.copy(video_path, os.path.join(SAMPLES_DIR, video_name))

    print(f"[PIPELINE] Running run_offline.py on {video_name}...")
    print(f"[PIPELINE] This may take 20-45 seconds (model loads on first run).")

    cmd = [
        VENV_PYTHON, "run_offline.py",
        f"samples/{video_name}",
        "--sparse", "--mock-robot",
        # Lower the confidence gate so we can see the pipeline's actual
        # intent proposals come through the adapter (instead of everything
        # being demoted to NEUTRAL). Set to 0.5 or higher for production.
        "--min-confidence", "0.1",
    ]
    result = subprocess.run(
        cmd, cwd=PIPELINE_DIR,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[PIPELINE] Failed with exit code {result.returncode}:")
        print(result.stderr[-500:])
        return False

    print(f"[PIPELINE] Complete.")
    return True


# ============================================================================
# Step 2: Read the CSV the pipeline produced
# ============================================================================
def read_pipeline_csv(video_path: str) -> list:
    """
    Locate and read the CSV that the pipeline produced for this video.
    The pipeline writes it to evaluation/session_csvs/{basename}_results.csv.
    """
    basename = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(CSV_DIR, f"{basename}_results.csv")

    if not os.path.exists(csv_path):
        print(f"[CSV] Expected CSV not found: {csv_path}")
        return []

    print(f"[CSV] Reading {csv_path}")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f"[CSV] Loaded {len(rows)} window(s) of pipeline output.")
    return rows


# ============================================================================
# Step 3: Dispatch each row's ReactionAction to the adapter
# ============================================================================
def dispatch_to_adapter(rows: list, adapter: MentorPiAdapter, pacing_s: float):
    """
    For each pipeline window, build a ReactionAction and call the adapter's
    apply_reaction() method. Wait pacing_s seconds between dispatches.
    """
    section(f"Dispatching {len(rows)} window(s) to MentorPiAdapter")

    for i, row in enumerate(rows):
        window_idx = row["window_index"]
        t_start    = row["t_start"]
        label      = row["state_label"]
        conf       = float(row["state_confidence"])

        # Build the ReactionAction from the CSV row
        ra = ReactionAction(
            intent    = row["effective_intent"],
            pose_mode = row["pose_mode"],
            valence   = float(row["valence"]),
            arousal   = float(row["arousal"]),
        )

        print(f"\n[LOOP] Window {window_idx}  (t={t_start}s)")
        print(f"       Pipeline: label={label}  conf={conf:.3f}  "
              f"V={ra.valence:+.3f}  A={ra.arousal:+.3f}")
        print(f"       Adapter :", end=" ")
        adapter.apply_reaction(ra)

        # Simulate real-time pacing (except after the last window)
        if i < len(rows) - 1:
            time.sleep(pacing_s)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline+adapter loop on a saved video."
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--pacing", type=float, default=2.0,
                        help="Seconds to wait between window dispatches "
                             "(default 2.0, use 0 for no delay)")
    args = parser.parse_args()

    section(f"Offline Full-Loop Runner")
    print(f"  Video : {args.video}")
    print(f"  Pacing: {args.pacing}s between window dispatches")
    print(f"  Mode  : MOCK (no motion)")

    # ---- Step 1: Run the pipeline ----
    if not os.path.exists(args.video):
        print(f"\n[ERROR] Video not found: {args.video}")
        sys.exit(1)

    if not run_pipeline_on(args.video):
        print(f"\n[ERROR] Pipeline failed. Cannot continue.")
        sys.exit(1)

    # ---- Step 2: Read the CSV ----
    rows = read_pipeline_csv(args.video)
    if not rows:
        print(f"\n[ERROR] No pipeline output to dispatch.")
        sys.exit(1)

    # ---- Step 3: Build the adapter and dispatch ----
    section("Building MentorPiAdapter (mock mode)")
    adapter = MentorPiAdapter(mock=True)
    adapter.state.person_visible  = True
    adapter.state.person_distance = 1.5
    adapter.state.person_bearing  = 0.0
    print("  Person state: visible=True, distance=1.5m, bearing=0.0 rad")

    dispatch_to_adapter(rows, adapter, args.pacing)

    # ---- Cleanup ----
    section("Shutdown")
    adapter.shutdown()

    print("\n" + "=" * 72)
    print("  Loop complete.")
    print("  Every [MENTORPI-MOCK] line above is what the robot WOULD do")
    print("  in response to the pipeline's analysis of that video window.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
