#!/usr/bin/env python3
"""
============================================================================
run_online_mentorpi.py — Real online pipeline, wired for the MentorPi
============================================================================
This is the MentorPi-compatible counterpart to run_online.py: it uses the
same genuine ring-buffer + pipelined StreamingSession architecture (camera
never pauses, analysis runs in a worker thread in parallel with the next
capture window) instead of continuous_runner_http.py's simpler loop, which
restarts a fresh subprocess -- and loses all session history -- every cycle.

MUST be run on the Pi itself (not bridged from a laptop): true continuous
capture needs to be co-located with the camera/mic hardware.

Placement: runs straight from the repo. It imports the robot modules from
robot/ and the team's pipeline (online, pipeline, test_emotions,
models/, config/) from the team's va-driven-behaviors/: next to this repo,
or ~/chaicenter-spot-basics/va-driven-behaviors. Set MENTORPI_VA_DIR to
point somewhere else if needed.

Usage (mock mode -- safe, no wheel motion, same default as
continuous_runner_http.py):
    source ~/va_pipeline_venv/bin/activate
    cd ~/MentorPi-code/va
    python3 run_online_mentorpi.py --duration 30 --window 3

    # Verbose per-window detail
    python3 run_online_mentorpi.py --duration 30 --debug

    # Real wheel motion (only once mock-mode output looks right!)
    python3 run_online_mentorpi.py --duration 30 --real-motion
============================================================================
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# ── Paths: robot/ for the robot modules, va-driven-behaviors/ for
#    the team's pipeline (pipeline, online, test_emotions). ─────────────────
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
# The team's VA pipeline (chaicenter-spot-basics/va-driven-behaviors): looked for
# next to this repo first, then at ~/chaicenter-spot-basics/va-driven-behaviors.
# Set MENTORPI_VA_DIR to use another location.
_REPO_VA_DIR = os.path.join(os.path.dirname(_MENTORPI_ROOT), "va-driven-behaviors")
PIPELINE_DIR = os.environ.get("MENTORPI_VA_DIR", _REPO_VA_DIR if os.path.isdir(_REPO_VA_DIR)
                              else os.path.expanduser("~/chaicenter-spot-basics/va-driven-behaviors"))
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # mentorpi_streaming_session
try:
    from mentorpi_adapter import MentorPiAdapter
    from person_tracker    import PersonTracker
    from lidar_reader       import LidarDistanceReader
except ImportError as e:
    print(f"ERROR: Could not import MentorPi robot modules from robot/: {e}")
    print("  Make sure mentorpi_adapter.py / person_tracker.py / lidar_reader.py "
          "are present there.")
    sys.exit(1)

# ── Pipeline imports (same as run_online.py) ──────────────────────────────────
from pipeline import load_calibration
from online.window_analyzer import WindowAnalyzer

from mentorpi_streaming_session import MentorPiStreamingSession, DEFAULT_STREAM_URL

try:
    from test_emotions import load_model
except ImportError as e:
    print(f"ERROR: Could not import test_emotions: {e}")
    print("  Place test_emotions.py in the project root directory.")
    sys.exit(1)


@dataclass
class ReactionAction:
    intent:    str
    pose_mode: str
    valence:   float = 0.0
    arousal:   float = 0.0


# How long the robot actually holds each motion command before an explicit
# stop -- same constant/reasoning as continuous_runner_http.py.
MOTION_HOLD_S = 2.5


def find_model(args_model, root: Path) -> Path:
    if args_model:
        p = Path(args_model)
        if p.exists():
            return p
    for name in ("jointcam_finetuned_v4.pt", "jointcam_model.pt"):
        p = root / "models" / name
        if p.exists():
            return p
    print(
        "ERROR: No model file found.\n"
        f"  Place jointcam_finetuned_v4.pt in:  {root / 'models'}\n"
        "  Or specify with:  --model path/to/model.pt"
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real online (ring-buffer + pipelined) VA session, wired for the MentorPi."
    )
    p.add_argument("--duration", type=float, default=30.0, metavar="S",
                    help="Total session duration in seconds (default: 30)")
    p.add_argument("--window", type=float, default=3.0, metavar="S",
                    help="Analysis window size in seconds (default: 3, matches offline default)")
    p.add_argument("--stream-url", default=DEFAULT_STREAM_URL, metavar="URL",
                    help="HTTP MJPEG URL served by web_video_server")
    p.add_argument("--fps", type=float, default=15.0, metavar="F",
                    help="Target camera capture rate (default: 15)")
    p.add_argument("--audio-device", type=int, default=0, metavar="N",
                    help="sounddevice index for the WonderEcho Pro (default: 0)")
    p.add_argument("--no-audio", action="store_true",
                    help="Disable microphone capture (video-only mode)")
    p.add_argument("--min-confidence", type=float, default=0.55, metavar="C",
                    help="Minimum state_confidence to accept a new intent (default: 0.55)")
    p.add_argument("--cooldown", type=float, default=0.5, metavar="S",
                    help="Min seconds between behavior changes (default: 0.5)")
    p.add_argument("--sparse", action="store_true",
                    help="Sparse sampling mode: 2 frames per window (CPU-efficient)")
    p.add_argument("--model", default=None, metavar="PATH",
                    help="Path to model .pt file (default: models/jointcam_finetuned_v4.pt)")
    p.add_argument("--debug", action="store_true",
                    help="Verbose per-window detail")
    p.add_argument("--real-motion", action="store_true",
                    help="Enable real wheel motion. Default is MOCK (safe, no motion).")
    p.add_argument("--max-workers", type=int, default=2, metavar="N",
                    help="Concurrent analysis worker threads (default: 2). "
                         "Diagnostic: try 1 to test whether two simultaneous "
                         "JointCAM analyses contend with each other for CPU.")
    p.add_argument("--no-tracking", action="store_true",
                    help="Disable MediaPipe person/LiDAR tracking in the capture loop "
                         "(diagnostic: isolates whether tracking CPU load is slowing "
                         "down JointCAM analysis). Safety person-presence logic will "
                         "not update while this is set.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(PIPELINE_DIR)   # models/ and config/ live in va-driven-behaviors/

    model_path = find_model(args.model, root)
    calib_path = root / "config" / "calibration.json"
    calibration = load_calibration(str(calib_path)) if calib_path.exists() else None
    if calibration:
        print(f"Calibration loaded: {calib_path.name}")
    else:
        print("Warning: calibration.json not found -- using fallback thresholds.")

    print(f"Loading model: {model_path.name} ...", end=" ", flush=True)
    model = load_model(str(model_path), "cpu")
    print("done.")

    analyzer = WindowAnalyzer(model=model, device="cpu", calibration=calibration, debug=args.debug)

    is_mock = not args.real_motion
    print()
    print("=" * 62)
    print("  MENTORPI ONLINE SESSION  (ring buffer + pipelined)")
    if is_mock:
        print("  Mode: MOCK (no motion). Ctrl+C to stop.")
    else:
        print("  Mode: *** REAL MOTION *** -- robot wheels will actually turn!")
        print("  Have hand on power switch. Robot must be in open space.")
    print("=" * 62)

    if not is_mock:
        print("\n  Starting real-motion session in 5 seconds -- Ctrl+C to abort.")
        time.sleep(5)

    adapter = MentorPiAdapter(mock=is_mock)
    adapter.state.person_visible  = False
    adapter.state.person_distance = 3.0
    adapter.state.person_bearing  = 0.0
    adapter.state.missing_windows = 0

    if args.no_tracking:
        print("[TRACKING] Disabled via --no-tracking -- capture loop will do "
              "nothing but grab frames (diagnostic mode). Person-presence "
              "safety logic will NOT update.")
        lidar_reader = None
        tracker = None
    else:
        lidar_reader = LidarDistanceReader(window_deg=10.0, aggregation="min")
        try:
            lidar_reader.start()
            print("[LIDAR] Using LiDAR for distance.")
        except Exception as e:
            print(f"[LIDAR] Could not start LiDAR reader: {e}")
            print("[LIDAR] Falling back to face-based distance estimation.")
            lidar_reader = None

        tracker = PersonTracker(
            camera_hfov_deg=90.0,
            distance_k=60.0,
            min_confidence=0.5,
            model_selection=1,
            lidar_reader=lidar_reader,
        )

    # Motion dispatch runs on its own dedicated thread, never on the analysis
    # worker thread -- otherwise the ~3s spent inside hold_for() (blocking
    # sleep/republish) adds directly to the analysis worker's occupancy per
    # window, on top of the VA analysis itself, causing severe backlog under
    # a single-worker (or even two-worker) pool. See MOTION_HOLD_S usage
    # below. "Interrupt & replace": a fresher decision preempts an
    # in-progress hold immediately rather than waiting for it to finish.
    _motion_lock       = threading.Lock()
    _current_stop_event = [None]  # boxed so the closure below can rebind it

    def _run_motion(ra, stop_event):
        adapter.hold_for(ra, MOTION_HOLD_S, stop_event=stop_event)
        if stop_event.is_set():
            return  # preempted -- let the newer motion own the robot
        stop_ra = ReactionAction(intent="NEUTRAL", pose_mode="REACTION_POSE")
        adapter.hold_for(stop_ra, 0.5, stop_event=stop_event)

    def on_behavior_update(intent, ra, analysis):
        """Called by the worker thread the instant a stability-gated
        decision is reached -- mirrors continuous_runner_http.py's dispatch,
        just triggered by the streaming session instead of a per-cycle
        subprocess loop. Hands motion off to a dedicated thread immediately
        so the analysis worker is free to pick up the next window right
        away, instead of blocking through the whole hold duration."""
        if ra is None:
            return
        ra.intent = intent  # respect the stability gate, not the raw model intent

        with _motion_lock:
            prev_stop_event = _current_stop_event[0]
            if prev_stop_event is not None:
                prev_stop_event.set()  # preempt whatever hold is in progress
            new_stop_event = threading.Event()
            _current_stop_event[0] = new_stop_event

        threading.Thread(
            target=_run_motion, args=(ra, new_stop_event),
            daemon=True, name="MotionDispatch",
        ).start()

    num_samples = 2 if args.sparse else None

    try:
        # max_workers needs the updated online/streaming_session.py (team-file
        # fix); older versions don't accept it, so only pass it if supported.
        import inspect
        from online.streaming_session import StreamingSession
        extra_kwargs = {}
        if "max_workers" in inspect.signature(StreamingSession.__init__).parameters:
            extra_kwargs["max_workers"] = args.max_workers
        elif args.max_workers != 2:
            print("Note: this version of online/streaming_session.py has no "
                  "max_workers option; using its default of 2 workers.")

        session = MentorPiStreamingSession(
            window_analyzer=analyzer,
            stream_url=args.stream_url,
            fps=args.fps,
            resolution=(320, 240),
            session_duration_s=args.duration,
            window_duration_s=args.window,
            capture_audio=not args.no_audio,
            audio_device_index=args.audio_device,
            min_confidence=args.min_confidence,
            cooldown_s=args.cooldown,
            debug=args.debug,
            num_samples=num_samples,
            on_behavior_update=on_behavior_update,
            tracker=tracker,
            lidar_reader=lidar_reader,
            adapter=adapter,
            **extra_kwargs,
        )
        session.run()

    except KeyboardInterrupt:
        print("\n\n  Session interrupted (Ctrl+C).")
    except Exception as e:
        print(f"\n  FATAL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        adapter.shutdown()
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass
        if lidar_reader is not None:
            try:
                lidar_reader.stop()
            except Exception:
                pass

    print("  Done.")


if __name__ == "__main__":
    main()
