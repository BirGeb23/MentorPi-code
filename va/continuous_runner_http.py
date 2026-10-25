#!/usr/bin/env python3
"""
============================================================================
continuous_runner_http.py  —  Continuous runner using HTTP camera capture
============================================================================
Same as continuous_runner.py, but uses web_video_server's HTTP stream for
camera capture instead of the docker exec + ROS 2 recorder. Much simpler
and more reliable.

Runs on the HOST Pi. Loops:
  1. Pull CYCLE_DURATION_S seconds of video from web_video_server via HTTP
  2. Run the pipeline (run_offline.py) on that video
  3. Read the resulting CSV
  4. Dispatch each window's ReactionAction to MentorPiAdapter (mock mode)
  5. Clean up and loop

Adapter is in MOCK mode — no wheel motion. Ctrl+C to stop.

Usage:
    source ~/va_pipeline_venv/bin/activate
    cd ~/MentorPi-code/va
    python3 continuous_runner_http.py            # mock (prints motion)
    python3 continuous_runner_http.py --real-motion
============================================================================
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import cv2

# sounddevice/scipy are optional -- video-only mode still works without them,
# same fallback pattern as mentorpi_live_capture.py.
try:
    import numpy as np
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    from scipy.signal import resample_poly
    HAS_AUDIO = True
except (ImportError, OSError):   # OSError: PortAudio library missing
    HAS_AUDIO = False

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from mentorpi_adapter import MentorPiAdapter
from person_tracker  import PersonTracker
from lidar_reader    import LidarDistanceReader

# ============================================================================
# Configuration
# ============================================================================
CYCLE_DURATION_S = 6            # seconds of camera to capture per cycle
# The team's VA pipeline (chaicenter-spot-basics/va-driven-behaviors): looked for
# next to this repo first, then at ~/chaicenter-spot-basics/va-driven-behaviors.
# Set MENTORPI_VA_DIR to use another location.
_REPO_VA_DIR = os.path.join(os.path.dirname(_MENTORPI_ROOT), "va-driven-behaviors")
PIPELINE_DIR = os.environ.get("MENTORPI_VA_DIR", _REPO_VA_DIR if os.path.isdir(_REPO_VA_DIR)
                              else os.path.expanduser("~/chaicenter-spot-basics/va-driven-behaviors"))
SAMPLES_DIR      = os.path.join(PIPELINE_DIR, "samples")
CSV_DIR          = os.path.join(PIPELINE_DIR, "evaluation/session_csvs")
# Python used to run the pipeline: the Pi's venv if present, else this one.
_PI_VENV = os.path.expanduser("~/va_pipeline_venv/bin/python")
VENV_PYTHON = _PI_VENV if os.path.exists(_PI_VENV) else sys.executable

# HTTP stream from web_video_server (running inside the MentorPi container)
CAMERA_TOPIC = "/ascamera/camera_publisher/rgb0/image"
STREAM_URL   = f"http://127.0.0.1:8080/stream?topic={CAMERA_TOPIC}"
FPS          = 15.0

# --- Audio: WonderEcho Pro USB mic ---
# Confirmed via `arecord -D hw:2,0 --dump-hw-params`: the device is FIXED at
# 48000 Hz / 2 channels / S16_LE -- it cannot record 16kHz mono directly.
# Capture at the native format and downmix + resample in software, same
# approach validated in mentorpi_live_capture.py. Device index 0 confirmed
# via `sd.query_devices()` on this Pi.
AUDIO_DEVICE_INDEX    = 0
NATIVE_AUDIO_RATE     = 48000
NATIVE_AUDIO_CHANNELS = 2
PIPELINE_AUDIO_RATE   = 16000

HOST_MP4 = os.path.join(SAMPLES_DIR, "live_capture.mp4")
HOST_WAV = os.path.join(SAMPLES_DIR, "live_capture.wav")
BASENAME = "live_capture"

# --- Live dashboard status file ---
# The runner writes its current state to this JSON file after each step,
# and the dashboard_server.py serves it to the browser at /status.
STATUS_FILE = "/tmp/spot_dashboard_status.json"

# Global snapshot of the runner's current state. Updated incrementally as
# each cycle progresses, then serialized to STATUS_FILE.
_status = {
    "cycle":     0,
    "phase":     "starting",     # starting | recording | pipeline | dispatch | idle
    "person":    {"visible": False, "distance_m": None, "bearing_rad": None,
                  "bearing_deg": None, "distance_source": None},
    "emotion":   {"valence": None, "arousal": None},
    "intent":    {"current": None, "history": []},
    "motion":    {"linear_mps": 0.0, "angular_radps": 0.0, "reason": None},
    "safety":    {"estop": "OK", "distance_floor": "OK", "battery": "OK",
                  "missing_windows": 0},
    "camera":    {"stream_url": "/camera"},   # dashboard proxies via server
    "runner":    {"mode": "MOCK"},
    # Rolling history of the last N completed pipeline windows.
    # Each entry is a snapshot of what the pipeline decided for that
    # window — VA values, intent, and the person state at that moment.
    # Newest entries appended, oldest dropped when we exceed HISTORY_MAX.
    "history":   [],
}

# How many recent cycles to keep in the dashboard history
HISTORY_MAX = 5


def write_status():
    """Serialize the current status snapshot to STATUS_FILE. Best-effort:
    swallows any errors so a dashboard problem never breaks the runner."""
    try:
        _status["timestamp"] = time.time()
        with open(STATUS_FILE, "w") as f:
            json.dump(_status, f)
    except Exception:
        pass


# ============================================================================
# Minimal ReactionAction
# ============================================================================
@dataclass
class ReactionAction:
    intent:    str
    pose_mode: str
    valence:   float = 0.0
    arousal:   float = 0.0


# ============================================================================
# Cycle steps
# ============================================================================
def record_from_http(tracker=None, adapter=None) -> "tuple[int, int] | None":
    """
    Capture CYCLE_DURATION_S seconds via web_video_server HTTP stream.

    If a `tracker` (PersonTracker) and `adapter` are supplied, we also run
    face detection on every FACE_DETECT_STRIDE-th frame and update
    adapter.state with the latest person visibility, bearing, and distance.

    Returns:
        (h, w) -- the camera frame height and width in pixels, read from
        the first frame of the stream -- if at least one frame was
        captured; None on failure. A tuple is always truthy and None is
        falsy, so `if not record_from_http(...)` still works as a
        success check.
    """
    print(f"[REC] Recording {CYCLE_DURATION_S}s from {STREAM_URL}")

    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print(f"[REC] Failed to open HTTP stream")
        return None

    ret, frame = cap.read()
    if not ret:
        print(f"[REC] Could not read first frame")
        cap.release()
        return None

    h, w = frame.shape[:2]
    print(f"[REC] Camera resolution: {w}x{h} (w x h)")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(HOST_MP4, fourcc, FPS, (w, h))
    writer.write(frame)
    count = 1

    # --- Start audio capture NOW -- at the same real-world moment the video
    # loop's own timer begins, not before cv2.VideoCapture() connects. HTTP
    # stream setup + waiting for the first frame takes a variable few
    # seconds over the network; starting audio before that left several
    # seconds of unmatched "dead" audio at the front of every clip. ---
    audio_chunks = []
    audio_stop = threading.Event()
    audio_thread_error = []
    t_audio = None

    if HAS_AUDIO:
        def _audio_callback(indata, frames, time_info, status):
            if not audio_stop.is_set():
                audio_chunks.append(indata.copy())

        def _audio_thread_fn():
            try:
                with sd.InputStream(
                    samplerate=NATIVE_AUDIO_RATE,
                    channels=NATIVE_AUDIO_CHANNELS,
                    dtype="float32",
                    device=AUDIO_DEVICE_INDEX,
                    callback=_audio_callback,
                ):
                    audio_stop.wait()
            except Exception as e:
                audio_thread_error.append(e)

        t_audio = threading.Thread(target=_audio_thread_fn, daemon=True)
        t_audio.start()
    else:
        print("[REC] sounddevice/scipy not installed -- recording video only.")

    # Person-tracking state across this cycle
    latest_detection = None
    frames_with_face = 0
    FACE_DETECT_STRIDE = 5    # process every 5th frame

    start = time.time()
    while time.time() - start < CYCLE_DURATION_S:
        ret, frame = cap.read()
        if ret:
            writer.write(frame)
            count += 1

            # Face detection every FACE_DETECT_STRIDE-th frame
            if tracker is not None and count % FACE_DETECT_STRIDE == 0:
                det = tracker.process_frame(frame)
                if det['visible']:
                    latest_detection = det
                    frames_with_face += 1
        else:
            time.sleep(0.02)

    writer.release()
    cap.release()

    # --- Stop audio and save it, downmixed + resampled to what the
    # pipeline expects (16kHz mono). Native capture is 48kHz stereo, an
    # exact 3:1 ratio, so resample_poly is the clean way to downsample. ---
    audio_stop.set()
    has_real_audio = False
    if HAS_AUDIO:
        t_audio.join(timeout=2.0)
        if audio_thread_error:
            print(f"[REC] Audio capture error: {audio_thread_error[0]}")
        elif audio_chunks:
            try:
                raw       = np.concatenate(audio_chunks, axis=0)  # (N, 2) float32
                mono      = raw.mean(axis=1)                       # downmix
                resampled = resample_poly(mono, up=1, down=3)       # 48000 -> 16000
                audio_int16 = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
                wavfile.write(HOST_WAV, PIPELINE_AUDIO_RATE, audio_int16)
                has_real_audio = True
                print(f"[REC] Audio saved: {len(audio_int16)/PIPELINE_AUDIO_RATE:.1f}s "
                      f"@ {PIPELINE_AUDIO_RATE}Hz")
            except Exception as e:
                print(f"[REC] Could not save audio: {e}")

    if not has_real_audio and os.path.exists(HOST_WAV):
        # Don't let a stale WAV from a previous cycle get silently reused.
        try:
            os.remove(HOST_WAV)
        except OSError:
            pass

    # Update adapter state with the freshest face detection
    if adapter is not None:
        if latest_detection is not None:
            adapter.state.person_visible   = True
            adapter.state.person_distance  = latest_detection['distance_m']
            adapter.state.person_bearing   = latest_detection['bearing_rad']
            adapter.state.missing_windows  = 0
            src = latest_detection.get('distance_source', '?')
            print(f"[TRK] Face seen in {frames_with_face} frames | "
                  f"distance={latest_detection['distance_m']:.2f}m ({src})  "
                  f"bearing={latest_detection['bearing_rad']:+.2f}rad "
                  f"({latest_detection['bearing_rad']*57.3:+.0f}°)")
            # Update dashboard status
            _status["person"] = {
                "visible":         True,
                "distance_m":      round(latest_detection['distance_m'], 2),
                "bearing_rad":     round(latest_detection['bearing_rad'], 3),
                "bearing_deg":     round(latest_detection['bearing_rad'] * 57.3, 1),
                "distance_source": src,
            }
        else:
            adapter.state.person_visible   = False
            adapter.state.missing_windows += 1
            print(f"[TRK] No face detected this cycle "
                  f"(missing_windows={adapter.state.missing_windows})")
            _status["person"] = {
                "visible":         False,
                "distance_m":      None,
                "bearing_rad":     None,
                "bearing_deg":     None,
                "distance_source": None,
            }
        _status["safety"]["missing_windows"] = adapter.state.missing_windows
        write_status()

    print(f"[REC] Captured {count} frames")
    return (h, w) if count > 0 else None


def run_pipeline() -> bool:
    """Run run_offline.py on the recorded video (+ real audio, if captured)."""
    print(f"[RUN] Running pipeline on {HOST_MP4} (~20-40 s)...")
    cmd = [
        VENV_PYTHON, "run_offline.py",
        f"samples/{BASENAME}.mp4",
        "--sparse", "--mock-robot",
    ]
    if os.path.exists(HOST_WAV):
        cmd += ["--audio", HOST_WAV]
    else:
        print("[RUN] No WAV this cycle -- pipeline will fall back to "
              "(silent) audio extracted from the video.")
    result = subprocess.run(cmd, cwd=PIPELINE_DIR,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[RUN] Pipeline failed:\n{result.stderr[-500:]}")
        return False
    return True


def read_pipeline_csv():
    csv_path = os.path.join(CSV_DIR, f"{BASENAME}_results.csv")
    if not os.path.exists(csv_path):
        print(f"[CSV] No CSV at {csv_path}")
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


# --- How long the robot actually moves after each dispatch, before we send
# an explicit stop. Prevents motion from persisting into the next cycle's
# camera recording (which would blur the video and drift the robot).
MOTION_HOLD_S = 2.5

# --- Personality gesture triggers ---
# ENGAGE's forward motion still runs on every ENGAGE dispatch, but the
# greet gesture is now a one-time "hello" per ENGAGE streak instead of
# firing every time (which looked repetitive):
#   - It waits for ENGAGE_GREET_THRESHOLD consecutive ENGAGE dispatches,
#     so a single noisy ENGAGE reading doesn't trigger it.
#   - It fires at most once per streak; the robot doesn't keep greeting
#     someone it's already engaged with. Leaving ENGAGE and coming back
#     starts a new streak.
#   - ENGAGE_GREET_COOLDOWN_S stops repeated greets when the intent
#     flickers ENGAGE -> other -> ENGAGE in quick succession.
# NEUTRAL and CAUTION fire their signals only after N consecutive cycles
# of the same intent, so the robot doesn't fidget every ~20 seconds when
# it's mostly holding still.
ENGAGE_GREET_THRESHOLD  = 2     # consecutive ENGAGE dispatches before greeting
ENGAGE_GREET_COOLDOWN_S = 30.0  # min seconds between greets
NEUTRAL_PULSE_THRESHOLD = 3   # cycles of consecutive NEUTRAL before pulsing
CAUTION_SCAN_THRESHOLD  = 2   # cycles of consecutive CAUTION before scanning

# Runtime tracker for consecutive-same-intent count. Updated each dispatch.
_intent_state = {
    "last_intent":       None,
    "consecutive_count": 0,
    "engage_greeted":    False,  # greeted during the current ENGAGE streak?
    "last_greet_time":   0.0,
}


def dispatch_to_adapter(rows, adapter):
    print(f"[ADPT] Dispatching {len(rows)} window(s):")
    for row in rows:
        ra = ReactionAction(
            intent    = row["effective_intent"],
            pose_mode = row["pose_mode"],
            valence   = float(row["valence"]),
            arousal   = float(row["arousal"]),
        )
        print(f"       window {row['window_index']}: "
              f"V={ra.valence:+.3f} A={ra.arousal:+.3f}")
        print(f"       ", end="")

        # Update dashboard status BEFORE motion so it shows the intent being executed
        _status["emotion"] = {
            "valence": round(ra.valence, 3),
            "arousal": round(ra.arousal, 3),
        }
        _status["intent"]["current"] = ra.intent
        # Keep a short history of recent intents
        history = _status["intent"]["history"]
        history.append(ra.intent)
        _status["intent"]["history"] = history[-5:]   # keep last 5

        # Look up the motion the adapter WOULD command (for dashboard display)
        # This mirrors what hold_for computes internally.
        if ra.pose_mode == "VA_POSE":
            preview_motion = adapter._va_to_motion(ra.valence, ra.arousal)
        else:
            preview_motion = adapter._motion_for_intent(ra.intent)
        from safety import apply_all
        safe_preview = apply_all(preview_motion, adapter.state)
        _status["motion"] = {
            "linear_mps":     round(safe_preview.linear, 3),
            "angular_radps":  round(safe_preview.angular, 3),
            "reason":         safe_preview.reason,
        }
        _status["phase"] = "dispatch"

        # Append this cycle's decision to the rolling history so it can
        # be shown in the dashboard's "Recent Cycles" panel. Newest first.
        history_entry = {
            "cycle":       _status["cycle"],
            "timestamp":   time.time(),
            "valence":     round(ra.valence, 3),
            "arousal":     round(ra.arousal, 3),
            "intent":      ra.intent,
            "distance_m":  _status["person"].get("distance_m"),
            "bearing_deg": _status["person"].get("bearing_deg"),
            "linear_mps":  round(safe_preview.linear, 3),
            "angular_radps": round(safe_preview.angular, 3),
        }
        _status["history"].insert(0, history_entry)   # newest at index 0
        _status["history"] = _status["history"][:HISTORY_MAX]   # cap at N

        write_status()

        # Publish the motion continuously at 20Hz for MOTION_HOLD_S seconds.
        # The MentorPi motor watchdog stops the wheels if we don't republish
        # every ~200ms, so a single publish only produces a brief twitch —
        # hold_for gives smooth continuous motion for the full window.
        adapter.hold_for(ra, MOTION_HOLD_S)

        # ---------- Personality gestures ----------
        # Track how many cycles in a row we've seen this same intent.
        if ra.intent == _intent_state["last_intent"]:
            _intent_state["consecutive_count"] += 1
        else:
            _intent_state["consecutive_count"] = 1
            _intent_state["last_intent"]       = ra.intent
            _intent_state["engage_greeted"]    = False  # new streak

        # ENGAGE: greet once per streak, after ENGAGE_GREET_THRESHOLD
        # consecutive dispatches, respecting the cooldown. The forward
        # motion above (hold_for) still runs every time regardless.
        if ra.intent == "ENGAGE":
            now = time.time()
            if (_intent_state["consecutive_count"] >= ENGAGE_GREET_THRESHOLD
                    and not _intent_state["engage_greeted"]
                    and now - _intent_state["last_greet_time"]
                        >= ENGAGE_GREET_COOLDOWN_S):
                print(f"       [after {_intent_state['consecutive_count']} "
                      f"ENGAGE cycles] ", end="")
                adapter.greet()
                _intent_state["engage_greeted"]  = True
                _intent_state["last_greet_time"] = now

        # NEUTRAL: pulse after N consecutive NEUTRAL cycles as a
        # subtle "still alive" signal. Then reset the counter so the next
        # pulse waits another N cycles.
        elif (ra.intent == "NEUTRAL"
              and _intent_state["consecutive_count"] >= NEUTRAL_PULSE_THRESHOLD):
            print(f"       [after {_intent_state['consecutive_count']} "
                  f"NEUTRAL cycles] ", end="")
            adapter.neutral_pulse()
            _intent_state["consecutive_count"] = 0

        # CAUTION: scan after N consecutive CAUTION cycles as a
        # "still watching" signal. Then reset counter.
        elif (ra.intent == "CAUTION"
              and _intent_state["consecutive_count"] >= CAUTION_SCAN_THRESHOLD):
            print(f"       [after {_intent_state['consecutive_count']} "
                  f"CAUTION cycles] ", end="")
            adapter.caution_scan()
            _intent_state["consecutive_count"] = 0

        # Then explicitly STOP the wheels — send stop at rate briefly to
        # make sure the watchdog picks it up.
        stop_ra = ReactionAction(intent="NEUTRAL", pose_mode="REACTION_POSE")
        print(f"       [STOP after {MOTION_HOLD_S}s] ", end="")
        adapter.hold_for(stop_ra, 0.5)

        # After stop, reset motion display
        _status["motion"] = {
            "linear_mps":    0.0,
            "angular_radps": 0.0,
            "reason":        "stopped between cycles",
        }
        _status["phase"] = "idle"
        write_status()


def cleanup():
    if os.path.exists(HOST_MP4):
        os.remove(HOST_MP4)
    if os.path.exists(HOST_WAV):
        os.remove(HOST_WAV)


# ============================================================================
# Main loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Continuous live camera + pipeline + adapter runner."
    )
    parser.add_argument("--real-motion", action="store_true",
                        help="Enable real wheel motion. Default is MOCK (safe).")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="Stop after N cycles. 0 = infinite until Ctrl+C.")
    args = parser.parse_args()

    is_mock = not args.real_motion

    print("=" * 72)
    print("  CONTINUOUS RUNNER (HTTP)  —  live MentorPi camera → pipeline → adapter")
    if is_mock:
        print("  Mode: MOCK (no motion). Ctrl+C to stop.")
    else:
        print("  Mode: *** REAL MOTION *** — robot wheels will actually turn!")
        print("  Have hand on power switch. Ctrl+C to stop.")
        print("  Robot must be in open space.")
    if args.max_cycles > 0:
        print(f"  Will stop after {args.max_cycles} cycles.")
    print("=" * 72)

    if not is_mock:
        # Extra safety pause when using real motion
        print("\n  Starting real-motion cycles in 5 seconds — Ctrl+C to abort.")
        time.sleep(5)

    adapter = MentorPiAdapter(mock=is_mock)
    # Initial state — will be updated in real time by the person tracker
    # once the first frame with a face is captured.
    adapter.state.person_visible  = False
    adapter.state.person_distance = 3.0    # placeholder until a face is seen
    adapter.state.person_bearing  = 0.0
    adapter.state.missing_windows = 0

    # LiDAR distance reader — provides accurate (±3 cm) distance readings
    # from the MentorPi's ORADAR MS200. Fused with the face bearing to give
    # the tracker a real physical distance instead of the noisy face-height
    # estimate. Falls back to face-based if the LiDAR is unavailable.
    #
    # aggregation="min": the LiDAR sees the person's LEGS (mounted at torso
    # height). At any given bearing, some beams hit the leg (~1.5m) and some
    # slip through leg-gap to the wall behind (~3m). MIN picks the leg
    # distance, which is the person's real distance. MEDIAN would incorrectly
    # get pulled toward the wall.
    lidar_reader = LidarDistanceReader(window_deg=10.0, aggregation="min")
    try:
        lidar_reader.start()
        print("[LIDAR] Using LiDAR for distance (accurate).")
    except Exception as e:
        print(f"[LIDAR] Could NOT start LiDAR reader: {e}")
        print("[LIDAR] Falling back to face-based distance estimation.")
        lidar_reader = None

    # MediaPipe face-based person tracker.
    # Using full-range model (model_selection=1) because the MentorPi
    # camera has a wide FOV, so a person standing at typical interaction
    # distance appears with a small face (~30-60px). The short-range
    # model only handles closer faces.
    print(f"[TRK] Initializing MediaPipe face detector (full-range)...")
    tracker = PersonTracker(
        camera_hfov_deg=90.0,        # Angstrong camera has a wide FOV
        distance_k=60.0,             # face-bbox fallback only (used if LiDAR None)
        min_confidence=0.5,
        model_selection=1,           # 1 = full-range (up to ~5m)
        lidar_reader=lidar_reader,   # fuse LiDAR distance with face bearing
    )
    print(f"[TRK] Tracker ready.")

    # Initial status write
    _status["runner"]["mode"] = "MOCK" if is_mock else "REAL_MOTION"
    write_status()

    cycle = 0
    try:
        while True:
            cycle += 1
            if args.max_cycles > 0 and cycle > args.max_cycles:
                print(f"\nReached max cycles ({args.max_cycles}). Stopping.")
                break
            cycle_start = time.time()
            print(f"\n----- Cycle {cycle} @ {time.strftime('%H:%M:%S')} -----")

            _status["cycle"] = cycle
            _status["phase"] = "recording"
            write_status()

            frame_size = record_from_http(tracker=tracker, adapter=adapter)
            if not frame_size:
                print(f"[CYCLE {cycle}] Recording failed, skipping.")
                time.sleep(2)
                continue
            frame_h, frame_w = frame_size

            _status["phase"] = "pipeline"
            write_status()

            if not run_pipeline():
                print(f"[CYCLE {cycle}] Pipeline failed, skipping.")
                cleanup()
                time.sleep(2)
                continue

            rows = read_pipeline_csv()
            if rows:
                dispatch_to_adapter(rows, adapter)
            else:
                print(f"[CYCLE {cycle}] No pipeline results.")

            cleanup()
            elapsed = time.time() - cycle_start
            print(f"----- Cycle {cycle} complete in {elapsed:.1f}s -----")

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
        print("Runner shut down.")


if __name__ == "__main__":
    main()
