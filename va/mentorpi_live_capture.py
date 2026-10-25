#!/usr/bin/env python3
"""
============================================================================
mentorpi_live_capture.py — Synchronized video + audio capture for MentorPi
============================================================================
Adapts online/live_capture.py's LiveCapture design to the MentorPi's
specific hardware setup:

    - Video comes from the ROS 2 camera topic, re-served as an HTTP MJPEG
      stream by web_video_server (not a local webcam device index).
    - Audio comes from the WonderEcho Pro USB voice box, which is a fixed-
      format USB audio codec: 48000 Hz, 2 channels, S16_LE only (confirmed
      via `arecord -D hw:2,0 --dump-hw-params` -- CHANNELS and RATE are
      fixed values, not ranges). It cannot record 16kHz or mono directly --
      requesting that from the raw device silently records at 48kHz anyway
      and mislabels the WAV header. So we capture at the native format and
      downmix/downsample in software afterward, the same way the offline
      pipeline's extract_audio_features() already resamples via
      librosa.load(sr=16000, mono=True).

Both streams start within the same Python call (the audio background
thread is launched, then the video loop begins immediately after), so
they're aligned to a common wall-clock start with only sub-millisecond
thread-launch skew -- the same synchronization strategy LiveCapture uses
for laptop webcam+mic sessions.

Usage:
    capture = MentorPiLiveCapture(audio_device_index=0)
    capture.open()
    result = capture.capture_window(duration_s=3.0, window_id=0)
    # result['video_path'] and result['audio_path'] ready for the pipeline
    capture.close()

    # Or as a context manager:
    with MentorPiLiveCapture() as capture:
        result = capture.capture_window(3.0, 0)

Requires: opencv-python, sounddevice, scipy (pip install sounddevice scipy)
============================================================================
"""

import os
import tempfile
import threading
import time
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

try:
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    from scipy.signal import resample_poly
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):   # OSError: PortAudio library missing
    HAS_SOUNDDEVICE = False


# Same URL used by mentorpi_camera_recorder.py / record_demo_video.py --
# web_video_server re-serving the Angstrong RGB topic as HTTP MJPEG.
DEFAULT_STREAM_URL = (
    "http://127.0.0.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image"
)

# Confirmed via `arecord -D hw:2,0 --dump-hw-params`: the WonderEcho Pro's
# USB codec is FIXED at these values -- not a negotiable range.
NATIVE_AUDIO_RATE = 48000
NATIVE_AUDIO_CHANNELS = 2
PIPELINE_AUDIO_RATE = 16000  # what extract_audio_features() expects


class MentorPiLiveCapture:
    """
    Captures a fixed-duration window of MentorPi camera + WonderEcho Pro
    mic to temp files, synchronized to a common start.

    Mirrors online/live_capture.py's LiveCapture design so the returned
    dict drops into the same pipeline entry points unchanged.
    """

    def __init__(
        self,
        stream_url: str = DEFAULT_STREAM_URL,
        fps: float = 15.0,
        audio_device_index: Optional[int] = 0,
        capture_audio: bool = True,
        temp_dir: Optional[str] = None,
    ):
        """
        Args:
            stream_url:         HTTP MJPEG URL served by web_video_server.
            fps:                Target video frame rate for the output file.
                                 The Angstrong camera publishes at ~15fps.
            audio_device_index: sounddevice index for the WonderEcho Pro,
                                 found via `sd.query_devices()` (index 0 on
                                 the Pi you tested). Passed explicitly so
                                 capture never silently falls back to the
                                 Pi's "default"/"pulse" virtual device.
            capture_audio:      Whether to attempt mic capture. Falls back
                                 to video-only if sounddevice/scipy is missing.
            temp_dir:           Where to write temp files (defaults to the
                                 system temp dir).
        """
        self.stream_url = stream_url
        self.fps = fps
        self.audio_device_index = audio_device_index
        self.capture_audio = capture_audio and HAS_SOUNDDEVICE
        self.temp_dir = temp_dir or tempfile.gettempdir()

        self.cap: Optional[cv2.VideoCapture] = None
        self.resolution: Tuple[int, int] = (0, 0)

        if capture_audio and not HAS_SOUNDDEVICE:
            print(
                "Warning: sounddevice/scipy not installed -- running video-only.\n"
                "         Install with: pip install sounddevice scipy"
            )

    # -------------------------------------------------------------------------
    def open(self) -> None:
        """Open the HTTP MJPEG stream. Must be called before capture_window()."""
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open MJPEG stream at {self.stream_url}. "
                "Is web_video_server running in the MentorPi container?"
            )
        # Read one frame to establish actual resolution -- the HTTP stream
        # doesn't reliably honor CAP_PROP_FRAME_WIDTH/HEIGHT the way a real
        # camera device does.
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(
                f"Opened {self.stream_url} but got no frame. "
                "Check the camera topic is publishing."
            )
        h, w = frame.shape[:2]
        self.resolution = (w, h)
        print(f"Stream opened: {w}x{h} (target {self.fps} fps)")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "MentorPiLiveCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -------------------------------------------------------------------------
    def capture_window(self, duration_s: float = 3.0, window_id: int = 0) -> Dict:
        """
        Capture duration_s seconds of video + audio to temp files, started
        together. Blocks until duration_s has elapsed.

        Returns dict with the same keys as LiveCapture.capture_window():
        video_path, audio_path, n_frames, actual_duration_s, fps_achieved,
        has_real_audio.
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Stream not open. Call open() or use context manager.")

        video_path = os.path.join(self.temp_dir, f"mentorpi_window_{window_id}.avi")
        audio_path = os.path.join(self.temp_dir, f"mentorpi_window_{window_id}.wav")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.resolution)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {video_path}.")

        # --- Audio capture in background thread, at the device's NATIVE
        # format (48kHz stereo) -- downsampling/downmixing happens after,
        # not during, capture. ---
        audio_chunks: list = []
        audio_stop = threading.Event()
        audio_thread_error: list = []

        if self.capture_audio:
            def _audio_callback(indata, frames, time_info, status):
                if not audio_stop.is_set():
                    audio_chunks.append(indata.copy())

            def _audio_thread():
                try:
                    with sd.InputStream(
                        samplerate=NATIVE_AUDIO_RATE,
                        channels=NATIVE_AUDIO_CHANNELS,
                        dtype="float32",
                        device=self.audio_device_index,
                        callback=_audio_callback,
                    ):
                        audio_stop.wait()  # Blocks until we signal stop
                except Exception as e:
                    audio_thread_error.append(e)

            t_audio = threading.Thread(target=_audio_thread, daemon=True)
            t_audio.start()

        # --- Video capture loop (main thread) ---
        frame_count = 0
        t_start = time.time()
        print(f"  [capture] Recording {duration_s:.1f}s from MentorPi stream...", flush=True)

        while True:
            elapsed = time.time() - t_start
            if elapsed >= duration_s:
                break
            ret, frame = self.cap.read()
            if not ret:
                print("  Warning: Frame read failed -- stream may have dropped.")
                time.sleep(0.02)
                continue
            if (frame.shape[1], frame.shape[0]) != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            writer.write(frame)
            frame_count += 1

        actual_duration = time.time() - t_start
        fps_achieved = frame_count / actual_duration if actual_duration > 0 else 0.0

        # --- Stop audio ---
        audio_stop.set()
        if self.capture_audio:
            t_audio.join(timeout=2.0)
            if audio_thread_error:
                print(f"  Warning: Audio thread error: {audio_thread_error[0]}")

        writer.release()
        print(
            f"  [capture] Done: {frame_count} frames, {actual_duration:.1f}s, "
            f"{fps_achieved:.1f} fps achieved",
            flush=True,
        )

        # --- Downmix stereo->mono, resample 48000->16000, then save ---
        has_real_audio = False
        if audio_chunks and not audio_thread_error:
            try:
                raw = np.concatenate(audio_chunks, axis=0)  # (N, 2) float32
                mono = raw.mean(axis=1)  # downmix -> (N,)
                # 48000 -> 16000 is an exact 3:1 ratio -- resample_poly is the
                # clean way to do that (polyphase, no arbitrary-ratio artifacts).
                resampled = resample_poly(mono, up=1, down=3)
                audio_int16 = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
                wavfile.write(audio_path, PIPELINE_AUDIO_RATE, audio_int16)
                has_real_audio = True
                print(
                    f"  [capture] Audio saved: {len(audio_int16)/PIPELINE_AUDIO_RATE:.1f}s "
                    f"@ {PIPELINE_AUDIO_RATE} Hz (downmixed + resampled from "
                    f"{NATIVE_AUDIO_RATE}Hz/{NATIVE_AUDIO_CHANNELS}ch)",
                    flush=True,
                )
            except Exception as e:
                print(f"  Warning: Could not save audio: {e}")
                audio_path = video_path  # Fallback: let pipeline extract from video

        if not has_real_audio:
            audio_path = video_path  # Pipeline will extract (silent) audio from the .avi

        return {
            "video_path": video_path,
            "audio_path": audio_path,
            "n_frames": frame_count,
            "actual_duration_s": actual_duration,
            "fps_achieved": fps_achieved,
            "has_real_audio": has_real_audio,
        }

    @staticmethod
    def cleanup_files(capture_result: Dict) -> None:
        """Delete temp files produced by capture_window()."""
        for key in ("video_path", "audio_path"):
            path = capture_result.get(key)
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == "__main__":
    # Capture a fixed-duration window and report what we got.
    # Run this ON THE MENTORPI, in whichever venv has cv2/sounddevice/scipy
    # installed, with web_video_server already running (it is by default
    # as part of the normal MentorPi bringup).
    #
    # Usage:
    #   python3 mentorpi_live_capture.py                  # 3s smoke test (default)
    #   python3 mentorpi_live_capture.py --duration 20     # longer clip, multi-window
    #   python3 mentorpi_live_capture.py --duration 20 --window-id 1
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture a synchronized video+audio window from the MentorPi."
    )
    parser.add_argument(
        "--duration", type=float, default=3.0, metavar="S",
        help="Seconds to record (default: 3.0). Use a longer value (e.g. 20-30) "
             "to get multiple analysis windows out of run_offline.py instead of just one.",
    )
    parser.add_argument(
        "--window-id", type=int, default=0, metavar="N",
        help="Suffix for the output filenames, e.g. mentorpi_window_<N>.avi/.wav "
             "(default: 0). Bump this if you don't want to overwrite a previous capture.",
    )
    args = parser.parse_args()

    with MentorPiLiveCapture(audio_device_index=0) as capture:
        result = capture.capture_window(duration_s=args.duration, window_id=args.window_id)
        print("\nResult:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print(f"\nFiles written to:\n  {result['video_path']}\n  {result['audio_path']}")
        print(
            "\nCopy them off the Pi (scp) and check: audio sounds correct/normal-speed, "
            "and the video plays without corruption or dropped frames."
        )
