"""
============================================================================
mentorpi_streaming_session.py — MentorPi variant of StreamingSession
============================================================================
Subclasses online.streaming_session.StreamingSession (the codebase's real
ring-buffer + pipelined online architecture) to work with the MentorPi's
actual hardware instead of a laptop's local webcam + default microphone.

Everything about the parent class is reused UNCHANGED except three things:

  1. Camera source: StreamingSession opens `cv2.VideoCapture(camera_index)`
     with a local integer device index. The MentorPi's Angstrong depth
     camera has no such device -- it's only reachable through the HTTP
     MJPEG stream served by web_video_server. This subclass overrides
     _open_camera() to point at that URL instead, and reads one frame up
     front to establish real resolution (the HTTP stream doesn't reliably
     honor CAP_PROP_FRAME_WIDTH/HEIGHT the way a real camera device does --
     same finding as mentorpi_live_capture.py).

  2. Microphone: StreamingSession's audio thread requests 16kHz mono
     directly from the system's default input device. The WonderEcho Pro
     is fixed at 48kHz/2-channel hardware (confirmed via
     `arecord -D hw:2,0 --dump-hw-params`) and cannot honor that request.
     This subclass captures at the native format from an explicit device
     index instead, and downmixes + resamples to 16kHz mono when each
     snapshot is saved to disk -- the same approach already validated in
     mentorpi_live_capture.py and continuous_runner_http.py.

  3. Person/LiDAR tracking: added directly into the continuous capture
     loop (Loop A), the same way continuous_runner_http.py already does
     it -- every FACE_DETECT_STRIDE-th frame gets passed to PersonTracker,
     and the latest detection updates adapter.state so the safety layer
     and CHECK_IN/ENGAGE distance logic keep working the same way they do
     in the cycle-based runner.

The ring buffer itself, the pipelined worker-pool analysis, stability
gating (ReactionHistory), dispatch, and all printing are inherited
directly from StreamingSession -- none of that needed to change.
============================================================================
"""

import os
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from online.streaming_session import StreamingSession, RingBuffer, HAS_SOUNDDEVICE

try:
    import sounddevice as sd
    import scipy.io.wavfile as wavfile
    from scipy.signal import resample_poly
except (ImportError, OSError):   # OSError: PortAudio library missing
    pass  # HAS_SOUNDDEVICE (imported above) already reflects availability


DEFAULT_STREAM_URL = (
    "http://192.168.149.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image"
)

# Confirmed via `arecord -D hw:2,0 --dump-hw-params`: fixed, not a range.
NATIVE_AUDIO_RATE     = 48000
NATIVE_AUDIO_CHANNELS = 2
PIPELINE_AUDIO_RATE   = 16000

FACE_DETECT_STRIDE = 5   # run face detection on every 5th captured frame


class MentorPiStreamingSession(StreamingSession):
    """
    StreamingSession, pointed at the MentorPi's HTTP camera stream and
    WonderEcho Pro mic, with person/LiDAR tracking folded into the
    continuous capture loop.
    """

    def __init__(
        self,
        *args,
        stream_url: str = DEFAULT_STREAM_URL,
        audio_device_index: int = 0,
        tracker=None,
        lidar_reader=None,
        adapter=None,
        **kwargs,
    ):
        """
        Additional args beyond StreamingSession's own:
            stream_url:          HTTP MJPEG URL served by web_video_server.
            audio_device_index:  sounddevice index for the WonderEcho Pro
                                  (found via sd.query_devices() -- index 0
                                  on the Pi this was built against).
            tracker:              Optional PersonTracker -- if supplied,
                                  runs face detection during capture and
                                  updates adapter.state with the result.
            lidar_reader:         Optional LidarDistanceReader, passed
                                  through to the tracker for fused distance.
            adapter:               Optional MentorPiAdapter -- state gets
                                  updated with person visibility/distance/
                                  bearing each time a face is (or isn't)
                                  detected, same as continuous_runner_http.py.
        """
        self.stream_url         = stream_url
        self.audio_device_index = audio_device_index
        self.tracker             = tracker
        self.lidar_reader        = lidar_reader
        self.adapter              = adapter

        super().__init__(*args, **kwargs)

        # The parent's __init__ sized the ring buffer's audio budget around
        # `sample_rate` (16000, what the pipeline expects). We're actually
        # going to push NATIVE 48kHz audio into it, so rebuild the buffer
        # sized correctly for that -- otherwise the sample-count-based
        # trimming logic would drop audio roughly 3x too aggressively.
        max_frames         = int(self.fps * self.window_dur * 1.5)
        max_audio_samples  = int(NATIVE_AUDIO_RATE * self.window_dur * 1.5)
        self._buffer = RingBuffer(
            max_frames=max_frames,
            max_audio_samples=max_audio_samples,
            sample_rate=NATIVE_AUDIO_RATE,
        )

    # ── Camera: HTTP stream instead of a local device index ───────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open MJPEG stream at {self.stream_url}. "
                "Is web_video_server running in the MentorPi container?"
            )
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(
                f"Opened {self.stream_url} but got no frame. "
                "Check the camera topic is publishing."
            )
        h, w = frame.shape[:2]
        self.resolution = (w, h)
        print(f"Stream opened: {w}x{h}  (target {self.fps} fps)")

        # This first frame would otherwise be lost -- _capture_loop only
        # reads frames itself, it doesn't know about this one. Push it in
        # directly, and run tracking on it too if a tracker was supplied.
        self._buffer.push_frame(frame)
        if self.tracker is not None:
            self._update_person_state(frame)

        return cap

    # ── Loop A: capture + person tracking (overrides parent's version to
    #    fold tracking into the same loop, instead of a separate step) ────────

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        count = 0
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("  [capture] Frame read failed -- stream may have dropped.")
                time.sleep(0.02)
                continue
            if (frame.shape[1], frame.shape[0]) != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            self._buffer.push_frame(frame)

            count += 1
            if self.tracker is not None and count % FACE_DETECT_STRIDE == 0:
                self._update_person_state(frame)

    def _update_person_state(self, frame: np.ndarray) -> None:
        """Run face detection on one frame and update adapter.state, same
        fields/semantics continuous_runner_http.py already uses."""
        if self.adapter is None:
            return
        det = self.tracker.process_frame(frame)
        if det["visible"]:
            self.adapter.state.person_visible  = True
            self.adapter.state.person_distance = det["distance_m"]
            self.adapter.state.person_bearing  = det["bearing_rad"]
            self.adapter.state.missing_windows = 0
        else:
            self.adapter.state.person_visible   = False
            self.adapter.state.missing_windows += 1

    # ── Audio: WonderEcho Pro native format instead of 16kHz mono default ─────

    def _audio_loop(self) -> None:
        def _callback(indata, frames, time_info, status):
            self._buffer.push_audio(indata)

        try:
            with sd.InputStream(
                samplerate=NATIVE_AUDIO_RATE,
                channels=NATIVE_AUDIO_CHANNELS,
                dtype="float32",
                device=self.audio_device_index,
                callback=_callback,
            ):
                self._stop_event.wait()
        except Exception as e:
            print(f"  [audio] Stream error: {e}")

    # ── Snapshot saving: downmix + resample native audio before writing ───────

    def _save_snapshot(
        self,
        frames: List[np.ndarray],
        audio: Optional[np.ndarray],
        window_id: int,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        if not frames:
            return None, None, False

        video_path = os.path.join(self.temp_dir, f"mentorpi_stream_{window_id}.avi")
        audio_path = os.path.join(self.temp_dir, f"mentorpi_stream_{window_id}.wav")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.resolution)
        for frame in frames:
            writer.write(frame)
        writer.release()

        has_audio = False
        if audio is not None and len(audio) > 0 and HAS_SOUNDDEVICE:
            try:
                # `audio` here is native 48kHz stereo, shape (N, 2) -- downmix
                # to mono, then resample the exact 3:1 ratio down to 16kHz.
                mono        = audio.mean(axis=1)
                resampled   = resample_poly(mono, up=1, down=3)
                audio_int16 = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
                wavfile.write(audio_path, PIPELINE_AUDIO_RATE, audio_int16)
                has_audio = True
            except Exception as e:
                print(f"  [snapshot] Could not save audio: {e}")

        return video_path, audio_path if has_audio else video_path, has_audio
