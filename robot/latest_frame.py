"""
============================================================================
latest_frame.py  —  Camera reader that keeps only the newest frame
============================================================================
Wrap a cv2.VideoCapture whose reader may be slower than the camera:

    cap = LatestFrameReader(cv2.VideoCapture(url))
    ok, frame = cap.read()      # newest frame not returned yet
    age = time.time() - cap.frame_t
    cap.release()
============================================================================
"""

import threading
import time


class LatestFrameReader:
    """Reads the camera on its own thread and keeps only the newest frame.
    cap.read() on the HTTP stream returns frames in order, so whenever a loop
    is slower than the camera, frames queue up and everything runs on
    images from seconds ago. read() returns the newest frame not yet
    returned, waiting up to timeout s for one."""

    def __init__(self, cap):
        self.cap = cap
        self.frame_t = 0.0   # when the frame last returned by read() arrived
        self._frame = None
        self._t = 0.0
        self._seq = self._taken = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="CameraReader")
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            with self._cond:
                self._frame, self._t = frame, time.time()
                self._seq += 1
                self._cond.notify_all()
        # Only this thread touches cap: releasing it from another thread
        # while a read is in progress crashes OpenCV (bus error on macOS)
        self.cap.release()

    def read(self, timeout=1.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._taken,
                                       timeout=timeout):
                return False, None
            self._taken = self._seq
            self.frame_t = self._t
            return True, self._frame

    def release(self):
        """Stop reading. The reader thread closes the stream once its
        current read returns; a read stuck on a dead stream is left to the
        (daemon) thread rather than pulled out from under it."""
        self._running = False
        self._thread.join(timeout=2.0)
