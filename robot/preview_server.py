"""
============================================================================
preview_server.py  —  Live video preview in a browser (MJPEG over HTTP)
============================================================================
The runners run headless on the Pi, so a cv2.imshow window has nowhere to
go. This serves the annotated frames instead; open the address it prints
in a browser on any computer on the robot's network:

    http://<pi-ip>:5001/          page with the live video
    http://<pi-ip>:5001/stream    the raw MJPEG stream

Usage:
    preview = PreviewServer(port=5001)
    preview.start()
    ...
    if preview.wants_frame():          # skip drawing when nobody needs it
        preview.update(annotated_bgr)
    ...
    preview.stop()
============================================================================
"""

import http.server
import socket
import socketserver
import threading
import time

import cv2

PREVIEW_PORT   = 5001
PREVIEW_FPS    = 10     # at most this many frames encoded per second
JPEG_QUALITY   = 70
VIEWER_WRITE_TIMEOUT_S = 5.0   # give up on a viewer that stops reading

_PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MentorPi preview</title>
<style>
  body { margin: 0; background: #111; color: #ccc; font: 14px sans-serif;
         display: flex; flex-direction: column; align-items: center; }
  img  { max-width: 100%; max-height: 92vh; margin-top: 8px; }
  p    { margin: 6px; }
</style></head>
<body><img src="/stream" alt="MentorPi camera">
<p>MentorPi preview. Reload if the video stops.</p></body></html>
"""


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class PreviewServer:
    def __init__(self, port=PREVIEW_PORT, fps=PREVIEW_FPS):
        self.port = port
        self.period = 1.0 / fps
        self._jpeg = None
        self._seq = 0
        self._cond = threading.Condition()
        self._last_t = 0.0
        self._last_sent_t = None   # last time a frame reached any viewer
        self._running = False
        self._httpd = None

    # ---- runner side ----
    def start(self):
        preview = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass   # keep the runner's terminal clean

            def do_GET(self):
                if self.path == "/stream":
                    preview._serve_stream(self)
                elif self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_PAGE)))
                    self.end_headers()
                    self.wfile.write(_PAGE)
                else:
                    self.send_error(404)

        self._httpd = _Server(("0.0.0.0", self.port), Handler)
        self._running = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True,
                         name="PreviewServer").start()
        print(f"[PREVIEW] Open http://{_lan_ip()}:{self.port}/ in a browser")

    def seconds_without_viewer(self):
        """Seconds since a frame last reached a viewer, or None if nobody
        has watched yet."""
        if self._last_sent_t is None:
            return None
        return time.time() - self._last_sent_t

    def wants_frame(self):
        """True when the next update() would be sent (rate limit)."""
        return time.time() - self._last_t >= self.period

    def update(self, frame_bgr):
        self._last_t = time.time()
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    # ---- browser side ----
    def _serve_stream(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type",
                            "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        # A viewer that vanished without closing (laptop off Wi-Fi) must not
        # hang this thread on a full socket
        handler.connection.settimeout(VIEWER_WRITE_TIMEOUT_S)
        seen = -1
        try:
            while self._running:
                with self._cond:
                    self._cond.wait_for(
                        lambda: (self._jpeg is not None and self._seq != seen)
                        or not self._running,
                        timeout=1.0)
                    if self._jpeg is None or self._seq == seen:
                        continue
                    jpeg, seen = self._jpeg, self._seq
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                    b"Content-Length: " + str(len(jpeg)).encode()
                                    + b"\r\n\r\n" + jpeg + b"\r\n")
                self._last_sent_t = time.time()
        except OSError:
            pass   # viewer closed the tab, crashed or went away


def _lan_ip():
    """Best guess at this machine's address on the network (no traffic sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 80))   # TEST-NET address, never reached
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()
