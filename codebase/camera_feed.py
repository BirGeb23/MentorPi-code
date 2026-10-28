"""Camera input and a small MJPEG preview server for the processed feed."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import cv2


# DEFAULT_ROBOT_STREAM = ("http://192.168.149.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image")
# ROBOT_IP = "192.168.149.1"
ROBOT_IP = "192.168.149.1"
DEFAULT_ROBOT_STREAM = (f"http://{ROBOT_IP}:8080/stream?topic=/ascamera/camera_publisher/rgb0/image")


class CameraFeed:
    """Read original frames from the MentorPi stream or a local webcam."""

    def __init__(self, source="robot", stream_url=DEFAULT_ROBOT_STREAM, webcam_index=0):
        self.source = source
        capture_target = stream_url if source == "robot" else int(webcam_index)
        self.capture = cv2.VideoCapture(capture_target)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open {source} camera: {capture_target}")
        if source == "webcam":
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        """Return (success, original_bgr_frame)."""
        return self.capture.read()

    def close(self):
        self.capture.release()


class LivePreviewServer:
    """Serve the latest annotated frame at /stream and a viewing page at /."""

    def __init__(self, host="0.0.0.0", port=8091):
        self._lock = threading.Lock()
        self._jpeg = None
        self._sequence = 0
        self._stopping = threading.Event()

        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = (
                        "<!doctype html><meta name='viewport' content='width=device-width'>"
                        "<title>MentorPi QR preview</title>"
                        "<style>body{margin:0;background:#111;color:#eee;font:16px sans-serif}"
                        "header{padding:12px 18px}img{display:block;width:100%;height:auto}</style>"
                        "<header>MentorPi live camera · QR overlay</header>"
                        "<img src='/stream' alt='Live processed camera feed'>"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                last_sequence = -1
                try:
                    while not preview._stopping.is_set():
                        with preview._lock:
                            jpeg = preview._jpeg
                            sequence = preview._sequence
                        if jpeg is None or sequence == last_sequence:
                            time.sleep(0.03)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                        last_sequence = sequence
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True, name="LivePreviewServer"
        )

    def start(self):
        self.thread.start()

    def publish(self, frame_bgr):
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return
        with self._lock:
            self._jpeg = encoded.tobytes()
            self._sequence += 1

    def close(self):
        self._stopping.set()
        self.server.shutdown()
        self.server.server_close()
