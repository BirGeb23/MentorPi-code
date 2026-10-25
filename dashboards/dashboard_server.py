#!/usr/bin/env python3
"""
============================================================================
dashboard_server.py  —  Live dashboard for the MentorPi robot
============================================================================
Serves a browser dashboard at http://<pi-ip>:5000/ that shows:
    - Live camera feed (proxied from web_video_server so the browser
      doesn't need direct access to port 8080)
    - Current intent, VA values, distance, bearing
    - Motion command being executed
    - Safety state
    - Cycle metadata

The dashboard is a static HTML page (dashboard.html) that polls this
server's /status endpoint every 500 ms.

Communication with the runner:
    - The runner (continuous_runner_http.py) writes state to
      /tmp/spot_dashboard_status.json after each cycle step
    - This server reads that file when the browser asks for /status
    - No shared memory, no sockets — just a file, so a dashboard crash
      never breaks the runner and vice versa

Usage:
    Terminal 1 (Pi):
        source ~/va_pipeline_venv/bin/activate
        cd ~/MentorPi-code/dashboards
        python3 dashboard_server.py

    Terminal 2 (Pi):
        source ~/va_pipeline_venv/bin/activate
        cd ~/MentorPi-code/dashboards
        python3 continuous_runner_http.py --real-motion

    Browser (laptop/phone on same network):
        http://<pi-ip>:5000/
============================================================================
"""

import http.server
import os
import socketserver
import sys
import urllib.parse
import urllib.request

# ---------- Configuration ----------
STATUS_FILE = "/tmp/spot_dashboard_status.json"
DASHBOARD_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "dashboard.html")
PORT = 5000

# The camera stream is served by web_video_server inside the MentorPi
# container. We proxy it through this server so the dashboard can just
# hit /camera and doesn't need to know the underlying URL.
CAMERA_UPSTREAM = "http://127.0.0.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image"


class DashboardHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        """Quiet the default access-log spam."""
        pass

    # ---------- GET routes ----------
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path in ("/", "/index.html", "/dashboard"):
            self._serve_html()
            return

        if path == "/status":
            self._serve_status()
            return

        if path == "/camera":
            self._proxy_camera()
            return

        # Unknown path
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not found")

    # ---------- Route handlers ----------
    def _serve_html(self):
        """Serve the dashboard HTML page."""
        try:
            with open(DASHBOARD_HTML, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"dashboard.html not found at "
                             f"{DASHBOARD_HTML}".encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_status(self):
        """Serve the current runner state as JSON."""
        try:
            with open(STATUS_FILE, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            data = b'{"error": "runner not started yet"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy_camera(self):
        """
        Stream the camera MJPEG from web_video_server through this server.
        This lets the browser hit our port 5000 for everything, without
        needing to know or reach port 8080 directly.
        """
        try:
            upstream = urllib.request.urlopen(CAMERA_UPSTREAM, timeout=5)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f"Camera unavailable: {e}".encode())
            return

        # Copy upstream headers (mostly the multipart Content-Type)
        self.send_response(200)
        for h in ("Content-Type", "Cache-Control", "Pragma"):
            v = upstream.headers.get(h)
            if v:
                self.send_header(h, v)
        self.end_headers()

        # Stream the MJPEG data chunk-by-chunk to the browser.
        try:
            while True:
                chunk = upstream.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browser disconnected — normal, don't spam errors
            pass
        except Exception as e:
            print(f"[DASHBOARD] Camera stream error: {e}", file=sys.stderr)


def main():
    print(f"[DASHBOARD] Starting server on http://0.0.0.0:{PORT}/")
    print(f"[DASHBOARD] Serving {DASHBOARD_HTML}")
    print(f"[DASHBOARD] Reading status from {STATUS_FILE}")
    print(f"[DASHBOARD] Proxying camera from {CAMERA_UPSTREAM}")
    print(f"[DASHBOARD] Open http://<pi-ip>:{PORT}/ in your browser")
    print(f"[DASHBOARD] Ctrl+C to stop.\n")

    # Threaded server so camera streaming doesn't block /status polling
    class ThreadedTCPServer(socketserver.ThreadingMixIn,
                            socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads      = True

    try:
        with ThreadedTCPServer(("", PORT), DashboardHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[DASHBOARD] Stopped by user.")


if __name__ == "__main__":
    main()
