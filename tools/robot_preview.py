#!/usr/bin/env python3
"""
============================================================================
robot_preview.py  —  Run the gesture runner on the Pi, watch it on your laptop
============================================================================
Run this ON YOUR LAPTOP (not the Pi). It:

    1. SSHes into the Pi and starts gesture/run_gesture_mentorpi.py there
       with --preview (its output shows up in this terminal)
    2. opens a window here with the robot's annotated live feed, drawn
       like qr_priority_live.py (faces, badge, hands, gesture, intent)
    3. on q in the window or Ctrl+C here, stops the runner on the Pi with
       a normal Ctrl+C, so it stops the wheels on the way out

Any option it doesn't know is passed to the runner:

    cd ~/chai-center-project-code/MentorPi-code/tools
    python3 robot_preview.py                     # mock mode, no motion
    python3 robot_preview.py --real-motion       # wheels turn!
    python3 robot_preview.py --real-motion --debug
    python3 robot_preview.py --host pi@10.0.0.5  # another robot

Needs opencv-python here and the repo at ~/MentorPi-code on the Pi (see
README.md). SSH asks for the Pi's password once per run; run
`ssh-copy-id pi@192.168.149.1` once to stop it asking.
============================================================================
"""

import argparse
import os
import shlex
import socket
import subprocess
import sys
import time

import cv2

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from latest_frame import LatestFrameReader

DEFAULT_HOST = "pi@192.168.149.1"
DEFAULT_PORT = 5001            # PREVIEW_PORT in robot/preview_server.py
REMOTE_REPO  = "~/MentorPi-code"
REMOTE_VENV  = "~/va_pipeline_venv"
CONNECT_WAIT_S = 90            # Pi start-up (models, camera) can be slow
STOP_WAIT_S    = 10            # time the runner gets to stop the wheels
STREAM_GAP_S   = 8             # reconnect after this long without a frame
                               # (the real-motion countdown sends none for 5 s)
VIEWER_LOST_S  = 15            # runner stops itself after this long unwatched
                               # (> STREAM_GAP_S + reconnect time)
WINDOW = "MentorPi preview (q to quit)"


def parse_args():
    p = argparse.ArgumentParser(
        description="Start the gesture runner on the Pi and show its live "
                    "preview in a window here. Other options go to the runner.")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"SSH user@address of the Pi (default {DEFAULT_HOST})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"Preview port on the Pi (default {DEFAULT_PORT})")
    return p.parse_known_args()


def ssh(sock, *args):
    """ssh command line that reuses the shared connection at sock."""
    return ["ssh", "-o", f"ControlPath={sock}", *args]


def open_master(host, sock):
    """One SSH connection for the whole run: the password is asked here,
    once, and the runner and the stop command reuse it."""
    print(f"[LAUNCH] Connecting to {host}...")
    r = subprocess.run(ssh(sock, "-o", "ControlMaster=yes",
                           "-o", "ConnectTimeout=10", "-fN", host))
    return r.returncode == 0


def close_master(host, sock):
    subprocess.run(ssh(sock, "-O", "exit", host),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_runner(host, sock, port, runner_args):
    # --stop-without-viewer: if this script dies without stopping the runner
    # (crash, laptop closed, Wi-Fi gone), the runner stops itself
    cmd = (f"source {REMOTE_VENV}/bin/activate && cd {REMOTE_REPO}/gesture && "
           f"exec python3 run_gesture_mentorpi.py --preview {port} "
           f"--stop-without-viewer {VIEWER_LOST_S} "
           + " ".join(shlex.quote(a) for a in runner_args))
    # -tt gives the runner a terminal on the Pi, so writing Ctrl+C (\x03)
    # to ssh's stdin reaches it as a real Ctrl+C. Own session: a Ctrl+C
    # typed here goes to this script only, which then stops the runner.
    return subprocess.Popen(ssh(sock, "-tt", host, cmd),
                            stdin=subprocess.PIPE, start_new_session=True)


def stop_runner(proc):
    if proc.poll() is not None:
        return
    print("\n[LAUNCH] Stopping the runner on the Pi (wheels stop)...")
    try:
        proc.stdin.write(b"\x03")
        proc.stdin.flush()
        proc.wait(timeout=STOP_WAIT_S)
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
        # Hang up instead; the runner treats SIGHUP like Ctrl+C
        proc.terminate()
        try:
            proc.wait(timeout=STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()


def open_stream(address, port, proc):
    """Wait for the runner's preview server; None if the runner exited."""
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < CONNECT_WAIT_S:
        try:   # quiet check first; OpenCV logs every refused connection
            socket.create_connection((address, port), timeout=1.0).close()
        except OSError:
            time.sleep(1.0)
            continue
        cap = cv2.VideoCapture(f"http://{address}:{port}/stream")
        if cap.isOpened():
            return LatestFrameReader(cap)   # never fall behind the stream
        cap.release()
        time.sleep(1.0)
    return None


def show(address, port, proc):
    """Show the feed until q, Ctrl+C or the runner exits."""
    cap = open_stream(address, port, proc)
    if cap is None:
        if proc.poll() is None:
            print(f"[LAUNCH] No preview from {address}:{port} after "
                  f"{CONNECT_WAIT_S} s.")
        return
    print("[LAUNCH] Showing the robot's camera (q in the window to quit, "
          "s to save the view to your Desktop)")
    last_frame_t = time.time()
    last_frame = None
    try:
        while proc.poll() is None:
            ok, frame = cap.read(timeout=0.1)
            if ok:
                last_frame_t, last_frame = time.time(), frame
                cv2.imshow(WINDOW, frame)
            elif time.time() - last_frame_t > STREAM_GAP_S:
                # Stream dropped: reconnect while the runner is alive
                cap.release()
                cap = open_stream(address, port, proc)
                if cap is None:
                    return
                last_frame_t = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord("s") and last_frame is not None:
                path = os.path.expanduser(
                    time.strftime("~/Desktop/robot_frame_%H%M%S.png"))
                cv2.imwrite(path, last_frame)
                print(f"[LAUNCH] Saved the robot's view to {path}")
    finally:
        if cap is not None:
            cap.release()


def main():
    args, runner_args = parse_args()
    address = args.host.split("@")[-1]
    # Short path on purpose: SSH caps socket paths at ~100 characters
    sock = f"/tmp/mentorpi-ssh-{os.getpid()}"

    if not open_master(args.host, sock):
        print(f"[LAUNCH] Could not SSH to {args.host}. Are you on the robot's "
              "Wi-Fi, and is the address right (--host)?")
        sys.exit(1)
    proc = start_runner(args.host, sock, args.port, runner_args)
    try:
        show(address, args.port, proc)
    except KeyboardInterrupt:
        pass
    finally:
        stop_runner(proc)
        cv2.destroyAllWindows()
        close_master(args.host, sock)
    print("[LAUNCH] Done.")


if __name__ == "__main__":
    main()
