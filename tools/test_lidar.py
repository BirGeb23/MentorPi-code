#!/usr/bin/env python3
"""
============================================================================
test_lidar.py  —  Verify the LiDAR is reachable and check what it sees
============================================================================
Diagnostic tool to run BEFORE integrating LiDAR into the person tracker.

What it does:
    1. Connects to rosbridge and subscribes to /scan
    2. Waits for a scan to arrive
    3. Prints scan metadata (angle range, number of readings, etc.)
    4. Prints the distance at a set of key bearings so you can eyeball
       whether the LiDAR is reading correctly
    5. Every second, refreshes and shows the distance directly in front
       of the robot — walk in front of it and watch the number change

Usage (on the Pi, with venv active):
    source ~/va_pipeline_venv/bin/activate
    cd ~/MentorPi-code/tools
    python3 test_lidar.py
============================================================================
"""

import math
import os
import sys
import time

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from lidar_reader import LidarDistanceReader


def rad_to_deg(r):
    return r * 180.0 / math.pi


def main():
    print("=" * 72)
    print("  LIDAR DIAGNOSTIC TEST")
    print("=" * 72)

    reader = LidarDistanceReader(window_deg=5.0)
    try:
        reader.start()
    except Exception as e:
        print(f"[ERROR] Could not start LiDAR reader: {e}")
        print("  Make sure the MentorPi bringup is running and that")
        print("  rosbridge_websocket is up on port 9090.")
        return

    # Wait a moment for the first scan
    print("\n[TEST] Waiting 1s for the first scan...")
    time.sleep(1.0)

    # -------------------- Metadata dump --------------------
    with reader._lock:
        scan = reader._latest_scan
    if scan is None:
        print("[ERROR] No scan received. Is /scan being published?")
        print("  From inside the MentorPi container, try:")
        print("    ros2 topic list | grep scan")
        print("    ros2 topic hz /scan")
        reader.stop()
        return

    n_ranges  = len(scan["ranges"])
    a_min     = scan["angle_min"]
    a_max     = scan["angle_max"]
    a_inc     = scan["angle_increment"]
    r_min     = scan.get("range_min", "?")
    r_max     = scan.get("range_max", "?")

    print("\n[SCAN METADATA]")
    print(f"    angle_min       = {a_min:+.4f} rad  ({rad_to_deg(a_min):+.1f}°)")
    print(f"    angle_max       = {a_max:+.4f} rad  ({rad_to_deg(a_max):+.1f}°)")
    print(f"    total coverage  = {rad_to_deg(a_max - a_min):.1f}°")
    print(f"    angle_increment = {a_inc:.5f} rad   ({rad_to_deg(a_inc):.3f}° per beam)")
    print(f"    number of beams = {n_ranges}")
    print(f"    range_min/max   = {r_min} to {r_max} m")

    # -------------------- Full 360° sweep --------------------
    print("\n[FULL 360° SWEEP]")
    print("  Distance at each 30° step around the robot.")
    print("  In our convention: 0° = tracker 'forward'.")
    print()
    print(f"    {'Bearing':>10}   {'Distance':>10}")
    print(f"    {'-------':>10}   {'--------':>10}")
    for deg in range(-180, 181, 30):
        b_rad = math.radians(deg)
        d = reader.get_distance_at(b_rad)
        if d is None:
            print(f"    {deg:>+9}°   (no valid reading)")
        else:
            print(f"    {deg:>+9}°   {d:>7.2f} m")

    # -------------------- Closest object finder --------------------
    print("\n[CLOSEST OBJECT]")
    print("  Sweeps ALL angles and finds the closest thing to the robot.")
    print("  If YOU are the closest thing, this tells us which raw LiDAR")
    print("  angle corresponds to your actual position — great for figuring")
    print("  out whether the LiDAR's 0° points at the robot's forward.")
    print()
    result = reader.find_closest(min_distance=0.15, max_distance=5.0)
    if result is None:
        print("    No object found within 5 m.")
    else:
        raw_bearing, dist = result
        # Present in both raw LiDAR frame and camera-style [-180, +180]
        cam_deg = rad_to_deg(raw_bearing)
        if cam_deg > 180.0:
            cam_deg -= 360.0
        print(f"    Closest object: {dist:.2f} m at raw LiDAR angle "
              f"{rad_to_deg(raw_bearing):.1f}° "
              f"(camera convention: {cam_deg:+.1f}°)")
        print()
        print(f"    If you're standing directly in front of the robot at that")
        print(f"    distance, the camera-convention angle above tells us the")
        print(f"    LiDAR's forward-offset. Ideally it should be near 0°.")

    # -------------------- Live watch (10 seconds) --------------------
    print("\n[LIVE WATCH - 10 s]")
    print("  Stand ~1.5 m directly in front of the robot.")
    print("  Prints distance at 0° (assumed forward) AND the closest object.")
    print()
    end = time.time() + 10.0
    while time.time() < end:
        d_front  = reader.get_distance_at(0.0)
        closest  = reader.find_closest(min_distance=0.15, max_distance=5.0)
        front_str = f"{d_front:5.2f}m" if d_front is not None else "  no  "
        if closest is None:
            close_str = "  no closest    "
        else:
            close_bearing, close_dist = closest
            cam_deg = rad_to_deg(close_bearing)
            if cam_deg > 180.0:
                cam_deg -= 360.0
            close_str = f"{close_dist:5.2f}m at {cam_deg:+5.0f}°"
        print(f"    forward: {front_str}  |  closest: {close_str}    ",
              end="\r", flush=True)
        time.sleep(0.15)
    print()   # newline after live watch

    reader.stop()
    print("\n[TEST] Done.")
    print("  If forward shows your actual distance, LiDAR's 0° = robot's forward.")
    print("  If forward is wrong but closest is right, we have a LiDAR offset")
    print("  angle to configure. Send Claude the output to figure out the fix.")


if __name__ == "__main__":
    main()
