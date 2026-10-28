#!/usr/bin/env python3
"""
============================================================================
lidar_reader.py  —  Subscribe to /scan via rosbridge and look up distances
============================================================================
Provides accurate distance measurements from the MentorPi's ORADAR MS200
LiDAR, replacing the noisy face-bounding-box estimate previously used by
PersonTracker.

Architecture:
    - Connects to rosbridge_websocket (the same bridge the adapter uses)
    - Subscribes to /scan (sensor_msgs/LaserScan)
    - Runs a background thread that reads scans and caches the latest one
    - Exposes get_distance_at(bearing_rad) for the tracker to call

Design decisions:
    - Uses its OWN WebSocket connection separate from the adapter's, so
      neither interferes with the other's message flow.
    - Averages (medians) a small angular window around the target bearing
      (default ±2.5°) so we get a stable reading of the person's legs even
      if one laser beam happens to slip past.
    - Filters out invalid readings (inf, NaN, out-of-range) before averaging.

Sensor details (ORADAR MS200):
    - 360° 2D scanning LiDAR
    - Range: 0.1 to 12 m
    - Accuracy: ±3 cm (~10x better than our face-based estimate)
    - Scan rate: ~10-12 Hz (so a fresh scan is always within 100 ms)
    - Sits at torso height (~35 cm), so it sees legs, not faces
============================================================================
"""

import json
import math
import threading
import time


class LidarDistanceReader:
    """
    Cached LaserScan reader. `start()` subscribes and begins caching scans
    in the background; `get_distance_at(bearing_rad)` returns the median
    distance in a small window around that bearing, or None if we don't
    have a fresh scan or no valid readings at that angle.
    """

    def __init__(self,
                 rosbridge_host: str = "192.168.149.1",
                 rosbridge_port: int = 9090,
                 topic: str = "/scan_raw",
                 window_deg: float = 10.0,
                 stale_ms: float = 500.0,
                 aggregation: str = "min"):
        """
        Args:
            rosbridge_host, rosbridge_port: rosbridge_websocket location
            topic:  LaserScan topic (MentorPi MS200 publishes to /scan_raw
                    by default; standard ROS 2 convention is /scan)
            window_deg: total angular window (in degrees) around the target
                        bearing. 10° = ±5°. Wider = more likely to catch a
                        leg; narrower = more precise but might miss.
            stale_ms: scans older than this are ignored (returns None).
            aggregation: how to combine multiple beams in the window.
                        "min" (default): closest valid reading — best for
                            person tracking because a person's LEG is always
                            in FRONT of whatever's behind them. Beams that
                            slip between the legs to a wall get ignored.
                        "median": robust average — good for general obstacles
                            but with a person, half the beams slip through
                            leg-gaps to the wall behind, pulling the median
                            away from the person's real distance.
        """
        if aggregation not in ("min", "median"):
            raise ValueError(f"aggregation must be 'min' or 'median', "
                             f"got {aggregation!r}")
        self._url          = f"ws://{rosbridge_host}:{rosbridge_port}"
        self._topic        = topic
        self._half_window  = math.radians(window_deg / 2.0)
        self._stale_secs   = stale_ms / 1000.0
        self._aggregation  = aggregation

        self._latest_scan   = None
        self._latest_ts     = 0.0
        self._lock          = threading.Lock()
        self._running       = False
        self._ws            = None
        self._thread        = None

    # ---------------------------------------------------------------
    # 1) START — open the WebSocket and start the background thread
    # ---------------------------------------------------------------
    def start(self):
        """Connect to rosbridge, subscribe to /scan, start the reader thread."""
        try:
            import websocket
        except ImportError:
            raise RuntimeError(
                "websocket-client is required. Install with: "
                "pip install websocket-client"
            )

        print(f"[LIDAR] Connecting to {self._url} for {self._topic}...")
        self._ws = websocket.create_connection(self._url, timeout=5)

        # Subscribe to the LaserScan topic
        self._ws.send(json.dumps({
            "op":    "subscribe",
            "topic": self._topic,
            "type":  "sensor_msgs/LaserScan",
            # throttle_rate is in ms; 100ms = 10Hz is plenty for our
            # cycle-based use case and reduces WebSocket traffic.
            "throttle_rate": 100,
        }))
        print(f"[LIDAR] Subscribed to {self._topic}, starting reader thread...")

        self._running = True
        self._thread  = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        # Give the first scan a chance to arrive before we return
        for _ in range(20):    # up to ~1s
            time.sleep(0.05)
            with self._lock:
                if self._latest_scan is not None:
                    n = len(self._latest_scan.get("ranges", []))
                    print(f"[LIDAR] First scan received ({n} ranges). Ready.")
                    return
        print("[LIDAR] Warning: no scan received in first second. "
              "Continuing anyway (may work later).")

    # ---------------------------------------------------------------
    # 2) BACKGROUND READER — pull scan messages off the WebSocket
    # ---------------------------------------------------------------
    def _reader_loop(self):
        while self._running:
            try:
                raw = self._ws.recv()
                if not raw:
                    continue
                msg = json.loads(raw)
                if msg.get("op") == "publish" and msg.get("topic") == self._topic:
                    with self._lock:
                        self._latest_scan = msg["msg"]
                        self._latest_ts   = time.time()
            except Exception as e:
                if self._running:
                    print(f"[LIDAR] Reader loop error: {e}")
                    time.sleep(0.5)   # brief pause before retry / exit

    # ---------------------------------------------------------------
    # 3) LOOKUP — get the distance at a given bearing
    # ---------------------------------------------------------------
    def get_distance_at(self,
                        bearing_rad: float,
                        angle_offset_rad: float = 0.0,
                        hint_distance_m: float = None,
                        hint_tolerance_m: float = 1.0):
        """
        Return the LiDAR distance (meters) in a small angular window around
        `bearing_rad`. Returns None if:
            - no scan has been received yet
            - the newest scan is stale
            - all readings in the window are invalid (inf, NaN, out-of-range)
            - a hint is given and no readings fall within its tolerance

        Bearing conventions:
          - `bearing_rad` comes in the camera/tracker convention:
                0 = forward, positive = right, negative = left
          - The MentorPi MS200 scan uses a [0, 2π] convention (0° to 360°);
            this function handles the wrap.
          - `angle_offset_rad` compensates if the LiDAR's 0° isn't the
            robot's forward axis. Default 0.

        Distance disambiguation:
          - If `hint_distance_m` is provided, we return the LiDAR reading
            in the window that is CLOSEST to that hint (within
            hint_tolerance_m). This is how we disambiguate between the
            person and other objects at the same bearing — the face-based
            distance estimate is used as the hint, so LiDAR picks the leg
            cluster (close to hint) instead of a wall behind (too far)
            or a chair in front (also too far).
          - If no hint is provided, falls back to the configured aggregation
            ("min" or "median" — see __init__).
        """
        with self._lock:
            scan = self._latest_scan
            ts   = self._latest_ts

        if scan is None:
            return None
        if (time.time() - ts) > self._stale_secs:
            return None

        angle_min       = scan["angle_min"]
        angle_max       = scan["angle_max"]
        angle_increment = scan["angle_increment"]
        ranges          = scan["ranges"]
        range_min       = scan.get("range_min", 0.05)
        range_max       = scan.get("range_max", 12.0)

        if angle_increment <= 0 or len(ranges) == 0:
            return None

        # ------------------- Translate bearing into LiDAR frame -------------
        # 1) apply optional forward-alignment offset
        # 2) wrap negative bearings into the LiDAR's [0, 2π] range
        b = bearing_rad + angle_offset_rad
        two_pi = 2.0 * math.pi
        b = b % two_pi                     # wraps negatives into [0, 2π)

        target_low_raw  = b - self._half_window
        target_high_raw = b + self._half_window

        # If the window crosses the 0/2π boundary, split it into two
        # sub-windows that stay inside [angle_min, angle_max].
        sub_windows = []
        if target_low_raw < 0.0:
            sub_windows.append((0.0, target_high_raw))
            sub_windows.append((two_pi + target_low_raw, two_pi))
        elif target_high_raw > two_pi:
            sub_windows.append((target_low_raw, two_pi))
            sub_windows.append((0.0, target_high_raw - two_pi))
        else:
            sub_windows.append((target_low_raw, target_high_raw))

        # Collect valid readings across all sub-windows
        valid = []
        for lo, hi in sub_windows:
            # Clamp to scan bounds
            lo_c = max(lo, angle_min)
            hi_c = min(hi, angle_max)
            if hi_c < lo_c:
                continue
            idx_low  = int(round((lo_c - angle_min) / angle_increment))
            idx_high = int(round((hi_c - angle_min) / angle_increment))
            idx_low  = max(0, min(idx_low,  len(ranges) - 1))
            idx_high = max(0, min(idx_high, len(ranges) - 1))
            if idx_low > idx_high:
                continue

            for r in ranges[idx_low : idx_high + 1]:
                if r is None:
                    continue
                try:
                    r_f = float(r)
                except (TypeError, ValueError):
                    continue
                if r_f != r_f:                       # NaN check
                    continue
                if r_f == float("inf") or r_f == float("-inf"):
                    continue
                if r_f < range_min or r_f > range_max:
                    continue
                valid.append(r_f)

        if not valid:
            return None

        # ---------- Hint-based disambiguation ----------
        # If a hint distance is provided (e.g., from face-bbox estimate),
        # find the reading closest to it. This filters out closer objects
        # (chairs, tables between person and robot) AND farther objects
        # (walls behind the person seen through leg-gaps).
        if hint_distance_m is not None:
            matching = [r for r in valid
                        if abs(r - hint_distance_m) <= hint_tolerance_m]
            if matching:
                # Return the reading closest to the hint. Ties broken by
                # closer distance (prefer the actual surface, not overshoot).
                best = min(matching, key=lambda r: (abs(r - hint_distance_m), r))
                return best
            # No LiDAR readings match the hint — return None and let the
            # caller fall back to the face-bbox estimate.
            return None

        # ---------- No hint: use configured aggregation ----------
        if self._aggregation == "min":
            # A person's leg is always in front of whatever's behind them.
            # Beams that slip past the leg to a wall return larger values;
            # min picks the leg, not the wall. Fails if something's CLOSER
            # than the person (chair in the way) — use hint mode for that.
            return min(valid)
        else:
            # Median = robust average, good for general obstacle sensing
            # but pulls away from a person's real distance when beams slip
            # through leg-gaps to the wall.
            valid.sort()
            n = len(valid)
            if n % 2 == 1:
                return valid[n // 2]
            else:
                return (valid[n // 2 - 1] + valid[n // 2]) / 2.0

    # ---------------------------------------------------------------
    # 3b) FIND CLOSEST — sweep all angles, return the nearest object
    # ---------------------------------------------------------------
    def find_closest(self, min_distance: float = 0.15, max_distance: float = 8.0):
        """
        Sweep every valid reading in the latest scan and return the (bearing,
        distance) of the CLOSEST object within the range window. Useful for
        finding out where a nearby object (like a person) is, without needing
        to know the camera bearing first.

        Returns (bearing_rad, distance_m) or None if no valid readings.
        The returned bearing is in the LiDAR's raw [0, 2π] frame.
        """
        with self._lock:
            scan = self._latest_scan
            ts   = self._latest_ts

        if scan is None or (time.time() - ts) > self._stale_secs:
            return None

        angle_min       = scan["angle_min"]
        angle_increment = scan["angle_increment"]
        ranges          = scan["ranges"]

        closest_d = None
        closest_i = None
        for i, r in enumerate(ranges):
            try:
                r_f = float(r) if r is not None else None
            except (TypeError, ValueError):
                continue
            if r_f is None or r_f != r_f:
                continue
            if r_f == float("inf") or r_f == float("-inf"):
                continue
            if r_f < min_distance or r_f > max_distance:
                continue
            if closest_d is None or r_f < closest_d:
                closest_d = r_f
                closest_i = i

        if closest_d is None:
            return None
        bearing = angle_min + closest_i * angle_increment
        return (bearing, closest_d)

    # ---------------------------------------------------------------
    # 4) STOP — unsubscribe and close cleanly
    # ---------------------------------------------------------------
    def stop(self):
        """Stop the background thread and close the WebSocket."""
        self._running = False
        try:
            if self._ws:
                self._ws.send(json.dumps({
                    "op":    "unsubscribe",
                    "topic": self._topic,
                }))
                self._ws.close()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2)
        print("[LIDAR] Disconnected.")
