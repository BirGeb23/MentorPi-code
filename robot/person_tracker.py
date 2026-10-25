#!/usr/bin/env python3
"""
============================================================================
person_tracker.py  —  MediaPipe face-based person tracker
============================================================================
Wraps MediaPipe Face Detection to give us the three things the adapter
needs to know about the person:

    - visible:      is a face in the camera frame right now?
    - bearing_rad:  angle from camera center (positive = right of center)
    - distance_m:   estimated distance from face bounding box size

The distance estimate uses a simple inverse-height formula:
    distance_m ≈ K / face_height_pixels

K needs calibration against your specific camera + typical face size,
but the default (K=120) is a reasonable starting point for a 640x480
webcam-style camera and average adult face.

Requires: mediapipe, opencv-python (both already installed on the Pi).
============================================================================
"""

import math

import cv2
import mediapipe as mp


class PersonTracker:
    """
    Face-detection-based person tracker. One method: process_frame().

    Distance measurement (in order of preference):
        1. If a LidarDistanceReader is passed in, use the LiDAR range at
           the face's bearing. This gives ±3 cm accuracy from a real
           physical sensor and is strongly preferred.
        2. Otherwise, fall back to the inverse face-height formula
           (distance_m = k / face_height_pixels). ~20-30% error but works
           anywhere the camera works — useful for testing off-robot.
    """

    def __init__(self,
                 camera_hfov_deg: float = 60.0,
                 distance_k: float = 120.0,
                 min_confidence: float = 0.5,
                 model_selection: int = 0,
                 lidar_reader=None):
        """
        Args:
            camera_hfov_deg: horizontal field of view in degrees.
                             ~60 for the MentorPi's Angstrong depth camera.
            distance_k:      face-based fallback formula constant.
                             distance_m = distance_k / face_height_pixels
                             Only used if lidar_reader is None or returns None.
            min_confidence:  MediaPipe detection confidence threshold (0-1).
            model_selection: 0 = short-range (up to ~2m), 1 = full-range (~5m).
                             Use 0 for close-quarter interaction, 1 if further.
            lidar_reader:    Optional LidarDistanceReader for accurate distance.
                             When provided, LiDAR distance replaces the noisy
                             face-based estimate. Falls back to face-based if
                             the LiDAR returns None (out-of-range, occluded).
        """
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_confidence,
        )
        self.hfov_rad = math.radians(camera_hfov_deg)
        self.k = distance_k
        self.lidar = lidar_reader

    def process_frame(self, frame_bgr):
        """
        Process one BGR frame. Returns a dict:
            {
                'visible':         bool,
                'bearing_rad':     float or None (positive = right of center),
                'distance_m':      float or None,
                'distance_source': 'lidar' or 'face_bbox' or None,
                'confidence':      float or None (detection confidence 0-1),
                'bbox':            (x, y, w, h) in pixels, or None,
            }
        """
        h, w = frame_bgr.shape[:2]

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)

        if not results.detections:
            return {
                'visible':         False,
                'bearing_rad':     None,
                'distance_m':      None,
                'distance_source': None,
                'confidence':      None,
                'bbox':            None,
            }

        # Multiple faces? Take the largest (assumed closest).
        best = None
        best_area = 0.0
        for det in results.detections:
            b = det.location_data.relative_bounding_box
            area = b.width * b.height
            if area > best_area:
                best_area = area
                best = det

        b = best.location_data.relative_bounding_box

        # --- Bearing: horizontal angle from camera center ---
        # Face center x in normalized coords (0..1)
        face_cx_norm = b.xmin + b.width / 2.0
        # Offset from center: -0.5 (left edge) to +0.5 (right edge)
        offset_from_center = face_cx_norm - 0.5
        # Scale by full FOV: at frame edges, angle = ±hfov/2
        bearing_rad = offset_from_center * self.hfov_rad

        # --- Distance: LiDAR + face-bbox fusion ---
        # We always compute the face-bbox estimate first, because it acts as
        # both a hint for the LiDAR (to disambiguate person from other objects
        # at the same bearing) and as a fallback if the LiDAR can't confirm.
        face_height_px = b.height * h
        face_estimate  = self.k / face_height_px if face_height_px > 0 else 5.0

        distance_m      = None
        distance_source = None

        if self.lidar is not None:
            # Pass the face-based estimate as a hint. LiDAR returns the
            # reading closest to that hint (within tolerance), so it picks
            # the person's leg even if a chair or wall shares the bearing.
            lidar_d = self.lidar.get_distance_at(
                bearing_rad,
                hint_distance_m=face_estimate,
                hint_tolerance_m=1.0,   # ±1m around face estimate
            )
            if lidar_d is not None:
                distance_m      = lidar_d
                distance_source = 'lidar'

        if distance_m is None:
            # Fallback: face-based estimate (noisy ±20-30% but always available)
            distance_m      = face_estimate
            distance_source = 'face_bbox'

        # --- Pixel bbox for optional debug rendering ---
        bbox_px = (
            int(b.xmin * w),
            int(b.ymin * h),
            int(b.width * w),
            int(b.height * h),
        )

        confidence = best.score[0] if best.score else 0.0

        return {
            'visible':         True,
            'bearing_rad':     bearing_rad,
            'distance_m':      distance_m,
            'distance_source': distance_source,
            'confidence':      confidence,
            'bbox':            bbox_px,
        }

    def close(self):
        """Release MediaPipe resources."""
        try:
            self.detector.close()
        except Exception:
            pass
