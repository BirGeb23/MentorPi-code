#!/usr/bin/env python3
"""
============================================================================
spot_skeleton.py  —  Full-body skeleton check (spot_gesture_tag.py --skeleton)
============================================================================
Without this, a hand counts for the controller if it is near the
controller's face. That lets a bystander "steal" a gesture by putting their
hand next to the controller's face or tag.

With --skeleton, MediaPipe's Pose Landmarker finds the body skeleton of every
person around the controller (33 points each: head, shoulders, elbows,
wrists, hips, ...). Then:

  1. The CONTROLLER'S SKELETON is the one whose head is on the controller's
     face (the face above tag 2) and, whenever the tag is in view, whose
     chest/torso contains the tag.
  2. A hand only counts if its wrist sits at one of that skeleton's wrists,
     AND the shoulder -> elbow -> wrist chain of that arm is visible.
     A bystander's hand next to the controller's face or tag is not at the
     end of the controller's arm, so it is ignored.
  3. No controller skeleton found (e.g. body hidden) = gestures pause until
     it is found again. Spot never acts on a hand it can't attach to the
     controller's arm.

Known limits: if bodies overlap (a bystander's arm reaching across the
controller), the pose model can attach the wrong arm; and a bystander's hand
held right beside the controller's own hand can still pass.

Model file (download once into the spot_cam folder), "lite" recommended:
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
============================================================================
"""

import math
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from spot_gestures import centre, size

# Pose landmark numbers (MediaPipe Pose, 33 points)
NOSE = 0
HEAD = range(0, 11)                       # nose, eyes, ears, mouth
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
ARMS = ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR))

# Area around the controller searched for skeletons (in face sizes). Wider
# and taller than the gesture crop so bystanders' bodies are seen whole.
POSE_WIDTH_FACES = 10.0
POSE_UP_FACES = 1.5
POSE_DOWN_FACES = 7.0
POSE_MIN_PX = 256

HEAD_MATCH_FACES = 0.6    # skeleton head within this of the controller's face centre
WRIST_MATCH_FACES = 0.75  # hand wrist within this of the skeleton's wrist
TORSO_MARGIN = 0.25       # tag may be this fraction outside the torso box
MIN_VISIBILITY = 0.5      # shoulder/elbow/wrist must be at least this visible
STALE_S = 0.5             # skeleton older than this = not usable


def _vis(lm):
    v = getattr(lm, "visibility", None)
    return 1.0 if v is None else float(v)


class SkeletonCheck:
    def __init__(self, model_path, num_poses=3, every=1):
        self.landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=num_poses,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5))
        self.every = max(1, int(every))
        self.frames = 0
        self.poses = []            # [[(x, y, visibility) x33], ...] full-frame px
        self.ctrl = None           # controller's skeleton (one entry of self.poses)
        self.ctrl_time = -1e9
        self.face_size = 1.0
        self.time_total = 0.0
        self.runs = 0
        self.rejected = 0          # hands rejected by the skeleton check
        self.accepted = 0
        self._was_found = None
        self.region_box = None     # area last given to the pose model (preview: magenta)

    # ------------------------------------------------------------------
    def _region(self, face_bb, w, h):
        cx, _ = centre(face_bb)
        s = size(face_bb)
        half = max(POSE_WIDTH_FACES * s, POSE_MIN_PX) / 2.0
        y0 = face_bb[1] - POSE_UP_FACES * s
        y1 = max(face_bb[1] + POSE_DOWN_FACES * s, y0 + POSE_MIN_PX)
        return (int(max(0, cx - half)), int(max(0, y0)),
                int(min(w, cx + half)), int(min(h, y1)))

    def update(self, frame, control, now, in_control):
        """Run the pose model around the controller and pick their skeleton."""
        self.frames += 1
        target = control.target if in_control else None
        if target is None:
            self.poses, self.ctrl, self.region_box = [], None, None
            return
        if (self.frames - 1) % self.every:
            return                              # reuse the last result
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._region(target, w, h)
        self.region_box = (x0, y0, x1, y1)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return
        t = time.time()
        rgb = np.ascontiguousarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        self.time_total += time.time() - t
        self.runs += 1
        cw, ch = x1 - x0, y1 - y0
        self.poses = [[(x0 + lm.x * cw, y0 + lm.y * ch, _vis(lm)) for lm in pose]
                      for pose in result.pose_landmarks]
        tags = [(tg[1], tg[2]) for tg in getattr(control, "current_tags", [])]
        self.face_size = size(target)
        self.ctrl = self.pick_controller(self.poses, target, tags)
        if self.ctrl is not None:
            self.ctrl_time = now
        found = self.ctrl is not None
        if found != self._was_found:
            self._was_found = found
            control.events.append("Controller's skeleton found" if found else
                                  "Controller's skeleton not found: gestures paused")

    @staticmethod
    def pick_controller(poses, face_bb, tags):
        """The skeleton whose head is on the controller's face and (when a tag
        is in view) whose torso contains the tag. None if no skeleton fits."""
        fcx, fcy = centre(face_bb)
        s = size(face_bb)
        best, best_d = None, None
        for pose in poses:
            head = [(x, y) for i, (x, y, v) in enumerate(pose) if i in HEAD and v >= MIN_VISIBILITY]
            if not head:
                continue
            hx = sum(p[0] for p in head) / len(head)
            hy = sum(p[1] for p in head) / len(head)
            d = math.hypot(hx - fcx, hy - fcy)
            if d > HEAD_MATCH_FACES * s:
                continue
            if tags and not any(SkeletonCheck.tag_on_torso(pose, t, s) for t in tags):
                continue
            if best_d is None or d < best_d:
                best, best_d = pose, d
        return best

    @staticmethod
    def tag_on_torso(pose, tag_xy, face_s):
        """Is the tag on this skeleton's chest/belly (shoulders down to hips)?"""
        sh = [pose[i] for i in (L_SH, R_SH)]
        hips = [pose[i] for i in (L_HIP, R_HIP) if pose[i][2] >= MIN_VISIBILITY]
        xs = [p[0] for p in sh]
        top = min(p[1] for p in sh)
        # Hips often fall outside the view; then assume ~3 face sizes of torso.
        bottom = max(p[1] for p in hips) if hips else top + 3.0 * face_s
        width = max(max(xs) - min(xs), face_s)
        m = TORSO_MARGIN * width
        return (min(xs) - m <= tag_xy[0] <= max(xs) + m and
                top - m <= tag_xy[1] <= bottom + m)

    # ------------------------------------------------------------------
    def hand_ok(self, hand, now, gesture=None):
        """hand = (cx, cy, size, wrist_x, wrist_y). True only if the wrist is
        at one of the controller skeleton's wrists, with that arm visible."""
        ok = self._hand_ok(hand, now)
        if gesture not in (None, "None"):
            if ok:
                self.accepted += 1
            else:
                self.rejected += 1
        return ok

    def _hand_ok(self, hand, now):
        if self.ctrl is None or now - self.ctrl_time > STALE_S or len(hand) < 5:
            return False
        wx, wy = hand[3], hand[4]
        for sh, el, wr in ARMS:
            if min(self.ctrl[i][2] for i in (sh, el, wr)) < MIN_VISIBILITY:
                continue
            px, py, _ = self.ctrl[wr]
            if math.hypot(wx - px, wy - py) <= WRIST_MATCH_FACES * self.face_size:
                return True
        return False

    # ------------------------------------------------------------------
    def status(self, now):
        ms = 1000 * self.time_total / max(self.runs, 1)
        state = ("skeleton OK" if self.ctrl is not None and now - self.ctrl_time <= STALE_S
                 else "NO SKELETON (gestures paused)")
        return (f"{state}, pose model {ms:.0f} ms/run, {len(self.poses)} skeleton(s), "
                f"gesture hands accepted {self.accepted} / rejected {self.rejected}")

    def summary(self):
        ms = 1000 * self.time_total / max(self.runs, 1)
        return (f"Skeleton check: pose model {ms:.0f} ms/run over {self.runs} runs; "
                f"gesture hands accepted {self.accepted}, rejected {self.rejected} "
                f"(rejected = not at the end of the controller's arm)")

    def draw(self, view):
        """Magenta box = area given to the pose model. Grey skeletons for
        everyone, green arms/torso for the controller."""
        if self.region_box is not None:
            x0, y0, x1, y1 = self.region_box
            cv2.rectangle(view, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 255), 1)
        bones = [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
                 (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP)]
        for pose in self.poses:
            colour, thick = ((0, 255, 0), 3) if pose is self.ctrl else ((160, 160, 160), 1)
            for a, b in bones:
                if pose[a][2] >= MIN_VISIBILITY and pose[b][2] >= MIN_VISIBILITY:
                    cv2.line(view, (int(pose[a][0]), int(pose[a][1])),
                             (int(pose[b][0]), int(pose[b][1])), colour, thick)
            if pose is self.ctrl:
                r = int(WRIST_MATCH_FACES * self.face_size)
                for wr in (L_WR, R_WR):
                    if pose[wr][2] >= MIN_VISIBILITY:
                        cv2.circle(view, (int(pose[wr][0]), int(pose[wr][1])), r, (0, 255, 0), 1)

    def close(self):
        self.landmarker.close()
