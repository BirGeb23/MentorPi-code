#!/usr/bin/env python3
"""
============================================================================
spot_tag_control.py  —  Only the person wearing the AprilTag controls Spot
============================================================================
Used by spot_gesture_tag.py (the AprilTag / fiducial version).

The controller wears one of the lab's AprilTag fiducials (family 36h11, the
same family Spot uses) on a lanyard at chest height, facing Spot.

  * Spot only reacts while the tag is visible. No tag (or tag hidden for
    more than TAG_RECENT_S) = nobody is in control, Spot ignores gestures.
  * The controller is the face directly above the tag. That is re-checked
    every time the tag is seen, so control can't drift to someone else.
  * Every face in view gets its own track ID (FaceTracker). Between tag
    sightings (e.g. an arm briefly covers the tag) Spot keeps following the
    controller's track; another person's face, which has its own ID, can
    never take over. If the controller's track is lost, nobody is in
    control until the tag is seen again.
  * Hand check: a hand only counts if it is in the controller's zone, about
    the right size for their face, AND closer to the controller's face
    than to anyone else's face.
  * Handing over control = handing over the tag.

With --tag-ids, only those tag IDs count (recommended: the lab may have
fiducials on walls or docks). Without it, any 36h11 tag counts, and the
status line lists the IDs it sees so you can pick one.
============================================================================
"""

import cv2

from spot_gestures import centre, size, hand_belongs, FaceTracker

TAG_RECENT_S = 1.5        # tag unseen longer than this -> nobody in control
# Where the tag must be relative to a face to belong to it (in face sizes):
TAG_X_FACES = 1.5         # tag centre within this many face widths left/right
TAG_Y_MIN_FACES = 0.5     # tag centre at least this far below the face's bottom edge
TAG_Y_MAX_FACES = 5.0     # ...and at most this far below it (chest/belly height)


# ---------------------------------------------------------------------------
# Tag detection
# ---------------------------------------------------------------------------
class TagDetector:
    """AprilTag 36h11 detection with OpenCV's ArUco module (OpenCV 4.7+
    API, with a fallback for older versions)."""

    def __init__(self, allowed_ids=None):
        self.allowed = set(allowed_ids) if allowed_ids else None
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") \
            else cv2.aruco.DetectorParameters_create()
        self._new_api = hasattr(cv2.aruco, "ArucoDetector")
        if self._new_api:
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)
        self.params = params
        self.seen_ids = set()     # every 36h11 ID seen (for the status line)

    def detect(self, frame_bgr):
        """Returns [(tag_id, centre_x, centre_y, side_px, corners)] for allowed tags."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary,
                                                      parameters=self.params)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            tag_id = int(i)
            self.seen_ids.add(tag_id)
            if self.allowed is not None and tag_id not in self.allowed:
                continue
            pts = c.reshape(-1, 2)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            side = float(max(pts[:, 0].max() - pts[:, 0].min(),
                             pts[:, 1].max() - pts[:, 1].min()))
            out.append((tag_id, cx, cy, side, pts))
        return out


def tag_belongs_to(tag, face_bb):
    """Is this tag where a badge worn by this person would be?"""
    _, tcx, tcy, _, _ = tag
    fx, fy, fw, fh = face_bb
    s = size(face_bb)
    if abs(tcx - (fx + fw / 2.0)) > TAG_X_FACES * s:
        return False
    below = tcy - (fy + fh)
    return TAG_Y_MIN_FACES * s <= below <= TAG_Y_MAX_FACES * s


# ---------------------------------------------------------------------------
# Controller (same interface as CentreLock in spot_gestures.py)
# ---------------------------------------------------------------------------
class TagControl:

    def __init__(self, allowed_ids=None):
        self.tags = TagDetector(allowed_ids)
        self.faces = FaceTracker()
        self.controller_id = None
        self.controller_tag = None
        self.target = None
        self.last_seen = -1e9
        self.last_tag = -1e9
        self.current_tags = []
        self.events = []
        self.armed = False
        ids = sorted(allowed_ids) if allowed_ids else "any 36h11 tag"
        self.intro = (f"[CTRL] AprilTag mode: only the person wearing tag {ids} controls Spot.")

    def update(self, frame, faces, now, frame_w, frame_h):
        self.current_tags = self.tags.detect(frame)
        self.faces.update(faces, now)
        visible = self.faces.visible(now)

        # 1. A tag in view: the controller is the face directly above it.
        if self.current_tags:
            best = None
            for tag in self.current_tags:
                for tid, bb in visible.items():
                    if tag_belongs_to(tag, bb):
                        d = abs(tag[1] - centre(bb)[0])
                        if best is None or d < best[0]:
                            best = (d, tid, tag)
            if best is not None:
                _, tid, tag = best
                if tid != self.controller_id:
                    self.events.append(f"Tag {tag[0]} seen: controller is face #{tid}")
                self.controller_id, self.controller_tag = tid, tag[0]
                self.last_tag = now

        # 2. Follow the controller's own face track (no jumping to others).
        if self.controller_id is not None and self.controller_id not in self.faces.tracks:
            self.events.append("Controller's face lost: show the tag to take control again")
            self.controller_id = None

        was_armed = self.armed
        self.armed = (self.controller_id is not None and now - self.last_tag <= TAG_RECENT_S)
        if was_armed and not self.armed and self.controller_id is not None:
            self.events.append("Tag not visible: Spot ignores gestures until it sees the tag")
        if not was_armed and self.armed:
            self.events.append(f"ARMED by tag {self.controller_tag}")

        if self.controller_id is not None and self.controller_id in visible:
            self.target = visible[self.controller_id]
            self.last_seen = now
            return self.target
        if self.controller_id is None:
            self.target = None
        return None

    def present(self, now):
        """Controller in view AND tag recently seen: gestures count."""
        return (self.target is not None and self.armed
                and now - self.last_seen <= TAG_RECENT_S)

    def hand_ok(self, hand, all_faces):
        """In the controller's zone, right size, and closer to the
        controller's face than to anyone else's."""
        return self.target is not None and hand_belongs(hand, self.target, all_faces)

    def status(self, now):
        if self.present(now):
            return f"CONTROLLER: tag {self.controller_tag} (face #{self.controller_id})"
        if self.controller_id is not None:
            return "TAG HIDDEN: show the tag"
        seen = sorted(self.tags.seen_ids)
        hint = f" (36h11 IDs seen so far: {seen})" if seen and self.tags.allowed is None else ""
        return "NO TAG: wear the tag facing Spot" + hint

    def draw(self, view):
        """Yellow outline + ID on every allowed tag in view."""
        for tag_id, cx, cy, _, pts in self.current_tags:
            cv2.polylines(view, [pts.astype("int32")], True, (0, 255, 255), 2)
            cv2.putText(view, f"tag {tag_id}", (int(cx) - 30, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def reset(self, now):
        """Operator reset (r): forget the controller; the tag re-assigns it."""
        self.controller_id = None
        self.target = None
        self.armed = False
        self.events.append("Control reset (whoever wears the tag will be picked up again)")
