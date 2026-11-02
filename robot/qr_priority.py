#!/usr/bin/env python3
"""
============================================================================
qr_priority.py  —  QR badge priority: whose gestures the robot follows
============================================================================
A person wearing a priority QR code (badge, lanyard, card held at the
chest) gets priority for gesture control. The pieces:

    QRDetector        finds QR codes in a frame and keeps only the allowed
                      payloads.
    match_qr_to_face  the "imaginary line": links a QR to the face it hangs
                      below, going straight down from the bottom-centre of
                      the face box. Limits are measured in face sizes, so
                      the rule works the same at 1 m and at 2 m.
    PriorityState     remembers who has priority across frames: follows
                      that face as it moves, keeps priority for
                      PRIORITY_HOLD_S after their QR was last seen (QR reads
                      drop in and out at distance), and releases it when the
                      person leaves.
    assign_hand_to_face / gesture_allowed
                      link a recognized hand to a face, and decide whether
                      that hand's gesture should count.

No MediaPipe and no robot code in here: faces, QRs and hands come in as
plain pixel coordinates, so this works with any detector and can be tested
offline (tools/test_qr_priority.py). Live webcam viewer:
tools/qr_priority_live.py.
============================================================================
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


# ---------- Configuration ----------
# Only QR codes with one of these payloads give priority. None = any QR.
PRIORITY_PAYLOADS = {"MENTORPI-PRIORITY"}

# The imaginary line, in units of the face box (fw = width, fh = height),
# from the bottom-centre of the face box ("chin") to the QR centre:
LINE_MAX_DOWN_FACES     = 4.0   # QR centre at most 4 fh below the chin
LINE_MAX_SIDEWAYS_FACES = 1.0   # ...and at most 1 fw left/right of it
LINE_DOWN_WEIGHT        = 0.2   # cost: sideways offset counts 5x more than drop

PRIORITY_HOLD_S      = 4.0   # keep priority this long after the QR was last seen
FACE_LOST_S          = 1.0   # release priority if their face is gone this long
TRACK_MAX_JUMP_FACES = 1.5   # a face may move this many fw between updates

# A hand belongs to a face if its centre is within these limits of the face
# centre (face units again). Covers hands raised beside the head down to
# chest height.
HAND_MAX_SIDEWAYS_FACES = 2.5
HAND_MAX_UP_FACES       = 2.0
HAND_MAX_DOWN_FACES     = 6.0
HAND_UPDOWN_WEIGHT      = 0.3   # cost: prefer the face most directly above/below


@dataclass(eq=False)   # compared by identity: "is this the same detection"
class Face:
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0

    @property
    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def chin(self):
        """Bottom-centre of the face box, where the imaginary line starts."""
        return (self.x + self.w / 2.0, self.y + self.h)


@dataclass(eq=False)
class QRCode:
    text:    str
    center:  tuple               # (x, y) pixels
    corners: np.ndarray = None   # 4x2 pixel corners, for drawing


@dataclass(eq=False)
class QRLink:
    qr:   QRCode
    face: Face = None     # None if no face fits the line rule
    cost: float = None


# ============================================================================
# QR detection
# ============================================================================
class QRDetector:
    def __init__(self, payloads=PRIORITY_PAYLOADS):
        self.detector = cv2.QRCodeDetector()
        self.payloads = set(payloads) if payloads is not None else None

    def detect(self, frame_bgr):
        """Return a list of QRCode whose payload was read and is allowed."""
        try:
            ok, texts, points, _ = self.detector.detectAndDecodeMulti(frame_bgr)
        except cv2.error:
            return []
        if not ok or points is None:
            return []
        codes = []
        for text, pts in zip(texts, points):
            if not text:
                continue   # found a QR shape but couldn't read it
            if self.payloads is not None and text not in self.payloads:
                continue
            pts = np.asarray(pts, dtype=float).reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            codes.append(QRCode(text, (float(cx), float(cy)), pts))
        return codes


# ============================================================================
# The imaginary line: QR -> face
# ============================================================================
def match_qr_to_face(qr, faces):
    """Return (face, cost) for the face this QR hangs below, or (None, None).
    Lower cost = more directly below the chin."""
    qx, qy = qr.center
    best, best_cost = None, None
    for f in faces:
        fx, fy = f.chin
        dx, dy = abs(qx - fx), qy - fy
        if dy <= 0 or dy > LINE_MAX_DOWN_FACES * f.h:
            continue
        if dx > LINE_MAX_SIDEWAYS_FACES * f.w:
            continue
        cost = dx / f.w + LINE_DOWN_WEIGHT * dy / f.h
        if best_cost is None or cost < best_cost:
            best, best_cost = f, cost
    return best, best_cost


def link_qrs(codes, faces):
    """One QRLink per QR. If two QRs land on the same face, only the better
    match keeps it (one badge per person)."""
    links = [QRLink(qr, *match_qr_to_face(qr, faces)) for qr in codes]
    best_for_face = {}
    for link in links:
        if link.face is None:
            continue
        prev = best_for_face.get(id(link.face))
        if prev is None or link.cost < prev.cost:
            best_for_face[id(link.face)] = link
    for link in links:
        if link.face is not None and best_for_face[id(link.face)] is not link:
            link.face, link.cost = None, None
    return links


# ============================================================================
# Priority over time
# ============================================================================
class PriorityState:
    """Who has priority right now. Call update() every frame, read .face.

    Only one person holds priority. While they keep it, another badge
    wearer can't take it; it passes on once the holder's QR has been
    unseen for hold_s, or their face has been gone for face_lost_s."""

    def __init__(self, hold_s=PRIORITY_HOLD_S, face_lost_s=FACE_LOST_S):
        self.hold_s = hold_s
        self.face_lost_s = face_lost_s
        self._clear()

    def _clear(self):
        self.face        = None   # priority person's face (latest box seen)
        self.text        = None   # payload of their QR
        self.since       = None   # when they got priority
        self.qr_seen_t   = None   # last time their QR was linked to them
        self.face_seen_t = None   # last time their face was found

    @property
    def active(self):
        return self.face is not None

    def update(self, now, faces=None, codes=None):
        """
        now:   time in seconds
        faces: list of Face for this frame, or None if face detection
               didn't run this frame
        codes: list of QRCode for this frame, or None if QR detection
               didn't run this frame. QRs are only linked when faces are
               given too.
        Returns (links, events): the QRLinks for this frame (empty if
        nothing was linked) and a list of "released" / "granted" for logging.
        """
        events = []

        # 1. Follow the priority face to where it is in this frame
        if self.face is not None and faces is not None:
            nxt = self._follow(faces)
            if nxt is not None:
                self.face, self.face_seen_t = nxt, now

        # 2. Is their QR still there?
        links = []
        if codes is not None and faces is not None:
            links = link_qrs(codes, faces)
        linked = [l for l in links if l.face is not None]
        if self.face is not None and any(l.face is self.face for l in linked):
            self.qr_seen_t = now

        # 3. Expire
        if self.face is not None and (
                now - self.qr_seen_t > self.hold_s or
                now - self.face_seen_t > self.face_lost_s):
            self._clear()
            events.append("released")

        # 4. Nobody has priority: the closest (largest face) badge wearer gets it
        if self.face is None and linked:
            link = max(linked, key=lambda l: l.face.w * l.face.h)
            self.face, self.text = link.face, link.qr.text
            self.since = self.qr_seen_t = self.face_seen_t = now
            events.append("granted")

        return links, events

    def _follow(self, faces):
        """The face in this frame nearest the priority face, if it's close
        enough to be the same person. Two people crossing paths can swap."""
        px, py = self.face.center
        limit = TRACK_MAX_JUMP_FACES * self.face.w
        best, best_d = None, None
        for f in faces:
            fx, fy = f.center
            d = math.hypot(fx - px, fy - py)
            if d <= limit and (best_d is None or d < best_d):
                best, best_d = f, d
        return best


# ============================================================================
# Hands
# ============================================================================
def assign_hand_to_face(point, faces):
    """Which face a hand belongs to, or None. point = hand centre (x, y) in
    pixels, e.g. the mean of its 21 landmarks."""
    hx, hy = point
    best, best_cost = None, None
    for f in faces:
        fx, fy = f.center
        dx, dy = abs(hx - fx), hy - fy
        if dx > HAND_MAX_SIDEWAYS_FACES * f.w:
            continue
        if dy < -HAND_MAX_UP_FACES * f.h or dy > HAND_MAX_DOWN_FACES * f.h:
            continue
        cost = dx / f.w + HAND_UPDOWN_WEIGHT * abs(dy) / f.h
        if best_cost is None or cost < best_cost:
            best, best_cost = f, costq
    return best


def gesture_allowed(hand_face, priority, require_priority=False):
    """With a priority person, only a hand assigned to their face counts.
    Without one, every hand counts (same as run_gesture_mentorpi.py today),
    unless require_priority is set: then no hand counts."""
    if not priority.active:
        return not require_priority
    return hand_face is not None and hand_face is priority.face
