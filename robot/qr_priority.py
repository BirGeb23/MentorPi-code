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
import threading
from dataclasses import dataclass

import cv2
import numpy as np


# ---------- Configuration ----------
# Only QR codes with one of these payloads give priority. None = any QR.
PRIORITY_PAYLOADS = {"MENTORPI-PRIORITY", "https://chai.gsu.edu/"}

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
# OpenCV warns "ECI is not supported properly" on every read of a QR from a
# website generator (e.g. the CHAI badge) but still decodes it. The log
# level API moved in OpenCV 5.
_cv_log = getattr(getattr(cv2, "utils", None), "logging", None) or cv2
_CV_LOG_ERROR = 2   # errors only (3 = warnings)


# On the robot's 640x480 camera a badge at ~1 m is only ~100 px wide, too
# few pixels per square for OpenCV's reader. Given the faces, detect() reads
# just the area under each one (where a badge must be to count), enlarged
# to about this many pixels tall.
QR_ZONE_TARGET_PX = 900
QR_ZONE_MAX_SCALE = 4.0


class QRDetector:
    def __init__(self, payloads=PRIORITY_PAYLOADS, use_wechat=False):
        """use_wechat: also try OpenCV's WeChat reader (opencv-contrib-python)
        first, if installed. Under the robot's ceiling lights the badge comes
        out dark grey on a tilted angle; the built-in reader can't read
        that, WeChat can."""
        self.detector = cv2.QRCodeDetector()
        self.wechat = None
        if use_wechat and hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
            self.wechat = cv2.wechat_qrcode_WeChatQRCode()
        self.payloads = set(payloads) if payloads is not None else None

    @property
    def reader_name(self):
        return ("WeChat, then OpenCV built-in" if self.wechat is not None
                else "OpenCV built-in")

    def detect(self, frame_bgr, faces=None):
        """Return a list of QRCode whose payload was read and is allowed.
        With WeChat on, it reads the whole frame first. Otherwise (or if it
        finds nothing): faces = list of Face. If given, only the zone under
        each face is searched, enlarged so small badges still read. Without
        faces the whole frame is searched as is."""
        if self.wechat is not None:
            codes = self._read_wechat(frame_bgr)
            if codes:
                return codes
        if not faces:
            return self._read(frame_bgr)
        h, w = frame_bgr.shape[:2]
        codes = []
        for f in faces:
            cx, cy = f.chin
            half_w = (LINE_MAX_SIDEWAYS_FACES + 1.0) * f.w   # + badge half-width
            x0, x1 = max(0, int(cx - half_w)), min(w, int(cx + half_w) + 1)
            y0 = max(0, int(cy - 0.5 * f.h))
            y1 = min(h, int(cy + (LINE_MAX_DOWN_FACES + 1.0) * f.h) + 1)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            scale = min(QR_ZONE_MAX_SCALE, max(1.0, QR_ZONE_TARGET_PX / (y1 - y0)))
            zone = frame_bgr[y0:y1, x0:x1]
            if scale > 1.0:
                zone = cv2.resize(zone, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            for c in self._read(zone, (x0, y0), scale):
                # Zones of people side by side can overlap: keep one copy
                if not any(o.text == c.text and
                           math.dist(o.center, c.center) < 0.5 * f.w
                           for o in codes):
                    codes.append(c)
        return codes

    def _read_wechat(self, img):
        try:
            texts, points = self.wechat.detectAndDecode(img)
        except cv2.error:
            return []
        codes = []
        for text, pts in zip(texts, points):
            if not text:
                continue
            if self.payloads is not None and text not in self.payloads:
                continue
            pts = np.asarray(pts, dtype=float).reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            codes.append(QRCode(text, (float(cx), float(cy)), pts))
        return codes

    def _read(self, img, offset=(0, 0), scale=1.0):
        """Read QRs in img; corners mapped back by /scale + offset."""
        level = _cv_log.getLogLevel()
        _cv_log.setLogLevel(_CV_LOG_ERROR)   # hide the ECI warning
        try:
            ok, texts, points, _ = self.detector.detectAndDecodeMulti(img)
            if not (ok and points is not None and any(texts)):
                # The multi reader often misses a lone badge the single
                # reader gets
                text, pts, _ = self.detector.detectAndDecode(img)
                ok, texts, points = bool(text), [text], [pts]
        except cv2.error:
            return []
        finally:
            _cv_log.setLogLevel(level)
        if not ok or points[0] is None:
            return []
        codes = []
        for text, pts in zip(texts, points):
            if not text:
                continue   # found a QR shape but couldn't read it
            if self.payloads is not None and text not in self.payloads:
                continue
            pts = np.asarray(pts, dtype=float).reshape(4, 2) / scale + offset
            cx, cy = pts.mean(axis=0)
            codes.append(QRCode(text, (float(cx), float(cy)), pts))
        return codes


class BackgroundQRDetector:
    """Runs a QRDetector on its own thread, so a slow read (enlarged zones
    on the Pi) never holds up the camera loop. submit() hands over a frame
    if the thread is idle (and is ignored while it's busy); poll() returns
    the codes of a finished read once, else None."""

    def __init__(self, detector):
        self.detector = detector
        self._job = None
        self._result = None
        self._busy = False
        self._running = True
        self._cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True,
                         name="QRDetector").start()

    def submit(self, frame_bgr, faces=None):
        with self._cond:
            if self._busy:
                return False
            # Copy: the caller may draw on its frame while we read it
            self._job, self._busy = (frame_bgr.copy(), list(faces or [])), True
            self._cond.notify()
            return True

    def poll(self):
        with self._cond:
            result, self._result = self._result, None
            return result

    def stop(self):
        with self._cond:
            self._running = False
            self._cond.notify()

    def _loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._job or not self._running)
                if not self._running:
                    return
                (frame, faces), self._job = self._job, None
            try:
                codes = self.detector.detect(frame, faces)
            except Exception:
                codes = []
            with self._cond:
                self._result, self._busy = codes, False


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
            best, best_cost = f, cost
    return best


def gesture_allowed(hand_face, priority, require_priority=False):
    """With a priority person, only a hand assigned to their face counts.
    Without one, every hand counts (same as run_gesture_mentorpi.py today),
    unless require_priority is set: then no hand counts."""
    if not priority.active:
        return not require_priority
    return hand_face is not None and hand_face is priority.face
