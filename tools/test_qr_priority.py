"""
test_qr_priority.py
===================
Offline checks for robot/qr_priority.py: the imaginary-line match, the
priority state over time, and hand-to-face assignment. No camera, no
robot, no MediaPipe: faces, QRs and hands are made-up pixel positions.

How to run:
    cd ~/MentorPi-code/tools
    python3 test_qr_priority.py

Prints PASS/FAIL per check and exits non-zero if anything failed.
Change the thresholds at the top of qr_priority.py and re-run to see
which situations they accept.
"""

import os
import sys

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from qr_priority import (Face, PriorityState, QRCode, assign_hand_to_face,
                         gesture_allowed, link_qrs, match_qr_to_face)

results = []


def check(name, cond):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def face_at(cx, top, size=60):
    """Square face box centred on cx. ~60 px is a face at ~1.5 m in 640x480."""
    return Face(cx - size / 2, top, size, size)


def qr_at(x, y, text="MENTORPI-PRIORITY"):
    return QRCode(text, (x, y))


# Face A: centre x 200, top 100, 60x60 -> chin at (200, 160)
# With the default limits the QR must sit 0..240 px below the chin and
# within 60 px left/right of it.

print("\n--- Imaginary line: QR -> face ---")
A = face_at(200, 100)
check("QR straight below the face is matched",
      match_qr_to_face(qr_at(200, 260), [A])[0] is A)
check("QR above the face is not matched",
      match_qr_to_face(qr_at(200, 80), [A])[0] is None)
check("QR too far to the side is not matched",
      match_qr_to_face(qr_at(300, 260), [A])[0] is None)
check("QR too far below is not matched",
      match_qr_to_face(qr_at(200, 410), [A])[0] is None)

A, B = face_at(200, 100), face_at(300, 100)
check("QR between two people goes to the one it's more directly under",
      match_qr_to_face(qr_at(240, 260), [A, B])[0] is A)

A = face_at(200, 100)
links = link_qrs([qr_at(200, 260), qr_at(230, 300)], [A])
check("Two QRs under one face: only one keeps the link",
      sum(l.face is A for l in links) == 1)

print("\n--- Priority over time ---")
p = PriorityState(hold_s=4.0, face_lost_s=1.0)
A = face_at(200, 100)
_, ev = p.update(0.0, [A], [qr_at(200, 260)])
check("Badge under a face grants priority", "granted" in ev and p.face is A)

A2 = face_at(220, 100)
p.update(1.0, [A2], [])
check("Priority follows the face as it moves", p.face is A2)

A3 = face_at(225, 100)
p.update(3.9, [A3], None)
check("Priority kept while the QR is briefly unseen (< hold_s)", p.active)

A4 = face_at(225, 100)
_, ev = p.update(4.5, [A4], [])
check("Priority released after hold_s without the QR",
      "released" in ev and not p.active)

p = PriorityState(hold_s=4.0, face_lost_s=1.0)
A = face_at(200, 100)
p.update(0.0, [A], [qr_at(200, 260)])
p.update(0.5, [], None)
check("Priority kept through a short face dropout", p.active)
p.update(1.6, [], None)
check("Priority released when the face is gone > face_lost_s", not p.active)

p = PriorityState(hold_s=4.0, face_lost_s=1.0)
A = face_at(200, 100)
p.update(0.0, [A], [qr_at(200, 260)])
far = face_at(400, 100)
p.update(0.1, [far], None)
check("A face far from the last position is not treated as the same person",
      p.face is A)

p = PriorityState(hold_s=4.0, face_lost_s=1.0)
A, B = face_at(200, 100), face_at(400, 100)
p.update(0.0, [A], [qr_at(200, 260)])
A2, B2 = face_at(200, 100), face_at(400, 100)
p.update(1.0, [A2, B2], [qr_at(200, 260), qr_at(400, 260)])
check("A second badge wearer can't take priority from the holder",
      p.face is A2)
B3 = face_at(400, 100)
_, ev = p.update(2.5, [B3], [qr_at(400, 260)])
check("Once the holder leaves, the other badge wearer gets it",
      ev == ["released", "granted"] and p.face is B3)

print("\n--- Hands ---")
A, B = face_at(200, 100), face_at(320, 100)
check("Hand at chest height below A belongs to A",
      assign_hand_to_face((210, 250), [A, B]) is A)
check("Hand raised beside B belongs to B",
      assign_hand_to_face((300, 120), [A, B]) is B)
check("Hand far from everyone belongs to nobody",
      assign_hand_to_face((620, 130), [A]) is None)

p = PriorityState()
check("No priority: any hand's gesture counts", gesture_allowed(None, p))
check("No priority + require_priority: no hand counts",
      not gesture_allowed(A, p, require_priority=True))
p.update(0.0, [A, B], [qr_at(200, 260)])
check("Priority holder's hand counts", gesture_allowed(A, p))
check("Someone else's hand is ignored", not gesture_allowed(B, p))
check("A hand with no face is ignored while someone has priority",
      not gesture_allowed(None, p))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
