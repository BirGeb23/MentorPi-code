"""
test_mentorpi_adapter_mock.py
=============================
Runs the MentorPi adapter through every intent + VA_POSE mode in mock
mode — no ROS 2, no robot, no network. The adapter prints what it WOULD
publish to /cmd_vel for each ReactionAction.

How to run:
    cd ~/MentorPi-code/tools
    python3 test_mentorpi_adapter_mock.py

What this proves:
  - The 5 discrete intents produce sensible motion commands
  - Distance-holding logic works (approach stops at target, retreat stops
    when far enough away)
  - Person-facing rotation only fires when out of alignment
  - VA_POSE continuous mapping produces expected forward/backward motion
  - The safety layer intercepts unsafe cases (too close, no person, low
    battery, obstacle, E-stop)
"""

import os
import sys
from dataclasses import dataclass

# Shared MentorPi modules live in robot/ (see README.md).
_MENTORPI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_MENTORPI_ROOT, "robot"))
from mentorpi_adapter import MentorPiAdapter


# ---------------------------------------------------------------------------
# Minimal stand-in for the real ReactionAction from va-driven-behaviors.
# Only the fields the adapter reads.
# ---------------------------------------------------------------------------
@dataclass
class ReactionAction:
    intent:    str
    pose_mode: str          # "VA_POSE" or "REACTION_POSE"
    valence:   float = 0.0
    arousal:   float = 0.0


def section(title: str):
    """Pretty section separator for terminal output."""
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66)


def main():
    adapter = MentorPiAdapter(mock=True)

    # -------------------------------------------------------------------
    # PART 1 — All 5 discrete intents, with a person clearly visible
    #          at 1.5m directly ahead. This is the "baseline" case.
    # -------------------------------------------------------------------
    section("PART 1 — Discrete intents (person visible, 1.5m ahead)")
    adapter.state.person_visible  = True
    adapter.state.person_distance = 1.5
    adapter.state.person_bearing  = 0.0     # dead ahead

    for intent in ["NEUTRAL", "CHECK_IN", "ENGAGE", "CAUTION", "DE_ESCALATE"]:
        adapter.apply_reaction(ReactionAction(intent=intent,
                                              pose_mode="REACTION_POSE"))

    # -------------------------------------------------------------------
    # PART 2 — Distance-holding logic: person already at ENGAGE distance
    #          Robot should NOT keep approaching once at target.
    # -------------------------------------------------------------------
    section("PART 2 — ENGAGE when already at 1.0m target (should NOT approach)")
    adapter.state.person_distance = 1.0     # already at target
    adapter.apply_reaction(ReactionAction(intent="ENGAGE",
                                          pose_mode="REACTION_POSE"))

    section("PART 3 — DE_ESCALATE when already at 2.0m target (should NOT back up)")
    adapter.state.person_distance = 2.0
    adapter.apply_reaction(ReactionAction(intent="DE_ESCALATE",
                                          pose_mode="REACTION_POSE"))

    # -------------------------------------------------------------------
    # PART 4 — Rotation-toward-person: person is off to the left
    # -------------------------------------------------------------------
    section("PART 4 — CHECK_IN with person 30 degrees to the left")
    adapter.state.person_distance = 1.8
    adapter.state.person_bearing  = 0.52    # ~30 degrees, positive = left
    adapter.apply_reaction(ReactionAction(intent="CHECK_IN",
                                          pose_mode="REACTION_POSE"))

    # -------------------------------------------------------------------
    # PART 5 — VA_POSE continuous mode: 4 different emotion mixes
    # -------------------------------------------------------------------
    section("PART 5 — VA_POSE (continuous mapping from valence/arousal)")
    adapter.state.person_distance = 1.5
    adapter.state.person_bearing  = 0.0
    samples = [
        ("Happy + excited",   +0.6, +0.7),   # → approach at higher speed
        ("Upset + agitated",  -0.5, +0.6),   # → retreat at higher speed
        ("Mild positive",     +0.3, -0.2),   # → gentle approach
        ("Sad + low energy",  -0.4, -0.3),   # → slow retreat
    ]
    for label, v, a in samples:
        print(f"\n  [{label}]  valence={v:+.2f}  arousal={a:+.2f}")
        adapter.apply_reaction(ReactionAction(intent="VA_POSE",
                                              pose_mode="VA_POSE",
                                              valence=v, arousal=a))

    # -------------------------------------------------------------------
    # PART 6 — Safety layer: ENGAGE while too close to person
    #          Forward motion should be BLOCKED even though intent says
    #          to approach.
    # -------------------------------------------------------------------
    section("PART 6 — SAFETY: ENGAGE when person is at 0.5m (below floor)")
    adapter.state.person_distance = 0.5     # inside safety floor of 0.8m
    adapter.apply_reaction(ReactionAction(intent="ENGAGE",
                                          pose_mode="REACTION_POSE"))

    # -------------------------------------------------------------------
    # PART 7 — Safety layer: no person detected for too long
    # -------------------------------------------------------------------
    section("PART 7 — SAFETY: person missing for 3 windows -> override to STOP")
    adapter.state.person_distance = 1.5
    adapter.state.person_visible  = False
    adapter.state.missing_windows = 3
    adapter.apply_reaction(ReactionAction(intent="ENGAGE",
                                          pose_mode="REACTION_POSE"))

    # -------------------------------------------------------------------
    # PART 8 — Safety layer: low battery
    # -------------------------------------------------------------------
    section("PART 8 — SAFETY: low battery (10%) -> speeds should be halved")
    adapter.state.person_visible  = True
    adapter.state.missing_windows = 0
    adapter.state.battery_pct     = 10.0
    adapter.apply_reaction(ReactionAction(intent="ENGAGE",
                                          pose_mode="REACTION_POSE"))
    adapter.state.battery_pct     = 100.0   # reset for later tests

    # -------------------------------------------------------------------
    # PART 9 — Safety layer: obstacle ahead
    # -------------------------------------------------------------------
    section("PART 9 — SAFETY: obstacle ahead -> forward motion blocked")
    adapter.state.obstacle_ahead = True
    adapter.apply_reaction(ReactionAction(intent="ENGAGE",
                                          pose_mode="REACTION_POSE"))
    adapter.state.obstacle_ahead = False

    # -------------------------------------------------------------------
    # PART 10 — E-stop overrides everything
    # -------------------------------------------------------------------
    section("PART 10 — SAFETY: E-stop overrides ENGAGE -> full stop")
    adapter.state.estop_active = True
    adapter.apply_reaction(ReactionAction(intent="ENGAGE",
                                          pose_mode="REACTION_POSE"))
    adapter.state.estop_active = False

    # -------------------------------------------------------------------
    # Clean shutdown
    # -------------------------------------------------------------------
    adapter.shutdown()
    print("\nDone. Every line above is exactly what the MentorPi WOULD do.\n")


if __name__ == "__main__":
    main()
