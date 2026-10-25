# MentorPi Motion Mapping — Design Proposal
## Valence-Arousal Driven Behavior Reactions

**Author:** Biruk
**Status:** Original design proposal, kept for reference. The implemented motions differ in places (for example, the GREET, ACK nod, PULSE and SCAN movements replaced the wiggles); see `README.md` and `robot/mentorpi_adapter.py` for current behavior.
**Related:** `va-driven-behaviors` pipeline, `spot_adapter_EXAMPLE.py`

---

## Purpose

This document proposes how the five behavioral intents produced by the `va-driven-behaviors` pipeline should be expressed as physical motion on the MentorPi mobile robot. It is a design proposal for team review before any adapter code is written.

The existing SpotAdapter translates each intent into a body posture — height changes, pitch, roll — because Spot is a quadruped with expressive body language. The MentorPi is a differential-drive wheeled robot; its motion vocabulary is different: **linear velocity, angular velocity, and distance maintained from the person**. This document works out how each intent should translate to that different vocabulary while preserving the meaning of the original behavior.

---

## Design principles

These apply across every intent below and shape the specific choices.

**Legibility over expressiveness.** A wheeled robot cannot mimic a quadruped's body language. Instead, motions should be simple and immediately readable to a human observer — clearly retreating, clearly approaching, clearly stopped. If a human watching the robot cannot instantly tell what it's doing, the motion is wrong.

**Safety first, always.** No intent can override the safety floor. Every motion command is clamped to safe speed limits and refuses to violate the minimum distance to the person, regardless of what the pipeline requests.

**Smooth over sudden.** Velocity changes ramp over ~0.5 seconds. Sudden accelerations look agitated and can startle a nearby person.

**Default to still.** When in doubt (missing person, low confidence, safety violation), the robot holds position. Motion is deliberate; stillness is the fallback.

---

## Global parameters

Hard limits that apply to every intent. All values are proposed starting points and should be tuned in the lab.

| Parameter | Proposed value | Rationale |
|---|---|---|
| Max linear speed | 0.25 m/s | Slow walking pace; safe indoors near a person |
| Max angular speed | 0.5 rad/s (~29°/s) | Comfortable rotation rate; not jerky |
| Min distance to person | 0.8 m | Personal-space floor; safe stopping distance at max speed |
| Max distance to track | 3.0 m | Beyond this, robot considers person "not present" |
| Motion ramp time | 0.5 s | Smooth acceleration and deceleration |
| Face-orient tolerance | ±10° | Rotate to align if outside this cone |
| Person-presence timeout | 2 windows (~6 s) | If no face detected this long, drop to NEUTRAL |

---

## Per-intent motion mapping

### 1. NEUTRAL

**Meaning:** Baseline. Person is calm, or the confidence gate rejected any other proposal.

**Motion:**
- Linear velocity: 0 m/s
- Angular velocity: 0 rad/s
- Optional idle: small periodic angular wiggle (±0.05 rad/s for 0.2 s, every 15 s) to signal the robot is "alive"

**Target distance:** hold current distance

**Rationale:** NEUTRAL is the default state when nothing needs doing. The optional idle wiggle prevents the robot from looking powered-off, which can be unnerving to a person nearby. This can be disabled entirely if the team prefers a truly still baseline.

---

### 2. CHECK_IN

**Meaning:** Person shows subtle negative or low-arousal state. The robot attends to them.

**Motion:**
1. Rotate to face person: angular velocity 0.3 rad/s toward person; stop rotating once bearing is within ±10°
2. After facing: if further than 1.5 m, slow gentle approach at +0.08 m/s
3. Hold position at 1.5 m
4. Remain facing the person during the hold

**Target distance:** 1.5 m (comfortable but attentive)

**Rationale:** CHECK_IN communicates "I notice you and I'm here." Facing the person signals attention. Approaching gently at low speed signals care without pressure. 1.5 m is close enough to feel attentive without invading personal space.

---

### 3. ENGAGE

**Meaning:** Person shows positive, high-arousal emotion. The robot approaches warmly.

**Motion:**
1. Rotate to face person if not aligned: angular velocity 0.4 rad/s
2. Approach at moderate speed: linear velocity +0.15 m/s until 1.0 m from person
3. Optional celebratory wiggle on arrival: ±0.15 rad/s for 0.4 s each way, repeat twice
4. Hold at 1.0 m, facing the person

**Target distance:** 1.0 m (social/friendly distance)

**Rationale:** ENGAGE is the most active positive behavior. The moderate approach speed conveys enthusiasm without aggression. The arrival wiggle is a small personality touch — a nod to Spot's expressive body without pretending to be a quadruped. Can be dropped if the team prefers a minimalist arrival.

---

### 4. CAUTION

**Meaning:** Person shows moderate distress or high emotional volatility. The robot pauses to be careful.

**Motion:**
- Full stop: linear velocity 0, angular velocity 0
- Ramp down any current motion within 0.5 s
- Hold position for the entire duration of the CAUTION state

**Target distance:** hold current distance (no adjustment)

**Rationale:** CAUTION is a pause, not a retreat. The robot doesn't approach, doesn't retreat, doesn't turn — it just stops and waits, letting the person control the next move. This embodies the "when uncertain, do nothing" principle. Deliberate stillness reads as attentive without being intrusive.

---

### 5. DE_ESCALATE

**Meaning:** Person shows strong negative emotion or the pipeline detects high volatility. The robot backs off to give space.

**Motion:**
1. Back away slowly: linear velocity −0.10 m/s
2. No rotation during retreat — a straight backward motion looks non-threatening and deliberate
3. Continue backing until 2.0 m from person, then stop
4. If already further than 2.0 m, hold position (no motion needed)

**Target distance:** 2.0 m (respectful space)

**Rationale:** DE_ESCALATE is the most conservative response and the most safety-critical. Backing away communicates "I recognize you need space; I'm giving it." Slow speed prevents startling the person. No rotation during the retreat — turning while backing could look evasive or confused; a clean straight backup is unambiguously deferential.

---

## Continuous VA_POSE mode

When the pipeline's stability gates approve and confidence is high, it may output `pose_mode = VA_POSE`, meaning the raw valence and arousal should drive continuous motion instead of snapping to one of the five discrete intents. For the MentorPi, the proposed mapping is:

**Valence → approach/retreat direction and magnitude**
- Positive valence → forward motion (approach)
- Negative valence → backward motion (retreat)
- Linear velocity formula: `v = 0.15 × valence`, clamped to safety limits
- Below the safety distance floor (0.8 m), forward motion is refused regardless of valence

**Arousal → motion intensity multiplier**
- Higher arousal → faster motion (up to the safety cap)
- Speed multiplier: `mult = 0.5 + 0.5 × |arousal|`
- Final velocity: `v_final = v × mult`, still clamped

**Orientation** remains toward the person throughout VA_POSE mode.

**Rationale:** VA_POSE gives the robot a continuous, expressive response range rather than snapping between the five discrete states. It's the wheeled analog of Spot's continuous body-pose modulation. All safety limits still apply.

---

## Safety layer

The safety layer sits between the adapter's intent dispatch and the actual wheel commands. It intercepts every proposed motion and applies the following checks, in order:

1. **Speed clamping** — clip linear and angular velocities to the max values
2. **Distance floor** — if the proposed motion would bring the robot within 0.8 m of the person, refuse forward motion (retreat is still allowed)
3. **Person presence check** — if no person has been detected for more than 2 consecutive windows, override the intent to NEUTRAL
4. **Obstacle avoidance** — if the LiDAR detects an obstacle (not the person) in the direction of motion, stop
5. **Battery check** — below 20% battery, reduce all speeds to 50% and expand target distances by 25%
6. **E-stop** — hardware or software E-stop overrides everything and stops the wheels immediately

**Rationale:** Any of these can trigger regardless of what intent is active. The safety layer is the last line of defense between the pipeline's decision and the robot's physical motion. It is deliberately independent of the intent logic so that safety cannot be accidentally bypassed by a change to the mapping.

---

## Person tracking

The safety layer and the intent motions all depend on knowing where the person is. Proposed method:

- **Angle to person:** computed from the face's horizontal position in the camera frame (via MediaPipe face detection). Center of frame = 0°; left edge ≈ −30° (half of camera FOV); right edge ≈ +30°.
- **Distance to person:** LiDAR reading in the direction of the detected face, averaged over a small angular cone (±5°) to reduce noise.
- **Fallback:** if the face detector fails but LiDAR shows an object at typical human distance (0.5–2 m) directly in front, assume it's the person (temporary safety approximation).

---

## Open questions for the team

Points where I'd value input before implementation:

1. **Speed and distance calibration** — the numbers above are starting points. What's the team's preferred process for tuning these against real observations?
2. **Idle motion** — is the periodic NEUTRAL wiggle desirable, or should NEUTRAL be truly still?
3. **ENGAGE arrival wiggle** — personality choice, not safety-driven. Keep or drop?
4. **Person tracking** — LiDAR-only, camera-only, or fused as proposed above?
5. **VA_POSE frequency** — is the continuous mode expected to fire often in practice, or is REACTION_POSE the dominant mode?
6. **MentorPi hardware limits** — I've proposed max linear 0.25 m/s and max angular 0.5 rad/s. Are the actual safe operating limits different?

---

## Implementation plan (once approved)

1. Build the safety layer as a separate module (`safety.py`) with clear unit tests — this is the most important piece to get right
2. Implement the discrete intent table (`_pose_for_intent()` equivalent)
3. Implement the VA_POSE continuous mapping
4. Build the `MentorPiAdapter` class following the same interface as `SpotAdapter` (with `mock=True` mode)
5. Write a mock test (`test_mentorpi_adapter_mock.py`) following the pattern of `test_spot_adapter_mock.py`
6. Deploy on the Pi, test in mock mode with the live pipeline
7. Enable real motion in an open area, at capped low speeds (50% of designed values initially)
8. Observe with a person present, iterate on the parameters
9. Gradually raise speeds toward the designed values as behavior is validated

---

## Summary

The five intents translate to the MentorPi's wheeled vocabulary as follows:

| Intent | One-line summary | Target distance |
|---|---|---|
| NEUTRAL | Hold position with optional idle wiggle | current |
| CHECK_IN | Face person, gently approach to attentive distance | 1.5 m |
| ENGAGE | Face person, approach to social distance, optional arrival wiggle | 1.0 m |
| CAUTION | Full stop, hold, no motion | current |
| DE_ESCALATE | Back straight away to respectful distance | 2.0 m |

Continuous VA_POSE mode maps valence to direction/magnitude of motion and arousal to speed multiplier, subject to the safety layer.

The safety layer wraps all motion dispatch, enforcing speed caps, distance floors, person-presence timeouts, obstacle avoidance, battery-aware degradation, and E-stop.
