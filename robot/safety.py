"""
============================================================================
safety.py  —  Safety layer for the MentorPi adapter
============================================================================
Every motion command the MentorPi adapter proposes passes through this
module before it becomes a real /cmd_vel publication. The safety layer:

  1. Clamps speeds to hard maximums
  2. Refuses forward motion if it would violate the distance floor to person
  3. Overrides intent to NEUTRAL if no person is visible
  4. Stops motion if an obstacle is detected in the direction of travel
  5. Degrades speeds and distances on low battery
  6. Respects the E-stop flag above all else

These are pure functions with no side effects (except reading the constants
below). That makes them easy to unit-test and impossible to accidentally
bypass by editing the adapter.

Values here are proposed starting points from the design doc — tune in the
lab.
============================================================================
"""

from dataclasses import dataclass


# --- Hard limits ---------------------------------------------------------
MAX_LINEAR_SPEED    = 0.25   # m/s  — slow indoor walking pace
MAX_ANGULAR_SPEED   = 0.5    # rad/s (~29 deg/s)
MIN_DISTANCE_FLOOR  = 0.8    # meters — closest the robot may ever be
MAX_TRACKING_RANGE  = 3.0    # meters — beyond this, person considered absent

# --- Low-battery degradation ---------------------------------------------
BATTERY_LOW_PCT     = 20.0   # below this, degrade motion
BATTERY_SPEED_MULT  = 0.5    # halve speed
BATTERY_DIST_MULT   = 1.25   # expand distance targets by 25%

# --- Person-presence timeout ---------------------------------------------
MAX_MISSING_WINDOWS = 2      # windows of no-face before overriding to NEUTRAL

# --- Obstacle detection --------------------------------------------------
OBSTACLE_STOP_DIST  = 0.4    # if LiDAR sees a non-person thing this close, stop


@dataclass
class Motion:
    """A proposed motion command. Positive linear = forward, negative = back."""
    linear:  float = 0.0     # m/s
    angular: float = 0.0     # rad/s
    reason:  str   = ""      # optional explanation (for logs/debugging)


@dataclass
class RobotState:
    """Snapshot of what the robot knows about itself and the person."""
    person_visible:   bool  = True
    person_distance:  float = 1.5    # meters (last observed)
    person_bearing:   float = 0.0    # radians (0 = straight ahead)
    missing_windows:  int   = 0      # consecutive windows without a face
    battery_pct:      float = 100.0
    obstacle_ahead:   bool  = False  # non-person obstacle in direction of motion
    estop_active:     bool  = False


# ---------------------------------------------------------------
# Individual safety checks — each returns a Motion (possibly modified)
# ---------------------------------------------------------------
def clamp_speeds(motion: Motion) -> Motion:
    """Clip linear and angular velocities to their hard maximums."""
    return Motion(
        linear  = max(-MAX_LINEAR_SPEED,  min(MAX_LINEAR_SPEED,  motion.linear)),
        angular = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, motion.angular)),
        reason  = motion.reason,
    )


def enforce_distance_floor(motion: Motion, state: RobotState) -> Motion:
    """
    Refuse forward motion if the robot is already too close to the person.
    Retreat (negative linear) is always allowed.
    """
    if motion.linear > 0 and state.person_distance <= MIN_DISTANCE_FLOOR:
        return Motion(
            linear  = 0.0,
            angular = motion.angular,        # rotation is still fine
            reason  = f"BLOCKED_FORWARD (person at {state.person_distance:.2f}m)",
        )
    return motion


def check_person_presence(motion: Motion, state: RobotState) -> Motion:
    """
    If no person has been seen for too long, override motion to a full stop.
    Prevents the robot from acting on stale person data.
    """
    if state.missing_windows > MAX_MISSING_WINDOWS:
        return Motion(
            linear  = 0.0,
            angular = 0.0,
            reason  = f"NO_PERSON ({state.missing_windows} windows missing)",
        )
    return motion


def check_obstacle(motion: Motion, state: RobotState) -> Motion:
    """
    If a non-person obstacle is close in the direction we're driving, stop.
    """
    if state.obstacle_ahead and motion.linear > 0:
        return Motion(
            linear  = 0.0,
            angular = motion.angular,
            reason  = "OBSTACLE_AHEAD",
        )
    return motion


def apply_battery_degradation(motion: Motion, state: RobotState) -> Motion:
    """On low battery, halve speeds. Distances handled by the adapter."""
    if state.battery_pct < BATTERY_LOW_PCT:
        return Motion(
            linear  = motion.linear  * BATTERY_SPEED_MULT,
            angular = motion.angular * BATTERY_SPEED_MULT,
            reason  = motion.reason + " [LOW_BATT]",
        )
    return motion


def enforce_estop(motion: Motion, state: RobotState) -> Motion:
    """Highest priority: E-stop overrides everything."""
    if state.estop_active:
        return Motion(linear=0.0, angular=0.0, reason="ESTOP")
    return motion


# ---------------------------------------------------------------
# The one function the adapter actually calls
# ---------------------------------------------------------------
def apply_all(motion: Motion, state: RobotState) -> Motion:
    """
    Run every safety check in the right order. This is the single entry
    point the adapter uses — never bypass this.

    Order matters:
      1. Speed clamp first  — catches wild proposed values
      2. Distance floor     — hard personal-space limit
      3. Person presence    — no acting on missing data
      4. Obstacle           — physical safety
      5. Battery            — graceful degradation
      6. E-stop last        — overrides everything else
    """
    motion = clamp_speeds(motion)
    motion = enforce_distance_floor(motion, state)
    motion = check_person_presence(motion, state)
    motion = check_obstacle(motion, state)
    motion = apply_battery_degradation(motion, state)
    motion = enforce_estop(motion, state)
    return motion
