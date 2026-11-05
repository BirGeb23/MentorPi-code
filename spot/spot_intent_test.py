#!/usr/bin/env python3
"""
============================================================================
spot_intent_test.py  —  Trigger each Spot intent by hand (no gestures)
============================================================================
For stages 1-4 of the test plan: test each movement on its own, so a
problem can't be confused with a gesture-recognition problem.

Type a key and press Enter:
    1  NEUTRAL       2  CHECK_IN       3  ENGAGE
    4  CAUTION       5  DE_ESCALATE
    n  nod           s  scan           b  bob
    x  stop (hold current posture)     q  quit (sit, power off, release lease)

There is no camera here, so the person is assumed to be straight ahead at
--distance metres (measure it with a tape). ENGAGE then walks the planned
gap (start distance minus target) by dead reckoning, and DE_ESCALATE backs
away 0.45 m. Turning to face is skipped (bearing is always 0).

Usage (Spot SDK venv, from examples/spot_cam, laptop on Spot's Wi-Fi):
    python3 spot_intent_test.py --mock                       # stage 1, no robot
    python3 spot_intent_test.py $ROBOT_IP --no-walk          # stage 3: postures only
    python3 spot_intent_test.py $ROBOT_IP --speed-scale 0.5 --distance 3.0   # stage 4
The E-stop operator must have the E-stop running and released before start.
============================================================================
"""

import argparse
import sys

from spot_adapter import SpotAdapter, MotionDispatcher, ReactionAction

KEYS = {"1": "NEUTRAL", "2": "CHECK_IN", "3": "ENGAGE",
        "4": "CAUTION", "5": "DE_ESCALATE"}
HOLD_S = 10.0   # long enough for any walk to finish; "x" or a new key stops it


def main():
    p = argparse.ArgumentParser(description="Manual Spot intent tester")
    p.add_argument("hostname", nargs="?", help="Robot IP (not needed with --mock)")
    p.add_argument("--mock", action="store_true", help="Print commands only, no robot")
    p.add_argument("--no-walk", action="store_true", help="Disable walking")
    p.add_argument("--speed-scale", type=float, default=0.5,
                   help="Fraction of the speed limits (default 0.5 = half speed)")
    p.add_argument("--distance", type=float, default=3.0,
                   help="Measured distance to the person in metres (default 3.0)")
    args = p.parse_args()
    if not args.mock and not args.hostname:
        sys.exit("Give the robot IP, or use --mock")

    adapter = SpotAdapter(hostname=args.hostname, mock=args.mock,
                          allow_walking=not args.no_walk, allow_turning=False,
                          speed_scale=args.speed_scale)
    adapter.state.person_visible = True
    adapter.state.person_distance = args.distance
    adapter.state.person_bearing = 0.0

    print(__doc__.split("There is no camera")[0].split("====\n")[-1])
    print(f"Walking {'OFF' if args.no_walk else 'ON'}, speed scale {args.speed_scale}, "
          f"person assumed at {args.distance:.2f} m.\n")

    motion = MotionDispatcher()
    try:
        adapter.start()
        while True:
            try:
                key = input("> ").strip()
            except EOFError:
                break
            if key == "q":
                break
            if key in KEYS:
                motion.stop_all()
                adapter.new_streak()     # pressing a key again repeats the movement
                motion.dispatch(adapter.hold_for, ReactionAction(KEYS[key]), HOLD_S)
            elif key == "n":
                motion.dispatch(adapter.checkin_ack)
            elif key == "s":
                motion.dispatch(adapter.caution_scan)
            elif key == "b":
                motion.dispatch(adapter.neutral_pulse)
            elif key == "x":
                motion.stop_all()
                adapter.stop()
            elif key.startswith("d "):          # change the assumed distance
                try:
                    adapter.state.person_distance = float(key.split()[1])
                    print(f"  person distance set to {adapter.state.person_distance:.2f} m")
                except ValueError:
                    print("  usage: d 2.5")
            else:
                print("  keys: 1-5, n, s, b, x, q   (d 2.5 = set distance)")
    except KeyboardInterrupt:
        print("\n  Ctrl+C")
    finally:
        motion.stop_all()
        adapter.shutdown()


if __name__ == "__main__":
    main()
