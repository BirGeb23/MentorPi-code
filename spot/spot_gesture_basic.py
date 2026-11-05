#!/usr/bin/env python3
"""
spot_gesture_basic.py  —  Show a gesture, Spot moves. (No tag, no claim.)

Spot follows the person nearest the centre of the PTZ view and reacts to
their gestures:
    Thumb up -> ENGAGE      Pointing up -> CHECK_IN      Open palm -> CAUTION
    Thumb down -> DE_ESCALATE      anything else -> NEUTRAL

Gesture side: spot_gestures.py      Movement side: spot_adapter.py

Run from ~/Spot_SDK/spot-sdk/python/examples/spot_cam (venv active):
    python3 spot_gesture_basic.py 192.168.80.3 --calibrate                 # once
    python3 spot_gesture_basic.py 192.168.80.3 --show                      # intents only
    python3 spot_gesture_basic.py 192.168.80.3 --show --drive mock
    python3 spot_gesture_basic.py 192.168.80.3 --show --drive real --no-walk --speed-scale 1.0
    python3 spot_gesture_basic.py 192.168.80.3 --show --drive real --speed-scale 1.0
"""
import argparse

from spot_gestures import add_common_args, run, CentreLock


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter))
    run(p.parse_args(), CentreLock())


if __name__ == "__main__":
    main()
