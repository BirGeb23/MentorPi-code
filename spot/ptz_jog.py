#!/usr/bin/env python3
"""
============================================================================
ptz_jog.py  —  Move the Spot CAM PTZ by hand from the keyboard
============================================================================
Type a key and press Enter to nudge the PTZ camera:

    a / d     pan left / right by 5 degrees     (A / D = 20 degrees)
    w / s     tilt up / down by 5 degrees
    + / -     zoom in / out
    p         print the current position
    q         quit

Each move prints the new pan / tilt / zoom, so you can write down the
values that frame you well and reuse them with:
    python3 -m command_line $ROBOT_IP ptz set_position mech PAN TILT ZOOM

If "a" turns the camera the wrong way for you, just use "d" instead; the
direction pan numbers turn depends on how the camera is mounted.

Usage (Spot SDK venv active, laptop on Spot's Wi-Fi):
    python3 ptz_jog.py $ROBOT_IP
It asks for the Spot username and password, like the command_line tool.
============================================================================
"""

import argparse

import bosdyn.client
import bosdyn.client.util
from bosdyn.api.spot_cam import ptz_pb2
from bosdyn.client import spot_cam
from bosdyn.client.spot_cam.ptz import PtzClient

PAN_STEP, PAN_BIG_STEP, TILT_STEP = 5.0, 20.0, 5.0
TILT_MIN, TILT_MAX = -30.0, 90.0     # limits reported by "ptz list" for mech
ZOOM_MIN, ZOOM_MAX = 1.0, 30.0
ZOOM_FACTOR = 1.25


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("--ptz", default="mech", help="PTZ name (default: mech)")
    options = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk("PtzJog")
    spot_cam.register_all_service_clients(sdk)
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    ptz = robot.ensure_client(PtzClient.default_service_name)
    desc = ptz_pb2.PtzDescription(name=options.ptz)

    pos = ptz.get_ptz_position(desc)
    pan, tilt, zoom = pos.pan.value, pos.tilt.value, max(pos.zoom.value, ZOOM_MIN)

    def show():
        print(f"  pan {pan:6.1f}   tilt {tilt:5.1f}   zoom {zoom:4.1f}")

    print(__doc__.split("Each move")[0].split("====\n")[-1])
    show()

    while True:
        try:
            key = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if key == "q":
            break
        if key == "p":
            show()
            continue
        if key == "a":
            pan -= PAN_STEP
        elif key == "d":
            pan += PAN_STEP
        elif key == "A":
            pan -= PAN_BIG_STEP
        elif key == "D":
            pan += PAN_BIG_STEP
        elif key == "w":
            tilt += TILT_STEP
        elif key == "s":
            tilt -= TILT_STEP
        elif key == "+":
            zoom *= ZOOM_FACTOR
        elif key == "-":
            zoom /= ZOOM_FACTOR
        else:
            print("  keys: a d A D w s + - p q")
            continue
        pan %= 360.0
        tilt = min(max(tilt, TILT_MIN), TILT_MAX)
        zoom = min(max(zoom, ZOOM_MIN), ZOOM_MAX)
        ptz.set_ptz_position(desc, pan, tilt, zoom)
        show()

    print("Final position (reuse with ptz set_position):")
    show()


if __name__ == "__main__":
    main()
