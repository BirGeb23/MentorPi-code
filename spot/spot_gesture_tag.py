#!/usr/bin/env python3
"""
spot_gesture_tag.py  —  AprilTag (fiducial) version: only the tag wearer controls Spot.

The controller wears an AprilTag 36h11 fiducial at chest height, facing Spot.
Spot only reacts to gestures from the person directly above a visible tag;
everyone else is ignored. Hide the tag for >1.5 s = Spot stops listening.
Hand the tag to someone else = they take over.

Same gestures and movements as spot_gesture_basic.py:
    Thumb up -> ENGAGE      Pointing up -> CHECK_IN      Open palm -> CAUTION
    Thumb down -> DE_ESCALATE      anything else -> NEUTRAL

Gesture side: spot_gestures.py   Tag logic: spot_tag_control.py   Movement: spot_adapter.py

Run from ~/Spot_SDK/spot-sdk/python/examples/spot_cam (venv active):
    python3 spot_gesture_tag.py 192.168.80.3 --show                         # find the tag ID
    python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 5             # intents only
    python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 5 --drive mock
    python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 5 --drive real --no-walk --speed-scale 1.0
    python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 5 --drive real --speed-scale 1.0

Skeleton option (spot_skeleton.py): add --skeleton to any command above. A hand
then only counts if it is at the end of the tag wearer's own arm, so a
bystander's hand next to the controller's face or tag is ignored. Needs
pose_landmarker_lite.task (or _full) in this folder.
    python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --skeleton
"""
import argparse
import os
import sys

from spot_gestures import add_common_args, run, HERE
from spot_tag_control import TagControl


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--tag-ids", type=int, nargs="+",
                   help="Only these AprilTag IDs give control (default: any 36h11 tag)")
    p.add_argument("--skeleton", action="store_true",
                   help="Only accept hands at the end of the tag wearer's own arm "
                        "(full-body skeleton check)")
    p.add_argument("--pose-model", default=os.path.join(HERE, "pose_landmarker_lite.task"),
                   help="Pose Landmarker model (lite recommended; full is slower)")
    p.add_argument("--skeleton-every", type=int, default=1,
                   help="Run the pose model every N frames (2 = half the cost)")
    p.add_argument("--num-hands", type=int, default=None,
                   help="Hands the gesture model looks for (default 2, or 4 with --skeleton)")
    args = p.parse_args()
    if args.num_hands is None:
        args.num_hands = 4 if args.skeleton else 2

    skeleton = None
    if args.skeleton and not args.calibrate:
        if not os.path.exists(args.pose_model) or os.path.getsize(args.pose_model) == 0:
            sys.exit(f"Missing pose model: {args.pose_model}\nDownload it with:\n  curl -L -o "
                     "pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/"
                     "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        from spot_skeleton import SkeletonCheck
        skeleton = SkeletonCheck(args.pose_model, num_poses=3, every=args.skeleton_every)
        print(f"[SKEL] Skeleton check ON ({os.path.basename(args.pose_model)}, "
              f"every {args.skeleton_every} frame(s), up to {args.num_hands} hands)")
    run(args, TagControl(args.tag_ids), skeleton)


if __name__ == "__main__":
    main()
