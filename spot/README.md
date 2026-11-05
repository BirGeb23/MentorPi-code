# Spot: gesture-driven intents

The same five intents as the MentorPi (`NEUTRAL`, `CHECK_IN`, `ENGAGE`, `CAUTION`, `DE_ESCALATE`), on a Boston Dynamics Spot with the Spot CAM+IR. Everything runs on a laptop connected to Spot's Wi-Fi: the laptop reads the Spot CAM PTZ video, runs MediaPipe, and sends movement commands to Spot through the Spot SDK. Spot itself only moves.

| Gesture | Intent | Spot's movement |
|---|---|---|
| Thumb up | `ENGAGE` | Leans forward, turns to face the person, walks closer (stops at about 1.5 m) |
| Pointing up | `CHECK_IN` | One "I see you" nod per streak, slight lean, turns to face |
| Open palm | `CAUTION` | Holds still; one slow look left and right if held |
| Thumb down | `DE_ESCALATE` | Leans back, backs away about 0.45 m |
| Anything else | `NEUTRAL` | Neutral stance; small bob after 30 s idle |

## Files

Two parts, kept separate: **gesture side** (camera → person → gesture → intent) and **movement side** (intent → Spot commands).

| File | Part | What it does |
|---|---|---|
| `spot_gestures.py` | Gesture | Shared gesture pipeline: Spot CAM stream, face detection and tracking, gesture crop that follows the person, hold check, gesture → intent, distance/direction calibration. |
| `spot_tag_control.py` | Gesture | AprilTag control: only the person wearing the tag (AprilTag 36h11) controls Spot. |
| `spot_skeleton.py` | Gesture | Optional full-body skeleton check (`--skeleton`): a hand only counts if it is at the end of the tag wearer's own arm. |
| `spot_adapter.py` | Movement | Intent → Spot body posture, turning and walking, with speed and distance limits. Mock mode prints commands without moving. |
| `spot_gesture_basic.py` | Program | Show a gesture, Spot moves. Follows the person nearest the centre of the view. |
| `spot_gesture_tag.py` | Program | AprilTag version: only the tag wearer's gestures count. Add `--skeleton` for the skeleton check. |
| `spot_intent_test.py` | Tool | Keyboard tester: keys 1-5 send each intent to Spot (no camera). |
| `ptz_jog.py` | Tool | Move the Spot CAM PTZ camera with the keyboard. |

## Setup (laptop)

1. Install the Spot SDK and its `examples/spot_cam` requirements (includes `aiortc` for the video stream) in a virtual environment, plus `mediapipe` and `opencv-contrib-python`.
2. Copy this folder's `.py` files into the SDK's `python/examples/spot_cam/` folder. The programs use that example's `webrtc.py` for the video stream.
3. Download the models into the same folder (not in Git):
   ```
   curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
   curl -L -o blaze_face_full_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite
   curl -L -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
   ```
   If the face model download fails, copy `blaze_face_full_range.tflite` from an existing setup.
4. Connect to Spot's Wi-Fi. Spot is at `192.168.80.3`. The programs ask for the Spot username and password.
5. Calibrate once (stand at the distances it asks for). Without `spot_calibration.json`, Spot won't walk or turn:
   ```
   python3 spot_gesture_tag.py 192.168.80.3 --calibrate
   ```

## Running

Always: E-stop operator ready (tablet or the SDK's estop example), area clear. Start with mock mode.

```
# AprilTag version (tag ID 2), intents printed only
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2
# Mock: prints Spot commands, no motion
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --drive mock
# Real motion without walking, then with walking
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --drive real --no-walk --speed-scale 1.0
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --drive real --speed-scale 1.0
# Add the skeleton check to any of the above
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --skeleton --drive mock

# Basic version (no tag)
python3 spot_gesture_basic.py 192.168.80.3 --show --drive mock
```

Use `--speed-scale 1.0`: Spot ignores the very slow velocities produced at half speed. In the preview, press `q` to quit and `r` to reset who is in control.

Status: the tag version has been tested on Spot in mock and real motion (with and without walking). The skeleton check is tested in simulation only so far.
