# Modular MentorPi camera, QR, gesture, and motion pipeline

This new pipeline is separated into four modules and an entry point:

- `camera_feed.py` opens the MentorPi HTTP camera stream or a local webcam.
- `qr_tracker.py` detects the largest QR code and reports its bearing and range without drawing on the preview.
- `gesture_recognition.py` runs MediaPipe on the original, unannotated frame, applies the existing gesture mapping, draws hand landmarks and fingertip labels on the preview, and requires a short stable vote before issuing an intent. It does not require face detection or QR tracking.
- `robot_motion.py` passes QR target state to the existing MentorPi adapter and executes intents asynchronously, refreshing its target-relative motion while an intent is active.
- `main.py` wires the modules together. It sends each original camera frame to QR tracking and gesture recognition in parallel, then displays the gesture-annotated frame.

The existing `robot/` adapter and safety layer are reused. `PersonTracker` remains available to the older VA pipeline; this QR tracker is specific to the new modular pipeline.

## Setup

Create a Python 3.14 environment and install the pipeline dependencies:

```bash
cd ~/MentorPi-code/codebase
python3.14 -m venv ~/mentorpi_gesture_venv
source ~/mentorpi_gesture_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements use MediaPipe 1.2.0, which provides Python 3.14 wheels for 64-bit Raspberry Pi OS (`aarch64`). This MediaPipe release is marked Alpha on PyPI. The model downloads automatically when the runner starts and the model file is missing; the machine needs `curl` and internet access for that first download. If the Pi has no internet, download the model on another machine and copy it to `codebase/models/gesture_recognizer.task`.

The QR code's printed width must be known for visual range estimation. The default is 0.15 m. Set `--qr-size` to the actual printed width. LiDAR is preferred when it returns a range; `--no-lidar` disables it. The camera's horizontal field of view also affects bearing and range; adjust it with `--camera-hfov` if needed.

## Run

Start in mock mode first:

```bash
cd ~/MentorPi-code/codebase
source ~/mentorpi_gesture_venv/bin/activate
python3 main.py --source robot
```

For a local USB webcam instead of the robot stream:

```bash
python3 main.py --source webcam --webcam-index 0
```

The processed feed opens in a local OpenCV window. Only the gesture recognizer draws on the feed, showing hand landmarks and fingertip labels. QR tracking continues to estimate target bearing and range for motion without adding QR graphics to the video. Press `q` or Esc in the preview window to stop.

Enable robot movement only when ready:

```bash
python3 main.py --source robot --real-motion
```

The robot runner opens rosbridge at `192.168.149.1:9090`, as the existing adapter does. The robot's standard ROS container bringup must be running. Keep clear space in front of and behind the robot when real motion is enabled; the intent motions can approach or back away. The runner defaults to mock mode.

Other useful options:

```text
--stream-url URL       Override the robot camera URL
--qr-size METERS       Printed QR code width, default 0.15
--camera-hfov DEGREES  Camera horizontal field of view, default 90
--no-lidar             Use QR apparent size for range
--model PATH           Override the Gesture Recognizer model file
--duration SECONDS     Stop after a fixed run time
--debug                Print per-frame gesture detections
```
# mentorpi
