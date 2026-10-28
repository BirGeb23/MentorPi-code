# MentorPi Code

Code for running the team's behavior system on the **Hiwonder MentorPi** wheeled robot. The robot picks one of five intents (`NEUTRAL`, `CHECK_IN`, `ENGAGE`, `CAUTION`, `DE_ESCALATE`) and expresses it through wheel motion. There are two ways to choose the intent:

- **Emotion (VA) pipeline:** the team's valence-arousal pipeline (`va-driven-behaviors/` in the team repo `chaicenter-spot-basics`) reads the person's face and voice. This repo contains the MentorPi side; the pipeline itself stays in the team repo.
- **Hand gestures:** MediaPipe's Gesture Recognizer reads a held hand sign (thumbs up, open palm, ...).

Both paths share the same robot code: one adapter turns intents into wheel commands, and an independent safety layer checks every command before it is sent.

```
                 ┌─ VA pipeline (va/)        ─┐
camera + mic ────┤                            ├─▶ intent ─▶ mentorpi_adapter ─▶ safety ─▶ wheels
                 └─ gesture recognizer (gesture/) ─┘          (robot/)            (robot/)
face tracker + LiDAR (robot/) ──▶ person distance / direction ──┘
```

## Folder layout

| Folder | Contents |
|---|---|
| `robot/` | Shared by everything. `mentorpi_adapter.py` (intent → wheel motion, plus the nod / pulse / scan / greet movements), `safety.py` (speed limits, minimum distance, no-person stop, obstacles, E-stop), `person_tracker.py` (face tracking: visible, direction, distance), `lidar_reader.py` (LiDAR distance). |
| `va/` | Emotion pipeline on the MentorPi. `continuous_runner_http.py` (records 6 s, runs the offline pipeline, acts, repeats), `run_online_mentorpi.py` + `mentorpi_streaming_session.py` (real-time online pipeline), `mentorpi_live_capture.py` (records synchronized video + audio for offline tests). |
| `gesture/` | Gesture pipeline. `run_gesture_mentorpi.py` (gesture → intent → motion), `gesture_check.py` (accuracy test tool). |
| `dashboards/` | `dashboard_server.py` + `dashboard.html` (live monitor, port 5000), `intent_demo_server.py` + `intent_demo.html` (button demo panel, port 5050). |
| `tools/` | Test and diagnostic scripts (see [Tools](#tools)). |
| `docs/` | `VA_Pipeline_Issues_Reference.md` (known VA pipeline issues), `MentorPi_Motion_Mapping_Design.md` (original motion design proposal). |
| `va_pipeline_patches/` | Fixes for files in the team's VA pipeline (frame seeking, per-window audio, `max_workers`), kept as patches to apply to the team repo. See its README. |

**Not in the repo:** the MediaPipe gesture model (download it, see setup) and the JointCAM checkpoints (`va-driven-behaviors/models/*.pt`, too large for Git; get them from the team).

## How the MentorPi side works

The MentorPi runs ROS 2 inside a Docker container named `MentorPi`. Its standard bringup starts automatically on boot and provides:

- **Camera:** `web_video_server` serves the RGB camera as an HTTP stream at `http://192.168.149.1:8080/stream?topic=/ascamera/camera_publisher/rgb0/image` (640×480, about 12 fps).
- **Motion and LiDAR:** `rosbridge` on `ws://192.168.149.1:9090`. The adapter publishes wheel commands to `/controller/cmd_vel`.

All scripts here run **on the Pi's host (outside Docker)** and talk to the robot through those two bridges. `ros2` commands only work inside the container (`docker exec -it MentorPi bash`).

## One-time setup on the Pi

1. **Clone the repos** into the Pi user's home folder:
   ```
   cd ~
   git clone https://github.com/BirGeb23/MentorPi-code.git MentorPi-code
   git clone <team repo URL> chaicenter-spot-basics       # only needed for the VA pipeline
   ```
   The scripts find `robot/` relative to their own location. For the VA pipeline they look for `va-driven-behaviors/` next to this repo, then at `~/chaicenter-spot-basics/va-driven-behaviors`. Set `MENTORPI_VA_DIR` to use another location. The gesture pipeline, dashboards and tools don't need the team repo.

2. **Python environment.** The scripts were developed with the venv at `~/va_pipeline_venv` (Python 3.11, MediaPipe 0.10.9):
   ```
   python3 -m venv ~/va_pipeline_venv
   source ~/va_pipeline_venv/bin/activate
   pip install -r ~/chaicenter-spot-basics/va-driven-behaviors/requirements.txt
   pip install opencv-python mediapipe==0.10.9 websocket-client sounddevice scipy numpy
   ```
   `person_tracker.py` uses MediaPipe's older `mp.solutions` API, which newer MediaPipe releases have removed, so stay on 0.10.9 (the version tested on the Pi) unless `person_tracker.py` is updated.

3. **JointCAM checkpoint** (VA pipeline only): place `jointcam_finetuned_v4.pt` in `va-driven-behaviors/models/`.

4. **Gesture model** (gesture pipeline only). The Pi usually has no internet when your laptop is on the MentorPi's Wi-Fi, so download it on a laptop with internet and copy it over:
   ```
   curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
   scp gesture_recognizer.task pi@<pi-ip>:~/MentorPi-code/gesture/
   ```

5. **Check the camera stream:** open `http://<pi-ip>:8080/stream?topic=/ascamera/camera_publisher/rgb0/image` in a laptop browser on the same network.

Updating later is just `cd ~/MentorPi-code && git pull`.

## Running

Every runner starts in **mock mode**: it prints the wheel commands it would send and the robot does not move. Add `--real-motion` to drive the wheels, only after the mock output looks right. Always start with:

```
source ~/va_pipeline_venv/bin/activate
```

### Emotion (VA) pipeline: cycle runner

Records 6 s of video and audio, runs the offline pipeline on it (`run_offline.py --sparse`), acts on each window's intent, then repeats.

```
cd ~/MentorPi-code/va
python3 continuous_runner_http.py                  # mock
python3 continuous_runner_http.py --real-motion    # wheels move
python3 continuous_runner_http.py --max-cycles 5   # stop after 5 cycles
```

### Emotion (VA) pipeline: online (real-time) runner

Uses the team's ring-buffer streaming session, with the MentorPi camera, WonderEcho Pro microphone and face/LiDAR tracking.

```
cd ~/MentorPi-code/va
python3 run_online_mentorpi.py --duration 60 --window 4.5 --sparse --max-workers 1
python3 run_online_mentorpi.py --duration 60 --window 4.5 --sparse --max-workers 1 --real-motion
```

Recommended settings, from testing on the Pi:

- **`--sparse`:** full (non-sparse) analysis takes far longer than real time on the Pi's CPU.
- **`--max-workers 1`:** two analysis workers compete for the CPU and both slow down.
- **`--window 4.5` with face tracking on:** analysis takes about 4 s per window, so a 3 s window builds a backlog.
- **`--min-confidence 0.1` (testing only):** in `--sparse` mode the pipeline's confidence stays around 0.16, below the default 0.55, so the robot would never act. Use 0.1 only for testing.

Other options: `--no-tracking` (no face/LiDAR tracking), `--no-audio`, `--debug`.

### Gesture pipeline

```
cd ~/MentorPi-code/gesture
python3 run_gesture_mentorpi.py                  # mock, runs until Ctrl+C
python3 run_gesture_mentorpi.py --real-motion    # wheels move
```

| Gesture | Intent |
|---|---|
| Thumb up | ENGAGE |
| Pointing up | CHECK_IN |
| Open palm | CAUTION |
| Thumb down | DE_ESCALATE |
| No hand / other | NEUTRAL |

**How the gesture runner decides:**

- **Face first:** the face tracker finds the person, and the gesture model looks at a crop that follows them (sized from the face). Without the crop, hands are too small to recognize at 1.5 m.
- **Hold check:** a gesture only counts after at least 4 frames of it within 1 s, so a single misread frame can't move the robot.
- **Repeat and release:** while the gesture is held, the intent is re-sent every 2.5 s. 1.5 s after the hand drops, the robot returns to NEUTRAL.
- **Person required:** gestures only count while a face has been seen in the last 2 s.
- **Working range:** tested working at 1.5–2 m.

Options:

- `--debug` prints every frame's gesture.
- `--snapshot PATH` saves what the camera sees after 10 s, with the face (green) and crop (blue) boxes drawn on it.
- `--face-conf 0.3` helps in dim or backlit rooms.
- `--no-lidar` uses face-size distance instead of the LiDAR.

**Accuracy test tool**, no robot motion. Hold one gesture for the whole run:

```
python3 gesture_check.py --duration 20 --delay 5 --crop 0.5 --crop-y 1.0 --expect Thumb_Up
```

- `--crop` / `--crop-y` choose a fixed window. The camera points upward, so people appear low in the frame: try `--crop-y` between 0.8 and 1.0.
- `--snapshot PATH` saves what the model saw.

### Dashboards

```
cd ~/MentorPi-code/dashboards
python3 dashboard_server.py            # live monitor  → http://<pi-ip>:5000/
python3 intent_demo_server.py          # button demo   → http://<pi-ip>:5050/  (add --real-motion to move)
```

- **The live monitor** shows the current intent, VA values, motion, distance and safety state. It reads `/tmp/spot_dashboard_status.json`, which `continuous_runner_http.py` and `run_gesture_mentorpi.py` write while they run, so start one of them in another terminal.
- **The demo panel** triggers any intent with a button, without a pipeline.

### Recording test clips

```
cd ~/MentorPi-code/va
python3 mentorpi_live_capture.py --duration 20
```

Saves a matching video + audio pair for `run_offline.py`. The microphone is fixed at 48 kHz stereo; the script converts to the pipeline's 16 kHz mono.

## What each intent does

Wheel motion, from `robot/mentorpi_adapter.py`:

| Intent | Motion | Stops at |
|---|---|---|
| NEUTRAL | Stop | — |
| CHECK_IN | Forward 0.08 m/s, turns to face the person (0.30 rad/s) | 1.5 m from the person |
| ENGAGE | Forward 0.15 m/s, turns to face the person (0.40 rad/s) | 1.0 m |
| CAUTION | Stop | — |
| DE_ESCALATE | Backward 0.10 m/s | 2.0 m |

Extra movements differ by runner:

| Movement | VA cycle runner | Gesture runner | Demo panel |
|---|---|---|---|
| GREET (ENGAGE, two forward-back pushes) | After 2 ENGAGE results in a row, once per streak, 30 s cooldown | Not used | 2nd ENGAGE press in a row, once per streak |
| ACK nod (CHECK_IN, one forward-back) | Not used | Every time CHECK_IN is sent (every 2.5 s while pointing); set `CHECKIN_ACK_EVERY_DISPATCH = False` for once per streak | Not used |
| PULSE (NEUTRAL, small back-forward) | After 3 NEUTRAL results in a row | Every 20 s while idle | 3 NEUTRAL presses in a row |
| SCAN (CAUTION, backs up turning left/right) | After 2 CAUTION results in a row | Once per CAUTION streak | 2 CAUTION presses in a row |

## Safety

Every command passes through `robot/safety.py`, which the adapter and runners cannot bypass:

| Rule | Value |
|---|---|
| Speed limits | 0.25 m/s forward/backward, 0.5 rad/s turning |
| Minimum distance | Never drives forward when the person is closer than 0.8 m |
| No person | Stops after the person has been missing for a few seconds |
| Obstacles | Stops if the LiDAR sees a non-person object within 0.4 m |
| Low battery | Below 20%, speeds are halved |
| E-stop | Supported |

When running with `--real-motion`:

- **Keep clear space:** about 2.5 m in front of and behind the robot. DE_ESCALATE and the SCAN movement back the robot up.
- **Stay ready to stop:** keep a hand near the power switch. Ctrl+C stops the wheels.
- **Battery:** the runners do not read the real battery level, so watch it yourself on long runs.

## Tools

| Script | Purpose |
|---|---|
| `test_mentorpi_adapter_mock.py` | Runs every intent and safety case through the adapter in mock mode. No robot needed. |
| `test_lidar.py` | Checks the LiDAR connection and what it sees. |
| `test_face_detection.py` | Saves a camera frame with face detection drawn on it. |
| `test_rotation_direction.py` | Checks that the robot turns toward the person, not away. |
| `test_all_intents.py` | Forces each intent in turn on the real robot. |
| `test_new_motions.py` | Tries the GREET, PULSE and SCAN movements. |
| `real_motion_showcase.py` | Real-motion run through all five intents. |
| `offline_full_loop.py` | Runs a saved video through the VA pipeline and then the adapter. |

## Troubleshooting

- **No internet on the Pi:** expected while connected through the MentorPi's own hotspot. Download files on a laptop and `scp` them over.
- **`ros2: command not found`:** `ros2` only exists inside the container. Run `docker exec -it MentorPi bash`, then `source /opt/ros/humble/setup.bash`.
- **Gesture runner shows `person not visible`:** the face isn't found. Use `--snapshot` to check the framing, avoid standing with bright ceiling lights behind you, and try `--face-conf 0.3`.
- **Gestures not recognized:** stand 1.5–2 m away and hold the hand out to the side of your body (not in front of your face or dark clothing), at chest or shoulder height.
- **`Worker overflow` warnings in the online runner:** analysis is slower than the window. Use `--sparse --max-workers 1`, a longer `--window`, or `--no-tracking`.
- **The robot reacts to stale data:** after a countdown or pause, old frames can sit in OpenCV's buffer. The runners drain the stream while waiting; if you write new scripts, do the same.

## Known issues

The VA model loads only 77 of 85 parameters from its checkpoint, because the model code in `test_emotions.py` no longer matches the architecture it was trained with. Its VA output is therefore unreliable, which is the main reason the gesture pipeline exists. See `docs/VA_Pipeline_Issues_Reference.md` for details and the other open VA issues.
