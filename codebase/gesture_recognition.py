"""MediaPipe gesture recognition, model setup, and gesture-to-intent voting."""

from collections import Counter, deque
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

GESTURE_TO_INTENT = {
    "Thumb_Up": "ENGAGE",
    "Pointing_Up": "CHECK_IN",
    "Open_Palm": "CAUTION",
    "Thumb_Down": "DE_ESCALATE",
}

MIN_CONFIDENCE_GESTURE_RECOGNITION = 0.6
MIN_CONFIDENCE_HAND_RECOGNITION = 0.3
MIN_TRACKING_CONFIDENCE = 0.7
FINGER_TIPS = {
    "Thumb": 4,
    "Index": 8,
    "Middle": 12,
    "Ring": 16,
    "Pinky": 20,
}


def ensure_model(model_path):
    """Download the MediaPipe task model with curl when missing or empty."""
    if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
        return
    folder = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(folder, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".gesture-model-", dir=folder)
    os.close(fd)
    try:
        print(f"[MODEL] Downloading Gesture Recognizer model to {model_path}")
        subprocess.run(
            ["curl", "-fL", "--retry", "2", "--connect-timeout", "15",
             "--max-time", "300", "-o", temp_path, MODEL_URL],
            check=True,
        )
        if os.path.getsize(temp_path) == 0:
            raise RuntimeError("download returned an empty file")
        os.replace(temp_path, model_path)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Could not download the gesture model: {exc}. Download it manually "
            f"to {model_path} or check curl and internet access."
        ) from exc


class GestureRecognizer:
    """Recognize gestures from original camera frames, without tracking input."""

    def __init__(self, model_path):
        ensure_model(model_path)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(f"MediaPipe Gesture Recognizer unavailable: {exc}") from exc

        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=MIN_CONFIDENCE_HAND_RECOGNITION,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._mp = mp
        self._drawing_utils = vision.drawing_utils
        self._hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_timestamp_ms = 0

    def process_frame(self, original_frame):
        """Return the best gesture and landmarks from the original full frame."""
        rgb = np.ascontiguousarray(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._recognizer.recognize_for_video(image, timestamp_ms)
        hands = []
        categories_per_hand = result.gestures or []
        for index, landmarks in enumerate(result.hand_landmarks or []):
            categories = categories_per_hand[index] if index < len(categories_per_hand) else []
            top = categories[0] if categories else None
            hands.append({
                "landmarks": landmarks,
                "label": top.category_name if top else "Hand detected",
                "score": float(top.score) if top else None,
            })
        if not categories_per_hand or not categories_per_hand[0]:
            return None, None, hands
        top = categories_per_hand[0][0]
        if (top.category_name not in GESTURE_TO_INTENT
                or top.score < MIN_CONFIDENCE_GESTURE_RECOGNITION):
            return None, None, hands
        return top.category_name, float(top.score), hands

    def draw_hand_skeletons(self, frame, hands):
        """Draw detected hand bones and landmark dots onto the preview frame."""
        h, w = frame.shape[:2]
        for hand in hands:
            landmarks = hand["landmarks"]
            self._drawing_utils.draw_landmarks(frame, landmarks, self._hand_connections)
            for name, index in FINGER_TIPS.items():
                if index >= len(landmarks):
                    continue
                point = landmarks[index]
                x, y = int(point.x * w), int(point.y * h)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        return frame

    def close(self):
        self._recognizer.close()


class GestureIntentVoter:
    """Map stable recognized gestures to intents using the runner's hold check."""

    def __init__(self, min_score=0.4, window_s=1.0, min_votes=4, min_share=0.75):
        self.min_score = min_score
        self.window_s = window_s
        self.min_votes = min_votes
        self.min_share = min_share
        self.votes = deque()

    def update(self, timestamp, label, score):
        intent = None
        if label in GESTURE_TO_INTENT and score is not None and score >= self.min_score:
            intent = GESTURE_TO_INTENT[label]
            self.votes.append((timestamp, intent))

        while self.votes and timestamp - self.votes[0][0] > self.window_s:
            self.votes.popleft()
        if not self.votes:
            return None
        counts = Counter(vote for _, vote in self.votes)
        winner, count = counts.most_common(1)[0]
        if count >= self.min_votes and count / len(self.votes) >= self.min_share:
            return winner
        return None
