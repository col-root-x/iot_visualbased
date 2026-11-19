import os
import time
import threading
import cv2
import joblib
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ==========================================
# 1. THREADED ZERO-LAG RTSP READER
# ==========================================
class RTSPStream:
    def __init__(self, url):
        # Use TCP to stop H.264 corruption/smearing
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()

# ==========================================
# 2. CONFIGURATION
# ==========================================
RTSP_URL = "rtsp://100.91.4.6:8554/cam"

# Paste the line printed by zone_mapper.py here:
restricted_zone = np.array([[43, 16], [56, 454], [603, 466], [606, 13], [45, 18]], np.int32)

WINDOW_SIZE = 30
LOITER_THRESHOLD = 5.0
FIGHT_CONF_THRESH = 0.65

# ==========================================
# 3. LOAD MODELS & START STREAM
# ==========================================
print("Loading YOLOv8 Pose model...")
model = YOLO("yolov8n-pose.pt")

print("Loading Random Forest fight classifier...")
fight_clf = joblib.load("fight_classifier.pkl")

pose_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
zone_entry_times = {}

print(f"Connecting to camera stream at {RTSP_URL}...")
stream = RTSPStream(RTSP_URL)

# Wait briefly for the first frame
time.sleep(1.0)

# ==========================================
# 4. MAIN LOOP
# ==========================================
while True:
    ret, frame = stream.read()
    if not ret:
        time.sleep(0.05)
        continue

    # imgsz=320 keeps inference fast on CPU (~60ms)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=320, conf=0.4, verbose=False)
    annotated_frame = results[0].plot(boxes=False)

    if len(restricted_zone) >= 3:
        cv2.polylines(annotated_frame, [restricted_zone], isClosed=True, color=(255, 255, 0), thickness=2)

    active_ids = set()
    now = time.time()
    combat_detected = False
    loitering_detected = False
    zone_breach = False

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        keypoints_xy = results[0].keypoints.xy.cpu().numpy()

        for box, track_id, kpts in zip(boxes, track_ids, keypoints_xy):
            active_ids.add(track_id)
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int(y2)

            flat_kpts = kpts.flatten()
            if flat_kpts.shape[0] == 34:
                pose_history[track_id].append(flat_kpts)

            is_person_fighting = False
            if len(pose_history[track_id]) == WINDOW_SIZE:
                feature_window = np.concatenate(pose_history[track_id]).reshape(1, -1)
                fight_prob = fight_clf.predict_proba(feature_window)[0][1]
                if fight_prob >= FIGHT_CONF_THRESH:
                    is_person_fighting = True
                    combat_detected = True

            is_inside = False
            if len(restricted_zone) >= 3:
                is_inside = cv2.pointPolygonTest(restricted_zone, (cx, cy), False) >= 0

            box_color = (0, 255, 0)
            status = f"ID:{track_id}"

            if is_person_fighting:
                box_color = (0, 0, 255)
                status = f"ID:{track_id} | COMBAT"
            elif is_inside:
                zone_breach = True
                if track_id not in zone_entry_times:
                    zone_entry_times[track_id] = now
                dwell_time = now - zone_entry_times[track_id]

                if dwell_time >= LOITER_THRESHOLD:
                    loitering_detected = True
                    box_color = (0, 0, 255)
                    status = f"ID:{track_id} | LOITERING ({dwell_time:.1f}s)"
                else:
                    box_color = (0, 165, 255)
                    status = f"ID:{track_id} | ZONE ({dwell_time:.1f}s)"
                cv2.circle(annotated_frame, (cx, cy), 5, box_color, -1)
            else:
                zone_entry_times.pop(track_id, None)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated_frame, status, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    for tid in list(pose_history.keys()):
        if tid not in active_ids:
            del pose_history[tid]
    for tid in list(zone_entry_times.keys()):
        if tid not in active_ids:
            del zone_entry_times[tid]

    alert_active = False
    if combat_detected:
        cv2.putText(annotated_frame, "CRITICAL: COMBAT DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        alert_active = True

    if loitering_detected:
        cv2.putText(annotated_frame, "ALERT: LOITERING DETECTED", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        alert_active = True
    elif zone_breach:
        cv2.putText(annotated_frame, "WARNING: ZONE BREACH", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    if alert_active:
        cv2.rectangle(annotated_frame, (0, 0),
                      (annotated_frame.shape[1] - 1, annotated_frame.shape[0] - 1),
                      (0, 0, 255), 4)

    cv2.imshow("Intelligence Layer Dashboard", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.release()
cv2.destroyAllWindows()
