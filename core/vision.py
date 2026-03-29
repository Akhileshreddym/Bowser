import cv2
import mediapipe as mp
import math
import logging
import numpy as np
import threading
import time
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """ Calculate the angle between three points (a, b, c) """
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/math.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class VisionTracker:
    def __init__(self, camera_index=0):
        self.lock = threading.Lock()
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        try:
            self.yolo = YOLO('yolov8n.pt')
            self.yolo.to('cpu') # Force CPU to avoid MPS issues on Mac
            logger.info("YOLOv8 model loaded successfully (CPU mode)")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo = None
        
        # Latest vision data produced by process_frame
        self.latest_data = {
            "squat_angle": 180, 
            "arm_angle": 180, 
            "center_x": 0.5, 
            "center_y": 0.5, 
            "distance_depth": 110,
            "shoulder_width": 0.1,
            "t_pose": False,
            "person_detected": False,
            "yolo_height": 0.0,
            "whole_body_visible": False
        }
        
        # Exponential moving average for angle smoothing
        self._ema_squat = 180.0
        self._ema_arm = 180.0
        self._ema_alpha = 0.3
        
        # Depth model
        try:
            from transformers import pipeline
            self.depth_pipe = pipeline("depth-estimation", model="Intel/dpt-large", device="cpu")
        except Exception as e:
            logger.error(f"Could not load depth model: {e}")
            self.depth_pipe = None
            
        self.latest_frame_for_depth = None
        self.depth_thread = threading.Thread(target=self._depth_worker, daemon=True)
        self.depth_thread.start()

    def _depth_worker(self):
        while True:
            frame = self.latest_frame_for_depth
            if frame is not None and self.depth_pipe is not None:
                try:
                    pil_image = Image.fromarray(frame)
                    result = self.depth_pipe(pil_image)
                    depth_map = result["depth"]
                    
                    # Extract depth at multiple points: nose and shoulders
                    with self.lock:
                        cx = self.latest_data.get("center_x", 0.5)
                        cy = self.latest_data.get("center_y", 0.5)
                        sw = self.latest_data.get("shoulder_width", 0.2)
                    
                    sample_pts = [
                        (cx, cy), 
                        (cx - sw/2 if sw > 0.05 else cx - 0.1, cy), 
                        (cx + sw/2 if sw > 0.05 else cx + 0.1, cy),
                    ]
                    
                    w, h = depth_map.size
                    depth_values = []
                    for px, py in sample_pts:
                        x_px = max(0, min(w - 1, int(px * w)))
                        y_px = max(0, min(h - 1, int(py * h)))
                        depth_values.append(depth_map.getpixel((x_px, y_px)))
                    
                    depth_val = float(np.median(depth_values))
                    
                    with self.lock:
                        self.latest_data["distance_depth"] = depth_val
                except Exception as e:
                    pass
            time.sleep(0.1)
        
    def set_camera(self, index_str: str):
        with self.lock:
            logger.info(f"Switching camera to {index_str}")
            self.camera_index = index_str
            if self.cap:
                self.cap.release()
                self.cap = None
            if str(index_str).lower() != "off":
                self.cap = cv2.VideoCapture(int(index_str))
        
    def process_frame(self):
        try:
            with self.lock:
                if not self.cap or not self.cap.isOpened():
                    return None, self.latest_data
                success, image = self.cap.read()
                if not success:
                    return None, self.latest_data

            # Timestamp Overlay
            ts = time.strftime("%H:%M:%S")
            cv2.putText(image, f"LIVE: {ts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Flip for selfie view
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.latest_frame_for_depth = image_rgb.copy()
            
            # ── YOLO PERSON DETECTION ──
            yolo_person_detected = False
            yolo_box = None
            yolo_height = 0.0
            if self.yolo:
                try:
                    yolo_results = self.yolo(image_rgb, classes=[0], conf=0.4, verbose=False)
                    if yolo_results and len(yolo_results[0].boxes) > 0:
                        yolo_person_detected = True
                        yolo_box = yolo_results[0].boxes[0] 
                        b = yolo_box.xyxy[0].cpu().numpy().astype(int)
                        yolo_height = (b[3] - b[1]) / 480.0
                        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
                        cv2.putText(image, "HUMAN", (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                except Exception as e:
                    logger.error(f"YOLO error: {e}")
            
            # ── POSE PROCESSING ──
            results = None
            if yolo_person_detected or self.yolo is None:
                try:
                    results = self.pose.process(image_rgb)
                except Exception as e:
                    logger.error(f"MediaPipe error: {e}")
            
            VIS_THRESH = 0.5
            vision_data = self.latest_data.copy()
            vision_data["person_detected"] = yolo_person_detected
            vision_data["yolo_height"] = yolo_height
            
            if results and results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark
                
                try:
                    # 1. Spatial Tracking (Safe Fallbacks)
                    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    l_hi = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    r_hi = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    l_kn = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    r_kn = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                    l_an = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    r_an = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                    # Center X: Nose is best, otherwise shoulder midpoint
                    if nose.visibility > VIS_THRESH:
                        vision_data["center_x"] = nose.x
                        vision_data["center_y"] = nose.y
                    elif l_sh.visibility > VIS_THRESH and r_sh.visibility > VIS_THRESH:
                        vision_data["center_x"] = (l_sh.x + r_sh.x) / 2.0
                        vision_data["center_y"] = (l_sh.y + r_sh.y) / 2.0

                    # Shoulder Width (Distance metric)
                    if l_sh.visibility > VIS_THRESH and r_sh.visibility > VIS_THRESH:
                        dx, dy = l_sh.x - r_sh.x, l_sh.y - r_sh.y
                        vision_data["shoulder_width"] = math.sqrt(dx*dx + dy*dy)

                    # 2. T-POSE DETECTION (Hip-independent fallback)
                    l_el = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    r_el = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    l_wr = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    r_wr = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                    def is_arm_t(sh, el, wr):
                        if sh.visibility < VIS_THRESH or el.visibility < VIS_THRESH or wr.visibility < VIS_THRESH:
                            return False
                        # Extended: 180 deg
                        ext = calculate_angle(sh, el, wr) > 150
                        # Horizontal: el and wr Y-coords close to sh Y-coord
                        horiz = abs(el.y - sh.y) < 0.12 and abs(wr.y - sh.y) < 0.18
                        return ext and horiz

                    vision_data["t_pose"] = is_arm_t(l_sh, l_el, l_wr) and is_arm_t(r_sh, r_el, r_wr)

                    # 3. Whole Body & Exercise (Optional but kept for clinical)
                    vision_data["whole_body_visible"] = l_an.visibility > VIS_THRESH and r_an.visibility > VIS_THRESH
                    # Squat angle (knee flexion) with EMA smoothing
                    knee_angles = []
                    if (
                        l_hi.visibility > VIS_THRESH and
                        l_kn.visibility > VIS_THRESH and
                        l_an.visibility > VIS_THRESH
                    ):
                        knee_angles.append(calculate_angle(l_hi, l_kn, l_an))
                    if (
                        r_hi.visibility > VIS_THRESH and
                        r_kn.visibility > VIS_THRESH and
                        r_an.visibility > VIS_THRESH
                    ):
                        knee_angles.append(calculate_angle(r_hi, r_kn, r_an))
                    if knee_angles:
                        raw_squat = float(sum(knee_angles) / len(knee_angles))
                        self._ema_squat = (self._ema_alpha * raw_squat) + ((1.0 - self._ema_alpha) * self._ema_squat)
                        vision_data["squat_angle"] = round(self._ema_squat, 2)

                    # Arm raise / side-reach signal from shoulder elevation angle
                    arm_angles = []
                    if (
                        l_el.visibility > VIS_THRESH and
                        l_sh.visibility > VIS_THRESH and
                        l_hi.visibility > VIS_THRESH
                    ):
                        arm_angles.append(calculate_angle(l_el, l_sh, l_hi))
                    if (
                        r_el.visibility > VIS_THRESH and
                        r_sh.visibility > VIS_THRESH and
                        r_hi.visibility > VIS_THRESH
                    ):
                        arm_angles.append(calculate_angle(r_el, r_sh, r_hi))
                    if arm_angles:
                        raw_arm = float(sum(arm_angles) / len(arm_angles))
                        self._ema_arm = (self._ema_alpha * raw_arm) + ((1.0 - self._ema_alpha) * self._ema_arm)
                        vision_data["arm_angle"] = round(self._ema_arm, 2)

                except Exception as e:
                    logger.error(f"Pose data extraction error: {e}")

            with self.lock:
                self.latest_data.update(vision_data)
            
            ret, buffer = cv2.imencode('.jpg', image)
            return buffer.tobytes(), self.latest_data
            
        except Exception as e:
            logger.error(f"Critical process_frame error: {e}", exc_info=True)
            return None, self.latest_data

    def release(self):
        if self.cap: self.cap.release()
