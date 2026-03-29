import cv2
import mediapipe as mp
import math
import logging
import numpy as np
import threading
import time
from PIL import Image

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
        self.latest_data = {"squat_angle": 180, "arm_angle": 180, "center_x": 0.5, "center_y": 0.5, "distance_depth": 128}
        # Exponential moving average for angle smoothing
        self._ema_squat = 180.0
        self._ema_arm = 180.0
        self._ema_alpha = 0.3  # Lower = smoother but laggier, higher = more responsive
        
        # Super accurate depth model via Transformers
        try:
            from transformers import pipeline
            self.depth_pipe = pipeline("depth-estimation", model="Intel/dpt-large")
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
                    
                    # Extract depth at center of shoulders
                    cx = self.latest_data.get("center_x", 0.5)
                    cy = self.latest_data.get("center_y", 0.5)
                    w, h = depth_map.size
                    x_px = max(0, min(w - 1, int(cx * w)))
                    y_px = max(0, min(h - 1, int(cy * h)))
                    
                    depth_val = depth_map.getpixel((x_px, y_px))
                    
                    with self.lock:
                        self.latest_data["distance_depth"] = depth_val
                except Exception as e:
                    pass
            time.sleep(0.1) # Max 10 fps to prevent CPU overload
        
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
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                return None, self.latest_data
            success, image = self.cap.read()
            if not success:
                logger.warning("Failed to grab frame")
                return None, self.latest_data

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store latest frame for asynchronous depth estimation
        self.latest_frame_for_depth = image_rgb.copy()
        
        # To improve performance, optionally mark the image as not writeable to pass by reference
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True
        
        VIS_THRESH = 0.5  # Lower threshold for better detection
        vision_data = {"squat_angle": 180, "arm_angle": 180, "center_x": 0.5, "shoulder_width": 0.2}
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # ── SQUAT ANGLE: Average both legs ──
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                
                squat_angles = []
                if l_hip.visibility > VIS_THRESH and l_knee.visibility > VIS_THRESH and l_ankle.visibility > VIS_THRESH:
                    squat_angles.append(calculate_angle(l_hip, l_knee, l_ankle))
                if r_hip.visibility > VIS_THRESH and r_knee.visibility > VIS_THRESH and r_ankle.visibility > VIS_THRESH:
                    squat_angles.append(calculate_angle(r_hip, r_knee, r_ankle))
                
                if squat_angles:
                    raw_squat = sum(squat_angles) / len(squat_angles)
                    # Smooth with EMA to prevent jitter
                    self._ema_squat = self._ema_alpha * raw_squat + (1 - self._ema_alpha) * self._ema_squat
                    vision_data["squat_angle"] = self._ema_squat
                    
                    display_knee = l_knee if l_knee.visibility > r_knee.visibility else r_knee
                    cv2.putText(image, f"Squat:{int(self._ema_squat)}", 
                                tuple(np.multiply([display_knee.x, display_knee.y], [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # ── SPATIAL TRACKING ──
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                if left_shoulder.visibility > VIS_THRESH and right_shoulder.visibility > VIS_THRESH:
                    dx = left_shoulder.x - right_shoulder.x
                    dy = left_shoulder.y - right_shoulder.y
                    shoulder_width = math.sqrt(dx*dx + dy*dy)
                    center_x = (left_shoulder.x + right_shoulder.x) / 2.0
                    center_y = (left_shoulder.y + right_shoulder.y) / 2.0
                    
                    vision_data["shoulder_width"] = shoulder_width
                    vision_data["center_x"] = center_x
                    vision_data["center_y"] = center_y
                    
                    # ── ARM RAISE ANGLE: Measure at SHOULDER joint (hip→shoulder→elbow) ──
                    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    
                    arm_angles = []
                    if l_hip.visibility > VIS_THRESH and left_shoulder.visibility > VIS_THRESH and left_elbow.visibility > VIS_THRESH:
                        arm_angles.append(calculate_angle(l_hip, left_shoulder, left_elbow))
                    if r_hip.visibility > VIS_THRESH and right_shoulder.visibility > VIS_THRESH and right_elbow.visibility > VIS_THRESH:
                        arm_angles.append(calculate_angle(r_hip, right_shoulder, right_elbow))
                    
                    if arm_angles:
                        raw_arm = sum(arm_angles) / len(arm_angles)
                        self._ema_arm = self._ema_alpha * raw_arm + (1 - self._ema_alpha) * self._ema_arm
                        vision_data["arm_angle"] = self._ema_arm
                        cv2.putText(image, f"Arm:{int(self._ema_arm)}", 
                                    (50, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    # Render absolute depth for debugging
                    depth = self.latest_data.get("distance_depth", 0)
                    vision_data["distance_depth"] = depth
                    cv2.putText(image, f"Depth: {depth}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # T-pose detection (both arms held out horizontally)
                    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                    # Check if arms are extended (elbow angle > 150 degrees)
                    left_arm_extended = False
                    right_arm_extended = False
                    if l_elbow.visibility > VIS_THRESH and l_wrist.visibility > VIS_THRESH and left_shoulder.visibility > VIS_THRESH:
                        elbow_angle = calculate_angle(left_shoulder, l_elbow, l_wrist)
                        left_arm_extended = elbow_angle > 150
                    if r_elbow.visibility > VIS_THRESH and r_wrist.visibility > VIS_THRESH and right_shoulder.visibility > VIS_THRESH:
                        elbow_angle = calculate_angle(right_shoulder, r_elbow, r_wrist)
                        right_arm_extended = elbow_angle > 150

                    # Check if arms are horizontal (shoulder-elbow angle around 90 degrees)
                    left_arm_horizontal = False
                    right_arm_horizontal = False
                    if l_hip.visibility > VIS_THRESH and left_shoulder.visibility > VIS_THRESH and l_elbow.visibility > VIS_THRESH:
                        shoulder_angle = calculate_angle(l_hip, left_shoulder, l_elbow)
                        left_arm_horizontal = 70 < shoulder_angle < 110
                    if r_hip.visibility > VIS_THRESH and right_shoulder.visibility > VIS_THRESH and r_elbow.visibility > VIS_THRESH:
                        shoulder_angle = calculate_angle(r_hip, right_shoulder, r_elbow)
                        right_arm_horizontal = 70 < shoulder_angle < 110

                    vision_data["t_pose"] = left_arm_extended and right_arm_extended and left_arm_horizontal and right_arm_horizontal
                    cv2.putText(image, f"T-Pose: {vision_data['t_pose']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
            except Exception as e:
                pass
                
        self.latest_data = vision_data
        
        vision_data["person_detected"] = results.pose_landmarks is not None
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        return frame, vision_data

    def release(self):
        self.cap.release()
