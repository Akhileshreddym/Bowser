import cv2
import numpy as np
import math
import logging
import threading
import time
import torch
from PIL import Image
from ultralytics import YOLO
from rtmlib import Wholebody
from transformers import pipeline

logger = logging.getLogger(__name__)

def calculate_angle(a, b, c):
    """ Calculate the angle between three points """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

class SOTAVisionEngine:
    """
    High-Performance SOTA Vision Engine (M3 Pro Optimization)
    Featuring: YOLOv11-seg (Real-time), RTMPose, Depth Anything V2
    """
    def __init__(self, camera_index=0):
        self.lock = threading.Lock()
        self.camera_index = camera_index
        # Use AVFoundation for Mac stability
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"SOTAVisionEngine optimized for {self.device}")

        # 1. POSE: RTMPose (Fastest skeletal tracking)
        try:
            self.pose_tracker = Wholebody(to_openpose=False, backend='onnxruntime', device='cpu')
            logger.info("RTMPose LOADED")
        except: self.pose_tracker = None

        # 2. DEPTH: Depth Anything V2 Small (MPS Accelerated)
        try:
            self.depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if self.device=="mps" else -1)
            logger.info("Depth V2 (MPS) LOADED")
        except: self.depth_pipe = None

        # 3. SEGMENTATION: YOLOv11-seg (SOTA Performance)
        try:
            # nano or small seg for ultra-low latency
            self.seg_model = YOLO("yolo11n-seg.pt")
            logger.info("YOLOv11-seg LOADED")
        except: self.seg_model = None

        self.latest_data = {
            "squat_angle": 180, "arm_angle": 180, "center_x": 0.5, "center_y": 0.5,
            "distance_depth": 110, "left_depth": 0, "mid_depth": 0, "right_depth": 0,
            "obstacle_depth": 0, "yolo_height": 0.0,
            "shoulder_width": 0.1, "t_pose": False, "person_detected": False,
            "whole_body_visible": False
        }
        
        self.latest_frame_for_workers = None
        self.depth_map = None
        self.segment_mask = None
        
        self.worker_thread = threading.Thread(target=self._heavy_model_worker, daemon=True)
        self.worker_thread.start()

        self._ema_squat, self._ema_arm, self._ema_height = 180.0, 180.0, 0.0
        self._ema_alpha = 0.5 # Faster smoothing for SOTA

    def set_camera(self, index):
        if index == 'off':
            self.cap.release()
            self.camera_index = -1
            return
        
        try:
            new_index = int(index)
            with self.lock:
                if self.cap.isOpened(): self.cap.release()
                self.camera_index = new_index
                time.sleep(0.3)
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
                if not self.cap.isOpened():
                    for i in range(4):
                        if i == new_index: continue
                        temp = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                        if temp.isOpened():
                            self.cap, self.camera_index = temp, i
                            break
        except: pass

    def _heavy_model_worker(self):
        """ Depth and Segmentation loop (Async) """
        while True:
            frame = self.latest_frame_for_workers
            if frame is not None:
                # 1. Depth (Relative Depth)
                if self.depth_pipe:
                    try:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        res = self.depth_pipe(pil_img)
                        self.depth_map = np.array(res["depth"])
                        h, w = self.depth_map.shape
                        norm = cv2.normalize(self.depth_map, None, 0, 255, cv2.NORM_MINMAX)
                        with self.lock:
                            self.latest_data["left_depth"] = int(np.mean(norm[int(h*0.4):int(h*0.8), int(w*0.1):int(w*0.3)]))
                            self.latest_data["mid_depth"] = int(np.mean(norm[int(h*0.4):int(h*0.8), int(w*0.4):int(w*0.6)]))
                            self.latest_data["right_depth"] = int(np.mean(norm[int(h*0.4):int(h*0.8), int(w*0.7):int(w*0.9)]))
                            self.latest_data["obstacle_depth"] = self.latest_data["mid_depth"]
                    except: pass
                
                # 2. YOLOv11-seg (SOTA Tracking)
                if self.seg_model:
                    try:
                        results = self.seg_model.predict(frame, device=self.device, verbose=False)
                        if results and results[0].masks:
                            self.segment_mask = results[0].masks.data[0].cpu().numpy()
                    except: pass
            time.sleep(0.02) # Fast worker loop (~50FPS potential)

    def process_frame(self):
        if self.camera_index == -1: return None, self.latest_data
        
        ret, frame = self.cap.read()
        if not ret:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy, "CAM OFFLINE / SYNCING", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            _, buf = cv2.imencode('.jpg', dummy)
            return buf.tobytes(), self.latest_data
        
        self.latest_frame_for_workers = frame.copy()
        display_frame = frame.copy()
        
        # 1. POSE (Main Thread for zero lag)
        if self.pose_tracker:
            try:
                kp_list, sc_list = self.pose_tracker(frame)
                if len(kp_list) > 0:
                    kp, sc = kp_list[0], sc_list[0]
                    self.latest_data["person_detected"] = True
                    
                    # ── STABLE METRIC MAPPING ──
                    # Use LS/RS midpoint to LH/RH midpoint for robust height
                    torso_h = np.abs(np.mean(kp[11:13, 1]) - np.mean(kp[5:7, 1]))
                    # Calibrate height: Target is ~0.8-0.9 frame height for "At Distance"
                    raw_height = float(torso_h * 2.6 / frame.shape[0])
                    self._ema_height = 0.5 * raw_height + 0.5 * self._ema_height
                    self.latest_data["yolo_height"] = self._ema_height
                    
                    knee_a = calculate_angle(kp[12], kp[14], kp[16])
                    arm_a = calculate_angle(kp[11], kp[5], kp[7])
                    self._ema_squat = 0.5 * knee_a + 0.5 * self._ema_squat
                    self._ema_arm = 0.5 * arm_a + 0.5 * self._ema_arm
                    self.latest_data["squat_angle"], self.latest_data["arm_angle"] = int(self._ema_squat), int(self._ema_arm)
                    self.latest_data["center_x"] = float(np.mean(kp[:, 0]) / frame.shape[1])
                    
                    for (x, y) in kp[:17]:
                        cv2.circle(display_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            except: pass

        # 2. OVERLAYS (Liquid Smooth)
        if self.segment_mask is not None:
            try:
                mask_res = cv2.resize(self.segment_mask, (display_frame.shape[1], display_frame.shape[0]))
                mask_color = np.zeros_like(display_frame)
                mask_color[mask_res > 0.5] = [211, 255, 52] 
                display_frame = cv2.addWeighted(display_frame, 1.0, mask_color, 0.4, 0)
            except: pass

        _, buffer = cv2.imencode('.jpg', display_frame)
        return buffer.tobytes(), self.latest_data

    def release(self): self.cap.release()
