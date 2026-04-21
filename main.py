import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import your helper classes
from calibration import AutoCalibrator 
from distance import DistanceCalibrator

# --- 1. SETTINGS ---
SETTINGS_FILE = "settings.json"
def load_settings():
    defaults = {
        "Happy": 20, "Surprise": 20, "Angry": 20, "Disgust": 20, "Sad": 20, 
        "Deadzone": 12, "show_boxes": True, "show_debug": True,
        "TargetWidth": 0.25 
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: return {**defaults, **json.load(f)}
        except: return defaults
    return defaults

state = load_settings()
show_boxes, show_debug, panel_open = state['show_boxes'], state['show_debug'], True

# Initialize both calibrators
emotion_calib = AutoCalibrator()
dist_calib = DistanceCalibrator()

# --- 2. UI HANDLERS ---
def on_mouse(event, x, y, flags, param):
    global show_boxes, show_debug, panel_open
    if event == cv2.EVENT_LBUTTONDOWN and y > param['h'] - 60:
        if 10 <= x <= 110: show_boxes = not show_boxes
        elif 120 <= x <= 220: show_debug = not show_debug
        elif 230 <= x <= 330:
            panel_open = not panel_open
            if not panel_open: cv2.destroyWindow('Tuning Panel')
            else: create_tuning_panel()
        elif 340 <= x <= 440: emotion_calib.start()
        elif 450 <= x <= 550: dist_calib.start() # Trigger distance calibration

def create_tuning_panel():
    cv2.namedWindow('Tuning Panel', cv2.WINDOW_NORMAL)
    for key in ["Happy", "Surprise", "Angry", "Disgust", "Sad", "Deadzone"]:
        cv2.createTrackbar(key, 'Tuning Panel', state[key], 100, lambda x: None)

# --- 3. STARTUP ---
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, running_mode=vision.RunningMode.VIDEO)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Robot Vision')
ui_params = {'h': 480}
cv2.setMouseCallback('Robot Vision', on_mouse, ui_params)
create_tuning_panel()
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    ui_params['h'] = h
    frame_count += 1

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, frame_count)

    if result.face_landmarks:
        face_pts = result.face_landmarks[0]
        b = {s.category_name: s.score for s in result.face_blendshapes[0]}
        
        # A. Process Emotion Calibration
        if emotion_calib.active:
            state = emotion_calib.process(frame, b, state)
            
        # B. Process Distance Calibration
        if dist_calib.active:
            state = dist_calib.process(frame, face_pts, state)

        # C. Distance Feedback HUD
        face_w = abs(face_pts[454].x - face_pts[234].x)
        target = state['TargetWidth']
        tol = 0.03 
        
        d_msg, d_col = "DISTANCE: OK", (0, 255, 0)
        if face_w < (target - tol): d_msg, d_col = "--- GET CLOSER ---", (0, 0, 255)
        elif face_w > (target + tol): d_msg, d_col = "+++ GET FARTHER +++", (0, 0, 255)
        cv2.putText(frame, d_msg, (w//2-140, 40), 1, 1.5, d_col, 2)

        # ... (Insert Mood Logic Here) ...

    # --- DRAW BUTTONS ---
    cv2.rectangle(frame, (0, h-60), (w, h), (30, 30, 30), -1)
    btns = [("BOXES", show_boxes), ("DEBUG", show_debug), ("PANEL", panel_open), ("EMOTE", emotion_calib.active), ("DIST", dist_calib.active)]
    for i, (lbl, act) in enumerate(btns):
        bx = 10 + (i * 110)
        cv2.rectangle(frame, (bx, h-50), (bx+100, h-10), (0, 255, 0) if act else (0, 0, 255), 2)
        cv2.putText(frame, lbl, (bx+20, h-22), 1, 1, (255, 255, 255), 1)

    cv2.imshow('Robot Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        with open(SETTINGS_FILE, "w") as f: json.dump(state, f)
        break

cap.release()
cv2.destroyAllWindows()