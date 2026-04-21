import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import your helpers
from calibration import AutoCalibrator 
from distance import DistanceCalibrator

# --- 1. SETTINGS & PERSISTENCE ---
SETTINGS_FILE = "settings.json"
def load_settings():
    defaults = {
        "Happy": 20, "Surprise": 20, "Angry": 20, "Disgust": 20, "Sad": 20, 
        "Deadzone": 12, "show_boxes": True, "show_debug": True, "TargetWidth": 0.25 
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: return {**defaults, **json.load(f)}
        except: return defaults
    return defaults

state = load_settings()
show_boxes, show_debug, panel_open = state['show_boxes'], state['show_debug'], True

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
            if not panel_open: 
                try: cv2.destroyWindow('Tuning Panel')
                except: pass
            else: create_tuning_panel()
        elif 340 <= x <= 440: emotion_calib.start()
        elif 450 <= x <= 550: dist_calib.start()

def create_tuning_panel():
    cv2.namedWindow('Tuning Panel', cv2.WINDOW_NORMAL)
    for key in ["Happy", "Surprise", "Angry", "Disgust", "Sad", "Deadzone"]:
        cv2.createTrackbar(key, 'Tuning Panel', state[key], 100, lambda x: None)

# --- 3. DRAWING HELPERS ---
FEAT = {'EYES': [33, 133, 362, 263], 'MOUTH': [61, 291, 13, 14], 'BROWS': [70, 107, 336, 300]}

def draw_feature_box(img, face_pts, indices, label):
    ih, iw, _ = img.shape
    pts = np.array([[face_pts[i].x * iw, face_pts[i].y * ih] for i in indices], dtype=np.int32)
    rx, ry, rw, rh = cv2.boundingRect(pts)
    cv2.rectangle(img, (rx-5, ry-5), (rx+rw+5, ry+rh+5), (0, 255, 255), 1)

# --- 4. STARTUP AI ---
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

    # Read sliders
    if panel_open:
        for k in ["Happy", "Surprise", "Angry", "Disgust", "Sad", "Deadzone"]:
            try: state[k] = cv2.getTrackbarPos(k, 'Tuning Panel')
            except: pass

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, frame_count)

    # Initialize current values for Debug Panel
    debug_vals = {"Smile": 0.0, "WideEye": 0.0, "BrowUp": 0.0, "BrowDn": 0.0, "Sneer": 0.0}
    mood, m_col = "NO FACE", (100, 100, 100)

    if result.face_landmarks:
        face_pts = result.face_landmarks[0]
        b = {s.category_name: s.score for s in result.face_blendshapes[0]}
        
        # A. CALIBRATIONS
        if emotion_calib.active:
            state = emotion_calib.process(frame, b, state)
        if dist_calib.active:
            state = dist_calib.process(frame, face_pts, state)

        # B. DISTANCE HUD
        face_w = abs(face_pts[454].x - face_pts[234].x)
        target_w = state['TargetWidth']
        d_msg, d_col = "DISTANCE: OK", (0, 255, 0)
        if face_w < (target_w - 0.03): d_msg, d_col = "--- GET CLOSER ---", (0, 0, 255)
        elif face_w > (target_w + 0.03): d_msg, d_col = "+++ GET FARTHER +++", (0, 0, 255)
        cv2.putText(frame, d_msg, (w//2-140, 40), 1, 1.5, d_col, 2)

        # C. PAN/TILT HUD
        dz, nose = state['Deadzone']/100, face_pts[1]
        p_t = "<< L" if nose.x < 0.5-dz else "R >>" if nose.x > 0.5+dz else "PAN OK"
        t_t = "^ U" if nose.y < 0.5-dz else "v D" if nose.y > 0.5+dz else "TILT OK"
        cv2.putText(frame, f"{p_t} | {t_t}", (20, 40), 1, 1.2, (255, 255, 255), 2)

        # D. MOOD ENGINE
        def get_shift(key):
            return b[key] - emotion_calib.baselines.get(key, 0)

        debug_vals = {
            "Smile": get_shift('mouthSmileLeft'), "WideEye": get_shift('eyeWideLeft'),
            "BrowUp": get_shift('browInnerUp'), "BrowDn": get_shift('browDownLeft'), "Sneer": get_shift('noseSneerLeft')
        }

        mood, m_col = "NEUTRAL", (200, 200, 200)
        if debug_vals["Smile"] > state['Happy']/100: mood, m_col = "HAPPY", (0, 255, 0)
        elif debug_vals["WideEye"] > state['Surprise']/100: mood, m_col = "SURP", (255, 255, 0)
        elif debug_vals["Sneer"] > state['Disgust']/100: mood, m_col = "DISG", (0, 165, 255)
        elif debug_vals["BrowDn"] > state['Angry']/100: mood, m_col = "ANGRY", (0, 0, 255)
        elif debug_vals["BrowUp"] > state['Sad']/100: mood, m_col = "SAD", (255, 0, 0)
        cv2.putText(frame, f"MOOD: {mood}", (20, 110), 1, 3, m_col, 4)

        if show_boxes:
            for name, idxs in FEAT.items(): draw_feature_box(frame, face_pts, idxs, name)

    # --- 5. UI (DEBUG & BUTTONS) ---
    if show_debug:
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-220, 0), (w, 280), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        for i, (lbl, val) in enumerate(debug_vals.items()):
            cv2.putText(frame, f"{lbl}: {val:.2f}", (w-210, 40 + i*45), 1, 1.2, (255,255,255), 2)

    cv2.rectangle(frame, (0, h-60), (w, h), (30, 30, 30), -1)
    btns = [("BOXES", show_boxes), ("DEBUG", show_debug), ("PANEL", panel_open), ("EMOTE", emotion_calib.active), ("DIST", dist_calib.active)]
    for i, (lbl, act) in enumerate(btns):
        bx = 10 + (i * 110)
        cv2.rectangle(frame, (bx, h-50), (bx+100, h-10), (0, 255, 0) if act else (0, 0, 255), 2)
        cv2.putText(frame, lbl, (bx+20, h-22), 1, 1, (255, 255, 255), 1)

    cv2.imshow('Robot Vision', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        state.update({"show_boxes": show_boxes, "show_debug": show_debug})
        with open(SETTINGS_FILE, "w") as f: json.dump(state, f)
        break

cap.release()
cv2.destroyAllWindows()