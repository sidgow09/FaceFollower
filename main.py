import cv2
import mediapipe as mp
import numpy as np
import json
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from calibration import AutoCalibrator 

# --- SETTINGS ---
SETTINGS_FILE = "settings.json"
def load_settings():
    defaults = {"Happy": 35, "Surprise": 30, "Angry": 25, "Disgust": 15, "Sad": 40, "Deadzone": 12, "show_boxes": True, "show_debug": True}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f: return {**defaults, **json.load(f)}
        except: return defaults
    return defaults

def save_settings(state_data):
    with open(SETTINGS_FILE, "w") as f: json.dump(state_data, f)

state = load_settings()
show_boxes, show_debug, panel_open = state['show_boxes'], state['show_debug'], True
calibrator = AutoCalibrator()

# --- AI ---
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, running_mode=vision.RunningMode.VIDEO)
detector = vision.FaceLandmarker.create_from_options(options)

def create_tuning_panel():
    cv2.namedWindow('Tuning Panel')
    for key in ["Happy", "Surprise", "Angry", "Disgust", "Sad", "Deadzone"]:
        limit = 40 if key == "Deadzone" else 100
        cv2.createTrackbar(key, 'Tuning Panel', state[key], limit, lambda x: None)

def on_mouse(event, x, y, flags, param):
    global show_boxes, show_debug, panel_open
    if event == cv2.EVENT_LBUTTONDOWN:
        h = param['h']
        if y > h - 60:
            if 10 <= x <= 110: show_boxes = not show_boxes
            elif 120 <= x <= 220: show_debug = not show_debug
            elif 230 <= x <= 330:
                panel_open = not panel_open
                if not panel_open: cv2.destroyWindow('Tuning Panel')
                else: create_tuning_panel()
            elif 340 <= x <= 440:
                calibrator.start()
            save_settings(state)

def draw_feature_box(img, face_pts, indices, label):
    ih, iw, _ = img.shape
    pts = np.array([[face_pts[i].x * iw, face_pts[i].y * ih] for i in indices], dtype=np.int32)
    rx, ry, rw, rh = cv2.boundingRect(pts)
    cv2.rectangle(img, (rx-5, ry-5), (rx+rw+5, ry+rh+5), (0, 255, 255), 1)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Robot Vision')
ui_params = {'h': 480}
cv2.setMouseCallback('Robot Vision', on_mouse, ui_params)
create_tuning_panel()

FEAT = {'EYES': [33, 133, 362, 263], 'MOUTH': [61, 291, 13, 14], 'BROWS': [70, 107, 336, 300]}

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    ui_params['h'] = h

    if panel_open:
        for k in ["Happy", "Surprise", "Angry", "Disgust", "Sad", "Deadzone"]:
            state[k] = cv2.getTrackbarPos(k, 'Tuning Panel')

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    if result.face_landmarks:
        face_pts = result.face_landmarks[0]
        b = {s.category_name: s.score for s in result.face_blendshapes[0]}
        
        if calibrator.active:
            state = calibrator.process(frame, b, state)
            if panel_open:
                # FIXED: Only update trackbars for keys that exist in state
                for k in calibrator.order:
                    if k in state: cv2.setTrackbarPos(k, 'Tuning Panel', state[k])
        
        if show_boxes:
            for name, idxs in FEAT.items(): draw_feature_box(frame, face_pts, idxs, name)

        # Tracking
        dz, nose = state['Deadzone']/100, face_pts[1]
        p_t = "<< L" if nose.x < 0.5-dz else "R >>" if nose.x > 0.5+dz else "PAN OK"
        t_t = "^ U" if nose.y < 0.5-dz else "v D" if nose.y > 0.5+dz else "TILT OK"
        cv2.putText(frame, f"{p_t} | {t_t}", (20, 40), 1, 1.2, (255,255,255), 2)

        # Mood Check
        mood, m_col = "NEUTRAL", (200, 200, 200)
        if max(b['mouthSmileLeft'], b['mouthSmileRight']) > state['Happy']/100: mood, m_col = "HAPPY", (0, 255, 0)
        elif max(b['eyeWideLeft'], b['jawOpen']) > state['Surprise']/100: mood, m_col = "SURP", (255, 255, 0)
        elif max(b['noseSneerLeft'], b['mouthUpperUpLeft']) > state['Disgust']/100: mood, m_col = "DISG", (0, 165, 255)
        elif max(b['browDownLeft'], b['browDownRight']) > state['Angry']/100: mood, m_col = "ANGRY", (0, 0, 255)
        elif b['browInnerUp'] > state['Sad']/100: mood, m_col = "SAD", (255, 0, 0)
        cv2.putText(frame, f"MOOD: {mood}", (20, 100), 1, 3, m_col, 4)

    # UI BUTTONS
    cv2.rectangle(frame, (0, h-60), (w, h), (30,30,30), -1)
    btns = [("BOXES", show_boxes), ("DEBUG", show_debug), ("PANEL", panel_open), ("AUTO", calibrator.active)]
    for i, (lbl, act) in enumerate(btns):
        bx = 10 + (i * 110)
        cv2.rectangle(frame, (bx, h-50), (bx+100, h-10), (0,255,0) if act else (0,0,255), 2)
        cv2.putText(frame, lbl, (bx+15, h-22), 1, 1, (255,255,255), 1)

    cv2.imshow('Robot Vision', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        save_settings(state)
        break

cap.release()
cv2.destroyAllWindows()