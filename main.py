import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os

# The name of our "memory" file
SETTINGS_FILE = "settings.json"

# Default values in case the file doesn't exist yet
default_settings = {
    "Happy": 35, "Surprise": 30, "Angry": 25, 
    "Disgust": 15, "Sad": 40, "Deadzone": 12
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return default_settings

def save_settings(current_vals):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(current_vals, f)

# Load them up right at the start
saved_vals = load_settings()

# --- 1. SETUP AI TASK ---
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.FaceLandmarker.create_from_options(options)

# --- 2. DETAILED FEATURE BOXES ---
FEAT = {
    'EYES': [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380],
    'MOUTH': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
    'BROWS': [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    'CHEEKS': [205, 203, 425, 423] # Added Cheeks for Disgust/Pain tracking
}

# --- 3. TUNING PANEL ---
def nothing(x): pass
cv2.namedWindow('Tuning Panel')
cv2.resizeWindow('Tuning Panel', 400, 450)
cv2.createTrackbar('Happy', 'Tuning Panel', saved_vals['Happy'], 100, nothing)
cv2.createTrackbar('Surprise', 'Tuning Panel', saved_vals['Surprise'], 100, nothing)
cv2.createTrackbar('Angry', 'Tuning Panel', saved_vals['Angry'], 100, nothing)
cv2.createTrackbar('Disgust', 'Tuning Panel', saved_vals['Disgust'], 100, nothing)
cv2.createTrackbar('Sad', 'Tuning Panel', saved_vals['Sad'], 100, nothing)
cv2.createTrackbar('Deadzone', 'Tuning Panel', saved_vals['Deadzone'], 100, nothing)

def draw_feature_box(img, face_pts, indices, label, color):
    h, w, _ = img.shape
    pts = np.array([[face_pts[i].x * w, face_pts[i].y * h] for i in indices], dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    cv2.rectangle(img, (x-5, y-5), (x+bw+5, y+bh+5), color, 1)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Read sliders
    t_happy = cv2.getTrackbarPos('Happy', 'Tuning Panel') / 100
    t_surp  = cv2.getTrackbarPos('Surprise', 'Tuning Panel') / 100
    t_ang   = cv2.getTrackbarPos('Angry', 'Tuning Panel') / 100
    t_dis   = cv2.getTrackbarPos('Disgust', 'Tuning Panel') / 100
    t_sad   = cv2.getTrackbarPos('Sad', 'Tuning Panel') / 100
    dz_val  = cv2.getTrackbarPos('Deadzone', 'Tuning Panel') / 100
    dz_low, dz_high = 0.5 - dz_val, 0.5 + dz_val

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    if result.face_landmarks:
        face_pts = result.face_landmarks[0]
        for name, idxs in FEAT.items():
            draw_feature_box(frame, face_pts, idxs, name, (100, 100, 100))

        # --- 2D TRACKING ---
        nose = face_pts[1]
        p_text = "<<< LEFT" if nose.x < dz_low else "RIGHT >>>" if nose.x > dz_high else "PAN: OK"
        t_text = "^^^ UP" if nose.y < dz_low else "vvv DOWN" if nose.y > dz_high else "TILT: OK"

        # --- REFINED EMOTION ENGINE ---
        b = {s.category_name: s.score for s in result.face_blendshapes[0]}
        
        # We define TIE-BREAKERS here
        is_smiling = max(b['mouthSmileLeft'], b['mouthSmileRight']) > t_happy
        is_wide_eyed = max(b['eyeWideLeft'], b['eyeWideRight']) > t_surp or b['jawOpen'] > t_surp
        is_scrunched = max(b['noseSneerLeft'], b['noseSneerRight']) > t_dis or b['mouthUpperUpLeft'] > t_dis
        is_furrowed = max(b['browDownLeft'], b['browDownRight']) > t_ang
        is_frowning = max(b['browInnerUp'], b['mouthFrownLeft'], b['mouthFrownRight']) > t_sad

        # PRIORITY DECISION TREE
        if is_smiling:
            mood, m_col = "HAPPY", (0, 255, 0)
        elif is_wide_eyed:
            mood, m_col = "SURPRISED", (255, 255, 0)
        elif is_scrunched: # Check Disgust BEFORE Anger to catch the nose scrunch
            mood, m_col = "DISGUSTED", (0, 165, 255)
        elif is_furrowed:
            mood, m_col = "ANGRY", (0, 0, 255)
        elif is_frowning:
            mood, m_col = "SAD", (255, 0, 0)
        else:
            mood, m_col = "NEUTRAL", (200, 200, 200)

        # UI DISPLAY
        cv2.putText(frame, f"{p_text} | {t_text}", (20, 40), 1, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"MOOD: {mood}", (20, 100), 1, 3, m_col, 4)

        # --- DETAILED DEBUG PANEL ---
        # Show specific conflict values
        debug_vars = [
            ('Smile', max(b['mouthSmileLeft'], b['mouthSmileRight'])),
            ('NoseSneer', max(b['noseSneerLeft'], b['noseSneerRight'])),
            ('UpperLipUp', max(b['mouthUpperUpLeft'], b['mouthUpperUpRight'])),
            ('BrowDown', max(b['browDownLeft'], b['browDownRight'])),
            ('MouthPress', max(b['mouthPressLeft'], b['mouthPressRight'])),
            ('BrowInnerUp', b['browInnerUp'])
        ]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-260, 0), (w, 250), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        for i, (label, val) in enumerate(debug_vars):
            color = (0, 255, 0) if val > 0.2 else (255, 255, 255)
            cv2.putText(frame, f"{label}: {val:.2f}", (w-240, 30 + (i*35)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow('Robot Vision v2.0', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break
# Right before cap.release()
current_tuning = {
    "Happy": cv2.getTrackbarPos('Happy', 'Tuning Panel'),
    "Surprise": cv2.getTrackbarPos('Surprise', 'Tuning Panel'),
    "Angry": cv2.getTrackbarPos('Angry', 'Tuning Panel'),
    "Disgust": cv2.getTrackbarPos('Disgust', 'Tuning Panel'),
    "Sad": cv2.getTrackbarPos('Sad', 'Tuning Panel'),
    "Deadzone": cv2.getTrackbarPos('Deadzone', 'Tuning Panel')
}
save_settings(current_tuning)
print("Settings saved to settings.json!")

cap.release()
cv2.destroyAllWindows()