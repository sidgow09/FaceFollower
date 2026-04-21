import time
import cv2

class DistanceCalibrator:
    def __init__(self):
        self.active = False
        self.timer = 0
        self.duration = 3.0 # Countdown length
        self.sample_width = 0

    def start(self):
        self.active = True
        self.timer = time.time()
        print("Distance Calibration Started...")

    def process(self, frame, face_pts, state):
        if not self.active:
            return state

        h, w, _ = frame.shape
        elapsed = time.time() - self.timer
        
        # Calculate current face width using landmarks 234 (right) and 454 (left)
        # Formula: $Width = |x_{right} - x_{left}|$
        face_left = face_pts[234].x 
        face_right = face_pts[454].x
        current_width = abs(face_right - face_left)

        # Draw UI Overlay
        cv2.rectangle(frame, (w//2-200, h//2-50), (w//2+200, h//2+50), (0,0,0), -1)
        
        if elapsed < self.duration:
            countdown = int(self.duration - elapsed) + 1
            cv2.putText(frame, f"SET DISTANCE: {countdown}s", (w//2-160, h//2+10), 1, 2, (0,255,255), 2)
        else:
            # Save the measurement and stop
            state['TargetWidth'] = current_width
            self.active = False
            print(f"Target Width Set to: {current_width:.4f}")
            
        return state