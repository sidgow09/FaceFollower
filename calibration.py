import time
import cv2

class AutoCalibrator:
    def __init__(self):
        self.active = False
        self.step = 0
        self.mode = "PAUSE" 
        self.timer = 0
        self.order = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust"]
        self.baselines = {} 
        self.scores = []
        self.pause_duration = 2.0
        self.record_duration = 3.0

    def start(self):
        self.active = True
        self.step = 0
        self.mode = "PAUSE"
        self.timer = time.time()
        self.scores = []
        self.baselines = {}

    def process(self, frame, blendshapes, state):
        if not self.active: return state

        h, w, _ = frame.shape
        current_target = self.order[self.step]
        elapsed = time.time() - self.timer

        # UI Overlay
        cv2.rectangle(frame, (w//2-220, h//2-70), (w//2+220, h//2+70), (0,0,0), -1)
        
        if self.mode == "PAUSE":
            countdown = int(self.pause_duration - elapsed) + 1
            cv2.putText(frame, f"GET READY: {current_target.upper()}", (w//2-180, h//2-10), 1, 1.2, (0,255,255), 2)
            cv2.putText(frame, f"Starting in {countdown}...", (w//2-80, h//2+30), 1, 1.2, (0,255,255), 2)
            
            if elapsed >= self.pause_duration:
                self.mode = "RECORD"
                self.timer = time.time()
                self.scores = []

        elif self.mode == "RECORD":
            cv2.putText(frame, f"HOLD {current_target.upper()} FACE", (w//2-180, h//2), 1, 1.5, (0,255,0), 2)
            bar_width = int((elapsed / self.record_duration) * 360)
            cv2.rectangle(frame, (w//2-180, h//2+25), (w//2-180 + bar_width, h//2+40), (0,255,0), -1)

            # Map targets to specific blendshapes
            mapping = {
                "Neutral": 0, 
                "Happy": max(blendshapes['mouthSmileLeft'], blendshapes['mouthSmileRight']),
                "Surprise": max(blendshapes['eyeWideLeft'], blendshapes['jawOpen']),
                "Sad": blendshapes['browInnerUp'],
                "Angry": max(blendshapes['browDownLeft'], blendshapes['browDownRight']),
                "Disgust": max(blendshapes['noseSneerLeft'], blendshapes['mouthUpperUpLeft'])
            }
            
            if current_target == "Neutral":
                self.scores.append(blendshapes)
            else:
                self.scores.append(mapping[current_target])

            if elapsed >= self.record_duration:
                state = self.finalize_step(current_target, state)
                self.step += 1
                if self.step >= len(self.order):
                    self.active = False
                else:
                    self.mode = "PAUSE"
                    self.timer = time.time()
        
        return state

    def finalize_step(self, target, state):
        if target == "Neutral":
            # Set baseline for every muscle
            for key in self.scores[0].keys():
                self.baselines[key] = sum(s[key] for s in self.scores) / len(self.scores)
        else:
            avg_peak = sum(self.scores) / len(self.scores)
            
            mapping_keys = {
                "Happy": "mouthSmileLeft", "Surprise": "eyeWideLeft",
                "Sad": "browInnerUp", "Angry": "browDownLeft", "Disgust": "noseSneerLeft"
            }
            resting_val = self.baselines.get(mapping_keys[target], 0)
            
            # Calibration logic: Threshold = Rest + (Peak - Rest) * 0.6
            # We use 0.6 to make it slightly harder to trigger, preventing false positives
            threshold = resting_val + (avg_peak - resting_val) * 0.6
            state[target] = int(max(0.1, threshold) * 100)
        return state