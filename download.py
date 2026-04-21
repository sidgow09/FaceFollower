import urllib.request
import os

# The official URL for the face detection model
url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
filename = "blaze_face_short_range.tflite"

print(f"Downloading {filename}...")
urllib.request.urlretrieve(url, filename)

if os.path.exists(filename):
    print("SUCCESS! The brain file is now in your folder.")
else:
    print("Something went wrong. Check your internet connection.")