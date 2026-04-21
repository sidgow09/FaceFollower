import urllib.request
url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
print("Downloading face_landmarker.task... this might take a minute.")
urllib.request.urlretrieve(url, "face_landmarker.task")
print("Done! The file is in your folder.")