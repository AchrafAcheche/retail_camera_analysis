import cv2
import time
from compreface import CompreFace
from compreface.service import RecognitionService

api_key = '2c49818b-7003-4b8b-8b0e-ab6cb97051fd'
host = 'http://localhost'
port = '8000'
compre_face = CompreFace(host, port, {
    "limit": 0,
    "det_prob_threshold": 0.8,
    "prediction_count": 1,
    "face_plugins": "gender,age",
    "status": False
})
recognition = compre_face.init_face_recognition(api_key)


image_path = "elon.jpg"
frame = cv2.imread(image_path)

_, im_buf_arr = cv2.imencode(".jpg", frame)
byte_im = im_buf_arr.tobytes()

data = recognition.recognize(byte_im)
results = data.get('result')

if results:
    print(results)


   