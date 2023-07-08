import cv2
from compreface import CompreFace
from compreface.service import DetectionService
import urllib.request
import numpy as np
FPS = 1/30000
DOMAIN: str = 'http://localhost'
PORT: str = '8000'
DETECTION_API_KEY: str = '00000000-0000-0000-0000-000000000003'

compre_face: CompreFace = CompreFace(DOMAIN, PORT, {
    "limit": 0,
    "det_prob_threshold": 0.8,
    "face_plugins": "age,gender",
    "status": "true"
})

detection: DetectionService = compre_face.init_face_detection(DETECTION_API_KEY)
def urldetection(urlde):
    req = urllib.request.urlopen(urlde)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    frame = cv2.imdecode(arr, -1)

    frame = cv2.flip(frame, 1)
    _, im_buf_arr = cv2.imencode(".jpg", frame)
    byte_im = im_buf_arr.tobytes()
    #yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byte_im + b'\r\n')
    data = detection.detect(byte_im)
    results = data.get('result')
    if results:
        results = results
        for result in results:
            box = result.get('box')
            age = result.get('age')
            gender = result.get('gender')
            if box:
                cv2.rectangle(img=frame, pt1=(box['x_min'], box['y_min']),
                            pt2=(box['x_max'], box['y_max']), color=(0, 255, 0), thickness=1)
                if age:
                    age = f"Age: {age['low']} - {age['high']}"
                    cv2.putText(frame, age, (box['x_max'], box['y_min'] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                if gender:
                    gender = f"Gender: {gender['value']}"
                    cv2.putText(frame, gender, (box['x_max'], box['y_min'] + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
        _, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_i = im_buf_arr.tobytes()
        
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byte_i + b'\r\n')       
    cv2.waitKey(0)


