import cv2
from compreface import CompreFace
from compreface.service import RecognitionService
import urllib.request
import numpy as np
FPS = 1/30000
DOMAIN: str = 'http://localhost'
PORT: str = '8000'
DETECTION_API_KEY: str = '2c49818b-7003-4b8b-8b0e-ab6cb97051fd'

compre_face: CompreFace = CompreFace(DOMAIN, PORT, {
    "limit": 0,
    "det_prob_threshold": 0.8,
    "status": "true"
})

recognition: RecognitionService = compre_face.init_face_recognition(DETECTION_API_KEY)
def urlrecognition(urlre):
    req = urllib.request.urlopen(urlre)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    frame = cv2.imdecode(arr, -1)

    frame = cv2.flip(frame, 1)
    _, im_buf_arr = cv2.imencode(".jpg", frame)
    byte_im = im_buf_arr.tobytes()
    #yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byte_im + b'\r\n')
    data = recognition.recognize(byte_im)
    results = data.get('result')
    if results:
        results = results
        for result in results:
            box = result.get('box')
            subjects = result.get('subjects')
            if box:
                cv2.rectangle(img=frame, pt1=(box['x_min'], box['y_min']),
                            pt2=(box['x_max'], box['y_max']), color=(0, 255, 0), thickness=1)
                if subjects:
                    subjects = sorted(
                        subjects, key=lambda k: k['similarity'], reverse=True)
                    subject = f"Employer: {subjects[0]['subject']}"
                    similarity = f"Similarity: {subjects[0]['similarity']}"

                    if float(similarity.split(': ')[1]) > 0.8:
                        cv2.putText(frame, subject, (box['x_max'], box['y_min'] + 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        #cv2.putText(frame, similarity, (box['x_max'], box['y_min'] + 95),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    subject = f"Cient"
                    cv2.putText(frame, subject, (box['x_max'], box['y_min'] + 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        _, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_i = im_buf_arr.tobytes()
        
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byte_i + b'\r\n') 
    cv2.waitKey(0)


