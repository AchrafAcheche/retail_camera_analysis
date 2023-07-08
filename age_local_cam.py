import cv2
import time
from compreface import CompreFace
from compreface.service import RecognitionService
FPS = 1/30000
api_key = '2c49818b-7003-4b8b-8b0e-ab6cb97051fd'
host = 'http://localhost'
port = '8000'
compre_face: CompreFace = CompreFace(host, port, {
            "limit": 0,
            "det_prob_threshold": 0.8,
            "prediction_count": 1,
            "face_plugins": "age",
            "status": False
        })
recognition: RecognitionService = compre_face.init_face_recognition(api_key)
def recoage():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
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
                        age = result.get('age')
                        if box:
                            cv2.rectangle(img=frame, pt1=(box['x_min'], box['y_min']),
                                        pt2=(box['x_max'], box['y_max']), color=(0, 255, 0), thickness=1)
                            if age:
                                age = f"Age: {age['low']} - {age['high']}"
                                cv2.putText(frame, age, (box['x_max'], box['y_min'] + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 0, 200), 2)
                    _, im_buf_arr = cv2.imencode(".jpg", frame)
                    byte_i = im_buf_arr.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byte_i + b'\r\n')
                            
        
        time.sleep(FPS)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
