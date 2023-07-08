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
            "face_plugins": "gender",
            "status": False
        })
recognition: RecognitionService = compre_face.init_face_recognition(api_key)
def recogenderlocal():
    #cap = cv2.VideoCapture(0)
    video_path="vid.mp4"
    cap = cv2.VideoCapture(video_path)

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
                        gender = result.get('gender')
                        if box:
                            cv2.rectangle(img=frame, pt1=(box['x_min'], box['y_min']),
                                        pt2=(box['x_max'], box['y_max']), color=(0, 255, 0), thickness=1)
                            if gender:
                                gender = f"{gender['value']}"
                                cv2.putText(frame, gender, (box['x_max'], box['y_min'] + 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)      
                    _, im_buf_arr = cv2.imencode(".jpg", frame)
                    byte_i = im_buf_arr.tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + byte_i + b'\r\n')
                            
        
        time.sleep(FPS)

        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
