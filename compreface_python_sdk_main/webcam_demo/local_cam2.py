import cv2
import time
from compreface import CompreFace
from compreface.service import RecognitionService
FPS = 1/30
api_key = '00000000-0000-0000-0000-000000000002'
host = 'http://localhost'
port = '8000'
compre_face: CompreFace = CompreFace(host, port, {
            "limit": 0,
            "det_prob_threshold": 0.8,
            "prediction_count": 1,
            "status": False
        })
recognition: RecognitionService = compre_face.init_face_recognition(api_key)

cap = cv2.VideoCapture(0)
results=[]
while True:

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
                            subject = f"Subject: {subjects[0]['subject']}"
                            similarity = f"Similarity: {subjects[0]['similarity']}"
                            cv2.putText(frame, subject, (box['x_max'], box['y_min'] + 75),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                            cv2.putText(frame, similarity, (box['x_max'], box['y_min'] + 95),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        else:
                            subject = f"No known faces"
                            cv2.putText(frame, subject, (box['x_max'], box['y_min'] + 75),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    ret, frame = cap.read()
    _, im_buf_arr = cv2.imencode(".jpg", frame)
    byte_im = im_buf_arr.tobytes()
    data = recognition.recognize(byte_im)
    results = data.get('result')

    cv2.imshow('CompreFace demo', frame)
    time.sleep(FPS)

    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
