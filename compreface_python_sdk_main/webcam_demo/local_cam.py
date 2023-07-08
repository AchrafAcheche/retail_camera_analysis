import cv2
import argparse
import time
from threading import Thread
from compreface import CompreFace
from compreface.service import RecognitionService
from flask import Flask, render_template, Response

app=Flask(__name__)
@app.route('/')
def index():
	return render_template('add_staff.html')

def test_new_staff():

    FPS = 1/30
    api_key = '00000000-0000-0000-0000-000000000002'
    host = 'http://localhost'
    port = '8000'
    compre_face: CompreFace = CompreFace(host, port, {
                "limit": 0,
                "det_prob_threshold": 0.8,
                "prediction_count": 1,
                "face_plugins": "age,gender",
                "status": False
            })
    recognition: RecognitionService = compre_face.init_face_recognition(api_key)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        _, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()
        data = recognition.recognize(byte_im)
        results = data.get('result')
        if results:
                    results = results
                    for result in results:
                        box = result.get('box')
                        subjects = result.get('subjects')

                        if subjects:
                                subjects = sorted(subjects, key=lambda k: k['similarity'], reverse=True)
                                subject = f"Subject: {subjects[0]['subject']}"
                                similarity = f"Similarity: {subjects[0]['similarity']}"
                                cv2.putText(frame, subject, (box['x_max'], box['y_min'] + 75),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                                cv2.putText(frame, similarity, (box['x_max'], box['y_min'] + 95),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        else:
                                subject = f"No known faces"
                                cv2.putText(frame, subject, (box['x_max'], box['y_min'] + 75),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    ret, buffer = cv2.imencode('.jpg', frame)
    img = buffer.tobytes()
    yield (b'--image\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

        #cv2.imshow('CompreFace demo', frame)
        #time.sleep(FPS)

        #if cv2.waitKey(1) & 0xFF == 27:
        #    cap.release()
        #    cv2.destroyAllWindows()
@app.route('/video_feed')
def video_feed():
	return Response(test_new_staff() ,mimetype='multipart/x-mixed-replace; boundary=image')
if __name__=='__main__':
	app.run(host='192.168.8.105', port='5000', debug=False, threaded = True)
