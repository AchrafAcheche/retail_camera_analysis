import time
import cv2
from flask import Flask, render_template, Response
from os.path import exists
from os import mkdir
from random import randint
counter = 0
ret = None
image = None
app=Flask(__name__)
@app.route('/')
def index():
	return render_template('add_staff.html')

def capt(nom):
	global counter
	if not exists('staffs/' + nom):
		mkdir('staffs/' + nom)
	img_name = 'staffs/' + nom + '/' + nom + "_{}.jpg".format(counter)
	while exists(img_name):
		counter += 1
		img_name = 'staffs/' + nom + '/' + nom + "_{}.jpg".format(counter)
	cv2.imwrite(img_name, image)

def add_new_staff():
	global ret, image
	#cap= cv2.VideoCapture("nvarguscamerasrc ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)
	cap= cv2.VideoCapture(0)
	pft=0
	nft=0



	while True:
#		try:
			ret, image = cap.read()
			#print(ret)

			nft=time.time()

			#count fps
			fps= 1/(nft-pft)
			pft=nft
			fps= int(fps)
			fps=str(fps)
			cv2.putText(image, fps, (7,70), cv2.FONT_HERSHEY_PLAIN, 3 , (70,0,255), 3,cv2.LINE_AA)



			ret,buffer = cv2.imencode('.jpg',image)
			img = buffer.tobytes()
			yield (b'--image\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')




#		except Exception as e:
#			print(e)
#			pass
@app.route('/video_feed')
def video_feed():
	return Response(add_new_staff() ,mimetype='multipart/x-mixed-replace; boundary=image')
if __name__=='__main__':
	app.run(host='192.168.8.105', port='5000', debug=False, threaded = True)  












