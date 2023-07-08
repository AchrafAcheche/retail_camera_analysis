from datetime import datetime
import time
import numpy as np
import cv2
import imagezmq
import socket
import datetime
import csv 
import tensorflow as tf
from TFLiteFaceDetector import UltraLightFaceDetecion
from centroidtracker import CentroidTracker
from flask import Flask, render_template, Response

dat={'x1':0, 'x2':0, 'y1':0, 'y2':0}
def setData3(x1,x2,y1,y2):
	global dat
	dat['x1']= x1
	dat['x2']= x2
	dat['y1']= y1
	dat['y2']= y2
	print(f'{x1} {x2} {y1} {y2} {dat}')
	
app = Flask(__name__)
@app.route('/')
def index():
	# Video streaming home page
	return render_template('time.html')




model_path="version-RFB-320_without_postprocessing.tflite"


tracked_objects = {}
trackerC = CentroidTracker(maxDisappeared=80, maxDistance=90)
cameraID=socket.gethostname()

with open("queue.csv","w") as f:
    writer=csv.writer(f)
    writer.writerow(["id","time", "occupancy", "date"])

model=model_path
threshold=0.6
detector = UltraLightFaceDetecion(model_path,
           input_size=(320, 240), conf_threshold=threshold,
           center_variance=0.1, size_variance=0.2,
           nms_max_output_size=200, nms_iou_threshold=0.5)
inObject = 0
outObject = 0

inside = 0

def time_local():
	total = 0
	pft = 0
	nft = 0

	video_path = "vid.mp4"
	cap = cv2.VideoCapture(video_path)

	# traçage de la région de comptage
	has_frame, frame = cap.read()
	# print(has_frame)
	frame = cv2.flip(frame, 1)
	# ROC = cv2.selectROI('roc', frame)
	
	contourROC  =np.array([(dat['x1'] ,dat['y1']),
		                (dat['x2'], dat['y1']),
		                (dat['x2'], dat ['y2']),
		                (dat['x1'] , dat['y2'])], np.int64)
	while True:
		try:
			ret,image=cap.read()
			#print(image)
			res=[]
			trackers = []
			rects=[]
			#orig_resolution=(image.shape[1], image.shape[0])
			scale=(1,1)


				# Create a TensorImage object from the RGB image
			boxes, scores = detector.inference(image)
				#print(len(boxes),len(scores))
			for result in range(len(boxes.astype(int))):
				if scores[result] < 0.6:
					continue

				x1=int(boxes[result][0])
				y1=int(boxes[result][1])
				x2=int(boxes[result][2])
				y2=int(boxes[result][3])

				bbox= x1,y1,x2,y2
				rects.append(bbox)
				cv2.rectangle(image,(int(x1), int(y1)),(int(x2), int(y2)),(0,0,255),2)

			if len(rects) != 0:
				rects = np.array(rects)
				objects =trackerC.update(rects)

				for (objectId, bbox) in objects.items():
					#print(bbox)
					x1, y1, x2, y2 = bbox
					x1 = int(x1)
					y1 = int(y1)
					x2 = int(x2)
					y2 = int(y2)

					centroide = ( 25*(int((x1+ x2) / (2*25))), 25*(int((y1+y2)/(2*25)) ))
					position = cv2.pointPolygonTest(contourROC , centroide, False)
						#cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
						#text = "ID: {}".format(objectId)
						#cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

						# Counting
					if objectId not in tracked_objects.keys():
						tracked_objects[objectId] = ( bbox, position, datetime.datetime.now())
						if position == 1:
							total += 1

					else:
						# s'il existe,  on le recupere, pour connaitre sa position dans l'ancienne frame
						(bbox, position_ini, firstTime)   = tracked_objects[objectId]
						if position != position_ini:

							if position == 1:
								total += 1
								tracked_objects[objectId] = (bbox, position_ini, datetime.datetime.now())

							if position == -1:
								if total > 0:
									total -= 1
								(bbox, position_ini, firstTime0)   = tracked_objects[objectId]
								t= datetime.datetime.now()
								et= int((datetime.datetime.now() - firstTime0).seconds )
								data=[objectId, et, total,t]
								with open("queue.csv","a") as f:
									writer=csv.writer(f)
									writer.writerow(data)
							tracked_objects[objectId] = ( bbox, position, datetime.datetime.now())
						(bbox, position_ini, firstTime0)   = tracked_objects[objectId]
						text = "Time: {}".format(int((datetime.datetime.now() - firstTime0).seconds ))
						cv2.putText(image, text, (x1+30, y1-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

			nft=time.time()

			#count fps
			fps= 1/(nft-pft)
			pft=nft
			fps= int(fps)
			fps=str(fps)
			cv2.putText(image, fps, (7,70), cv2.FONT_HERSHEY_PLAIN, 3 , (70,0,255), 3,cv2.LINE_AA)


				#self.total = self.inObject - self.outObject +self.inside
			cv2.rectangle(image, (contourROC[0][0], contourROC[0][1]),(contourROC[2][0], contourROC[2][1]), (0, 255, 0), 2)
			cv2.putText(image, ' inside : ' + str(total) , (150, 400), cv2.FONT_HERSHEY_PLAIN,   2, (0, 0, 255), 2)
			#cv2.imshow('Test Jetson', image)while True:
			ret,image=cap.read()
			#print(ret)
			res=[]
			trackers = []
			rects=[]
			orig_resolution=(image.shape[1], image.shape[0])
			scale=(1,1)


				# Create a TensorImage object from the RGB image
			boxes, scores = detector.inference(image)
				#print(len(boxes),len(scores))
			for result in range(len(boxes.astype(int))):
				if scores[result] < 0.6:
					continue

				x1=int(boxes[result][0])
				y1=int(boxes[result][1])
				x2=int(boxes[result][2])
				y2=int(boxes[result][3])

				bbox= x1,y1,x2,y2
				rects.append(bbox)
				cv2.rectangle(image,(int(x1), int(y1)),(int(x2), int(y2)),(0,0,255),2)

			if len(rects) != 0:
				rects = np.array(rects)
				objects =trackerC.update(rects)

				for (objectId, bbox) in objects.items():
					#print(bbox)
					x1, y1, x2, y2 = bbox
					x1 = int(x1)
					y1 = int(y1)
					x2 = int(x2)
					y2 = int(y2)

					centroide = ( 25*(int((x1+ x2) / (2*25))), 25*(int((y1+y2)/(2*25)) ))
					position = cv2.pointPolygonTest(contourROC , centroide, False)
						#cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
						#text = "ID: {}".format(objectId)
						#cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

						# Counting
					if objectId not in tracked_objects.keys():
						tracked_objects[objectId] = ( bbox, position, datetime.datetime.now())
						if position == 1:
							total += 1

					else:
						# s'il existe,  on le recupere, pour connaitre sa position dans l'ancienne frame
						(bbox, position_ini, firstTime)   = tracked_objects[objectId]
						if position != position_ini:

							if position == 1:
								total += 1
								tracked_objects[objectId] = (bbox, position_ini, datetime.datetime.now())

							if position == -1:
								if total > 0:
									total -= 1
								(bbox, position_ini, firstTime0)   = tracked_objects[objectId]
								t= datetime.datetime.now()
								et= int((datetime.datetime.now() - firstTime0).seconds )
								data=[objectId, et, total,t]
								with open("queue.csv","a") as f:
									writer=csv.writer(f)
									writer.writerow(data)
							tracked_objects[objectId] = ( bbox, position, datetime.datetime.now())
						(bbox, position_ini, firstTime0)   = tracked_objects[objectId]
						text = "Time: {}".format(int((datetime.datetime.now() - firstTime0).seconds ))
						cv2.putText(image, text, (x1+30, y1-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

			nft=time.time()

			#count fps
			fps= 1/(nft-pft)
			pft=nft
			fps= int(fps)
			fps=str(fps)
			cv2.putText(image, fps, (7,70), cv2.FONT_HERSHEY_PLAIN, 3 , (70,0,255), 3,cv2.LINE_AA)


				#self.total = self.inObject - self.outObject +self.inside
			cv2.rectangle(image, (contourROC[0][0], contourROC[0][1]),(contourROC[2][0], contourROC[2][1]), (0, 255, 0), 2)
			cv2.putText(image, ' inside : ' + str(total) , (150, 400), cv2.FONT_HERSHEY_PLAIN,   2, (0, 0, 255), 2)
			#cv2.imshow('Test Jetson', image)

			ret, buffer = cv2.imencode('.jpg', image)  # compress and store image to memory buffer
			image = buffer.tobytes()
			yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and return frame

			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
		except Exception as e:
			print(e)
			pass
	cap.release()
	cap.destroyAllWindow()

@app.route('/video_feed')
def video_feed():
    #Video streaming route
    return Response(time_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
	app.run(host='192.168.8.105', port='5000', debug=True, threaded=True)

                    
