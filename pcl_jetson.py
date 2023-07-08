from datetime import datetime
import time
import numpy as np
import cv2
import imagezmq
import socket
from TFLiteFaceDetector import UltraLightFaceDetecion
from centroidtracker import CentroidTracker
from flask import Flask, render_template, Response

data={'x1':200}
def setData2(x1):
	global data
	data['x1']= x1

	print(f'{x1} {data}')

app=Flask(__name__)
@app.route('/')
def index():
	return render_template('line.html')



model_path="version-slim-320_without_postprocessing.tflite"
tracked_objects = {}
trackerC = CentroidTracker(maxDisappeared=80, maxDistance=90)
cameraID=socket.gethostname()

detector = UltraLightFaceDetecion(model_path,input_size=(320, 240),
				  conf_threshold=0.6,
				  center_variance=0.1,
				  size_variance=0.2,
				  nms_max_output_size=200,
				  nms_iou_threshold=0.5)

def crossing_line():

	cap= cv2.VideoCapture(0)
	#video_path="vid.mp4"
	#cap = cv2.VideoCapture(video_path)
	pft=0
	nft=0

	inObject = 0
	outObject = 0
	up = 0
	down = 0
	inside = 0
	total=0


	# traçage de la région de comptage 
	has_frame,frame = cap.read()
	print(has_frame)
	frame = cv2.flip(frame,1)

	# traçage de la région de comptage 
	has_frame,frame = cap.read()
	print(has_frame)
	frame = cv2.flip(frame,1)
	#ROC = cv2.selectROI('roc', frame)
	contourROC  = np.array([(data['x1'] ,-5),
		                (680, -5),
		                (680, 500),
		                (data['x1'] , 500)], np.int64)
	while True:
		try:
			ret,image=cap.read()
			res=[]
			trackers = []
			rects=[]
			orig_resolution=(image.shape[1], image.shape[0])
			scale=(1,1)
			boxes, scores = detector.inference(image)
		      
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
				box=[0 if i<0 else i for i in bbox]
				obj = {"box": box, "class": "Face","score": scores[result] , "index": 1}
				res.append(obj)


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
					text = "ID: {}".format(objectId)
					cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
				
				# Counting 
					if objectId not in tracked_objects.keys():
						tracked_objects[objectId] = ( bbox, position)
				
						if position == 1:
							inside += 1
				        

					else:
				        # s'il existe,  on le recupere, pour connaitre sa position dans l'ancienne frame
						(bbox, position_ini)   = tracked_objects[objectId]     
				
						if position != position_ini:
							if  position == 1:
				                #inside += 1
								inObject += 1
								up += 1
								t= datetime.now().strftime(" %Y-%m-%d-%H-%M")
								filename="image_{}_{}_{}_{}_{}_face_in_{} ".format(bbox[0],bbox[1],bbox[2],bbox[3],str(cameraID),str(t))
								cv2.imwrite('output/'+ filename+'.jpg', image)
							if position == -1:
				                #inside -= 1
								outObject += 1
								down += 1
								t= datetime.now().strftime(" %Y-%m-%d-%H-%M")
								filename="image_{}_{}_{}_{}_{}_face_out_{} ".format(bbox[0],bbox[1],bbox[2],bbox[3],str(cameraID),str(t))
								cv2.imwrite('output/'+ filename+'.jpg', image)

				                 
							tracked_objects[objectId] = ( bbox, position)

			nft=time.time()
			
			#count fps
			fps= 1/(nft-pft)
			pft=nft
			fps= int(fps)
			fps=str(fps)
			cv2.putText(image, fps, (7,70), cv2.FONT_HERSHEY_PLAIN, 3 , (70,0,255), 3,cv2.LINE_AA)


			#cv2.rectangle(image, (contourROC[x1][y1], contourROC[x2][y1]),(contourROC[x2][y2], contourROC[x1][y2]), (0, 255, 0), 2)
			cv2.rectangle(image, (contourROC[0][0], contourROC[0][1]),(contourROC[2][0], contourROC[2][1]), (0, 255, 0), 2)
			cv2.putText(image, 'in :' + str(inObject)+ " out : "+ str(outObject), (150, 400), cv2.FONT_HERSHEY_PLAIN,   2, (0, 0, 255), 2)
			#cv2.imshow('Test Jetson', image)   
			ret,buffer = cv2.imencode('.jpg',image)
			image = buffer.tobytes()
			yield (b'--image\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

			#key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			#if key == ord("q"):
			#	break   
		except Exception as e:
			print(e)
			pass
#cap.release()
#cap.destroyAllWindow() 

@app.route('/video_feed')
def video_feed():
	return Response(crossing_line() ,mimetype='multipart/x-mixed-replace; boundary=image')
if __name__=='__main__':
	app.run(host='192.168.8.105', port='5000', debug=False, threaded = True)  












