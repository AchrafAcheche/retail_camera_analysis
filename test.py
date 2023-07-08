from datetime import datetime
import time
import numpy as np
import cv2
import imagezmq
import socket
from TFLiteFaceDetector import UltraLightFaceDetecion
from centroidtracker import CentroidTracker

model_path="/home/jetson/Downloads/apps/face_detect/version-RFB-320_without_postprocessing.tflite"


tracked_objects = {}
trackerC = CentroidTracker(maxDisappeared=80, maxDistance=90)
cameraID=socket.gethostname()


detector = UltraLightFaceDetecion(model_path,
			input_size=(320, 240), conf_threshold=0.6,
                        center_variance=0.1, size_variance=0.2,
                        nms_max_output_size=200, nms_iou_threshold=0.5)
inObject = 0
outObject = 0
total = 0
inside = 0
cap=cv2.VideoCapture(0)



# traçage de la région de comptage 
has_frame,frame = cap.read()
print(has_frame)
frame = cv2.flip(frame,1)


while True:
	ret,image=cap.read()
	print(ret)
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
		       
		        #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
		        #text = "ID: {}".format(objectId)
		        #cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
		        
		        # Counting 
		        if objectId not in tracked_objects.keys():
		            tracked_objects[objectId] = ( bbox, position)
		        
		            if position == 1:
		                total += 1
		                

		        else:
		        # s'il existe,  on le recupere, pour connaitre sa position dans l'ancienne frame
		            (bbox, position_ini)   = tracked_objects[objectId]     
		        
		            if position != position_ini:
		                
		                if position == 1:
		                    total += 1
		                    t= datetime.now().strftime(" %Y-%m-%d-%H-%M")
		                    filename="image_{}_{}_{}_{}_{}_face_event_{} ".format(bbox[0],bbox[1],bbox[2],bbox[3],str(cameraID),str(t))
		                    cv2.imwrite('/home/jetson/Downloads/apps/face_detect/output/'+ filename+'.jpg', image)
		                if position == -1:
		                    total -= 1
		                    
		                tracked_objects[objectId] = ( bbox, position)
                
                
        #self.total = self.inObject - self.outObject +self.inside
        
	cv2.putText(image, 'total :' + str(total), (150, 400), cv2.FONT_HERSHEY_PLAIN,   2, (0, 0, 255), 2)
	cv2.imshow('Test Jetson', image)   
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break   

cap.release()
cap.destroyAllWindow()   
               

                    
