import mediapipe as mp
from threading import Thread
from os.path import dirname, join
from datetime import datetime
import time
#import socket
import numpy as np
#import imagezmq
#import argparse
from pyimagesearch.centroidtracker import CentroidTracker
#import imutils
import kritter
import cv2

#ap = argparse.ArgumentParser()
#ap.add_argument("-s", "--server-ip", required=False, help="ip address of the server", default='127.0.0.1')
#args = vars(ap.parse_args())

win_name = 'DataDoIt'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
resolution=(450,450)
framerate=30
cadre = 15
#cameraID = socket.gethostname()
maxDisappeared=60 
maxDistance=100
min_detection_confidence=0.5
model_selection=1 # 0 = de 0m à 2m , 1 = de 2m à 5m 

#sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(args["server_ip"]))

mpFaceDetection = mp.solutions.face_detection


mpDraw = mp.solutions.drawing_utils

faceDetection = mpFaceDetection.FaceDetection(model_selection= model_selection ,min_detection_confidence=min_detection_confidence)

camera = kritter.Camera(hflip=True, vflip=True)

stream = camera.stream()
frame=stream.frame()

frame=cv2.flip(frame[0],1)
ROC = cv2.selectROI(win_name, frame)

contourROC  = np.array([(ROC[0] , ROC[1]),
                        (ROC[0] + ROC[2], ROC[1]),
                        (ROC[0] + ROC[2], ROC[1]+ ROC[3]),
                        (ROC[0] , ROC[1]+ ROC[3]) ], np.int64)

tracked_objects = {}
tracker = CentroidTracker(maxDisappeared, maxDistance)

count = 0
pTime = 0

#cv2.imshow(win_name, frame[0])
while cv2.waitKey(1)!=27:
    #frame=stream.frame()
    count+=1
    img = stream.frame()
    img = cv2.flip(img[0],1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    
    
    object_current_frame = []
    
    if results.detections:
        
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            #cv2.rectangle(img, bbox, (255, 0, 255), 2)
            #cv2.putText(img, str(id) , (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN,   1, (0, 0, 255), 1)
           
            centroide = int((bbox[0]+ bbox[0]+bbox[2]) / 2), int((bbox[1]+ bbox[1]+bbox[3])/2)
            # cv2.circle(img, (centroide[0], centroide[1]) , 10, (0, 0, 255), 1)

            position = cv2.pointPolygonTest( contourROC , centroide, False)
            
            object_current_frame.append((bbox , centroide , position , 0 , False))  

    if len(object_current_frame) != 0:

        rects=[]
        for object in object_current_frame:
            rects.append((object[0][0], object[0][1], object[0][0] + object[0][2] , object[0][1]+object[0][3]))
            
        tracker.update(rects)

        
        # Affectation de l'ID
        for object in (object_current_frame):       
            
            for key in tracker.objects.keys():
                if tracker.objects[key][0] == object[1][0] and tracker.objects[key][1] == object[1][1]:
                    bbx , center, position , id_object , croped = object_current_frame[object_current_frame.index(object)]    #object[3]
                    object_current_frame[object_current_frame.index(object)] = (bbx , center, position , key , croped)
                    # tracked_objects[key] = (center, position, croped)

        

        for object in (object_current_frame): 
            bbx , center, position , id_object , croped = object
            # comparaison avec l'ancien frame
            if position !=-1:
                
                if id_object not in tracked_objects.keys():
                    tracked_objects[id_object] = (center, position, croped)
                    print('croping object %s' % id_object)
                    crop_img = img[bbx[1]-cadre:bbx[1]+bbx[3]+cadre, bbx[0]-cadre:bbx[0]+bbx[2]+cadre]
                    t= datetime.now().strftime(" %Y-%m-%d-%H-%M")
                    filename="{},bbx: {},{},{},{} ".format(str(t),bbx[0],bbx[1],bbx[2], bbx[3])    
                    cv2.imwrite('output/' +filename+'.jpg', img)
    #effacer le permier 
    #if len(tracked_objects) > 20:
        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_PLAIN,   1, (255, 0, 0), 1)

    # ROC
    cv2.rectangle(img, (contourROC[0][0], contourROC[0][1]),(contourROC[2][0], contourROC[2][1]), (0, 255, 0), 2)
    cv2.imshow(win_name, img)

cv2.destroyAllWindows()

         

