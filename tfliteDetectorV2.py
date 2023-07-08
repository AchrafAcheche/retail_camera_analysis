from datetime import datetime
import time
import numpy as np
#from pyimagesearch.centroidtracker import CentroidTracker
import kritter
from kritter import KimageDetector
import cv2
import imagezmq
import socket
import schedule
from TFLiteFaceDetector import UltraLightFaceDetecion
from centroidtracker import CentroidTracker

model_path="/home/pi/face/version-RFB-320_without_postprocessing.tflite"


tracked_objects = {}
trackerC = CentroidTracker(maxDisappeared=80, maxDistance=90)
cameraID=socket.gethostname()
class TFLiteDetector(KimageDetector):
    def __init__(self,model=None, threshold=0.60):
        super().__init__()
        if not model:
            model=model_path
        self.threshold=threshold
        self.detector = UltraLightFaceDetecion(model_path,
                        input_size=(320, 240), conf_threshold=0.6,
                        center_variance=0.1, size_variance=0.2,
                        nms_max_output_size=200, nms_iou_threshold=0.5)
        self.inObject = 0
        self.outObject = 0
        self.total = 0
        self.inside = 0
        

     
    
    def detect(self, image, threshold=None, x1=193, y1=717, x2=717, y2=193, mode= "stopped"):
        if not threshold:
            threshold=self.threshold
        res=[]
        
        trackers = []
        rects=[]
        orig_resolution=(image.shape[1], image.shape[0])
        scale=(1,1)
        contourROC  = np.array([(x1,y1),
                        (x2,y1),
                        (x2,y2),
                        (x1,y2)], np.int64)
        
        # Create a TensorImage object from the RGB image.
        if mode == "rein":
                del self.total
                self.total=0
                #self.inObject = 0
                #self.outObject = 0
                tracked_objects.clear()
        
        if mode == "started":
            boxes, scores = self.detector.inference(image)
            #print(len(boxes),len(scores))
            for result in range(len(boxes.astype(int))):
                    if scores[result] < threshold:
                            continue
                    x1=int(boxes[result][0])
                    y1=int(boxes[result][1])
                    x2=int(boxes[result][2])
                    y2=int(boxes[result][3])
                    
                    bbox= x1,y1,x2,y2
                    rects.append(bbox)
                    #cv2.rectangle(image,(int(x1), int(y1)),(int(x2), int(y2)),(0,0,255),2)
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
                #text = "ID: {}".format(objectId)
                #cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                
                # Counting 
                if objectId not in tracked_objects.keys():
                    tracked_objects[objectId] = ( bbox, position)
                
                    if position == 1:
                        self.total += 1
                        

                else:
                # s'il existe,  on le recupere, pour connaitre sa position dans l'ancienne frame
                    (bbox, position_ini)   = tracked_objects[objectId]     
                
                    if position != position_ini:
                        
                        if position == 1:
                            self.total += 1
                            t= datetime.now().strftime(" %Y-%m-%d-%H-%M")
                            filename="image_{}_{}_{}_{}_{}_face_event_{} ".format(bbox[0],bbox[1],bbox[2],bbox[3],str(cameraID),str(t))
                            cv2.imwrite('/home/pi/vizy/apps/face_detect/output/'+ filename+'.jpg', image)
                        if position == -1:
                            self.total -= 1
                            
                        tracked_objects[objectId] = ( bbox, position)
                
                
        #self.total = self.inObject - self.outObject +self.inside
        
        #cv2.putText(image, 'out :' + str(self.outObject) + ' in : ' + str(self.inObject) 
        #        + ' inside : ' + str(self.total) , (150, 400), cv2.FONT_HERSHEY_PLAIN,   2, (0, 0, 255), 2)
              
               
                
        return res,self.total

