import kritter
from kritter import KimageDetector
from TFLiteFaceDetector import UltraLightFaceDetecion
from pyimagesearch.centroidtracker import CentroidTracker
import cv2
import numpy as np 
import time
from os.path import dirname, join
from datetime import datetime
import imagezmq
import socket


cameraID=socket.gethostname()
#sender=imagezmq.ImageSender(connect_to="tcp://172.18.4.121:5555")
filepath = "/home/pi/face/version-slim-320_without_postprocessing.tflite"
tracked_objects = {}
trackerC = CentroidTracker(40,60)
class TFLiteDetector(KimageDetector):
    def __init__(self,model=None, threshold=0.75):
        super().__init__()
        if not model:
            model=filepath
        self.threshold=threshold
        self.total=0
        self.fd = UltraLightFaceDetecion(filepath,
                            input_size=(320, 240), conf_threshold=0.6,
                            center_variance=0.1, size_variance=0.2,
                            nms_max_output_size=200, nms_iou_threshold=0.5)
        

        
    def detect(self, image, threshold=None, x1=193, y1=717, x2=717, y2=193, mode= "stopped"):
        if not threshold:
            threshold=self.threshold
        res=[]
        object_current_frame = []
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
                tracked_objects.clear()
        #input_tensor = self.detector.process(image)
        boxes, scores = self.fd.inference(image)
        if mode == "started":
            for result in boxes.astype(int):
                bbox=result[0], result[1], result[2], result[3]
                centroide = int((bbox[0]+ bbox[0]+bbox[2]) / 2), int((bbox[1]+ bbox[1]+bbox[3])/2)
                #cv2.rectangle(frame, (result[0], result[1]), (result[2], result[3]), color, 2)
                position = cv2.pointPolygonTest( contourROC , centroide, False)
            
                object_current_frame.append((bbox , centroide , position , 0 , False))
                
                max_score=max(scores)
                #box=[int((bboxC.xmin * iw)*scale[0]), int((bboxC.ymin * ih)*scale[1]),int((bboxC.width * iw)*scale[0]),int((bboxC.height * ih)*scale[1])]
                box=[int(result[0]*scale[0]), int(result[1]*scale[1]),int(result[2]*scale[0]),int(result[3]*scale[1])]

                box=[0 if i<0 else i for i in box]
                obj = {"box": box, "class": "person","score": scores[0], "index": 0} 
                res.append(obj)

                
        if len(object_current_frame) != 0:
            rec=[]
            for object in object_current_frame:
                rec.append((object[0][0], object[0][1], object[0][0] + object[0][2] , object[0][1]+object[0][3]))
                
            trackerC.update(rec)

        
            # Affectation de l'ID
            for object in (object_current_frame):
                for key in trackerC.objects.keys():
                    if trackerC.objects[key][0] == object[1][0] and trackerC.objects[key][1] == object[1][1]:
                        bbx , center, position , id_object , croped = object_current_frame[object_current_frame.index(object)] #object[3]
                        object_current_frame[object_current_frame.index(object)] = (bbx , center, position , key , croped)
    

        
        
            for object in (object_current_frame): 
                bbx , center, position , id_object , croped = object
                
                # comparaison avec l'ancien frame
                if position !=-1:
                
                    if id_object not in tracked_objects.keys():
                        tracked_objects[id_object] = (center, position, croped)
                        #print('croping object %s' % id_object)
                        #crop_img = img[bbx[1]-cadre:bbx[1]+bbx[3]+cadre, bbx[0]-cadre:bbx[0]+bbx[2]+cadre]
                        t= datetime.now().strftime(" %Y-%m-%d-%H-%M")
                        filename="image_{}_{}_{}_{}_{}_face_event_{} ".format(bbx[0],bbx[1],bbx[2],bbx[3],str(cameraID),str(t))
                        cv2.imwrite('/home/pi/vizy/apps/face_detect/output/'+ filename+'.jpg', image)
                        self.total=self.total+1
                        #sender.send_image(filename+'.jpg',image)
                elif position == -1 and id_object in tracked_objects.keys():
                        self.total=self.total - 1
                        del tracked_objects[id_object]
                
        return res,self.total
    

