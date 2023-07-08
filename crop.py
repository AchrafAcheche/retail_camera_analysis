
import numpy as np
import cv2
from TFLiteFaceDetector import UltraLightFaceDetecion
import os

model_path="version-RFB-320_without_postprocessing.tflite"


model=model_path
threshold=0.6
detector = UltraLightFaceDetecion(model_path,
           input_size=(320, 240), conf_threshold=threshold,
           center_variance=0.1, size_variance=0.2,
           nms_max_output_size=200, nms_iou_threshold=0.5)
dir_path: str = 'old'
list_file=os.listdir(dir_path)
for name in list_file:
    img = cv2.imread('output/imo.jpg')
    boxes, scores = detector.inference(img)
    for result in range(len(boxes.astype(int))):
        if scores[result] < 0.6:
            continue

        x1=int(boxes[result][0])
        y1=int(boxes[result][1])
        x2=int(boxes[result][2])+10
        y2=int(boxes[result][3])+10

        bbox= x1,y1,x2,y2
        face_img = img[y1:y2, x1:x2]


        filename, extension = os.path.splitext(os.path.basename(name))
        cv2.imwrite('new/' + filename + extension, face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
		
                    
