import cv2
import time
from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import FaceCollection
from compreface.collections.face_collections import Subjects

api_key = '2c49818b-7003-4b8b-8b0e-ab6cb97051fd'
host = 'http://localhost'
port = '8000'
compre_face = CompreFace(host, port, {
    "limit": 0,
    "det_prob_threshold": 0.8,
    "prediction_count": 1,
    "face_plugins": "gender,age",
    "status": False
})
recognition: RecognitionService = compre_face.init_face_recognition(api_key)

face_collection: FaceCollection = recognition.get_face_collection()

subjects: Subjects = recognition.get_subjects()

def delete_em(nom):
    subjects.delete(nom)