from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import FaceCollection
from compreface.collections.face_collections import Subjects
from os.path import exists
from os import mkdir, listdir
nom_subject=""
DOMAIN: str = 'http://localhost'
PORT: str = '8000'
API_KEY: str = '2c49818b-7003-4b8b-8b0e-ab6cb97051fd'

compre_face: CompreFace = CompreFace(DOMAIN, PORT)

recognition: RecognitionService = compre_face.init_face_recognition(API_KEY)

face_collection: FaceCollection = recognition.get_face_collection()

subjects: Subjects = recognition.get_subjects()


def runTC(imagePath, Subject):
    try:
        face_collection.add(image_path=imagePath, subject=Subject)
        print(True)
    except Exception:
        print(False)

