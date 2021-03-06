import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as npc

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, # model selection
    min_detection_confidence=0.5 # confidence threshold
)

def detect_face(img):
    n,m,_ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb_img)
    if (results.detections is None) or (len(results.detections)>1):
        return None
    detection=results.detections[0]
    location = detection.location_data
    rbb = location.relative_bounding_box # relative bounding box
    if rbb is None:
        return None
    startpoint = npc(rbb.xmin,rbb.ymin,n,m)
    endpoint = npc(rbb.xmin+rbb.width,rbb.ymin+rbb.height,n,m)
    if (startpoint is None) or (endpoint is None):
        return None
    x1,y1 = startpoint
    x2,y2 = endpoint
    return rgb_img[y1:y2, x1:x2]


def train_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        print(dir_name)
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            img = detect_face(image)
            if img is None:
                print("Could not detect complete face or detected multiple faces from",image_path)
                continue
            face = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            if face is None:
                print("Bad image:",image_path)
                continue
            faces.append(face)
            labels.append(label)
    return faces, labels

faces, labels = train_data("training-data")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.write("trained_model.yml")
print("Trained model successfully")