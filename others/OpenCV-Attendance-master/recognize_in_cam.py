import cv2
import sys
import threading
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as npc

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, # model selection
    min_detection_confidence=0.5 # confidence threshold
)

def detect_faces_mediapipe(img):
    n,m,_ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = mp_face.process(rgb_img)
    if results.detections is None:
        return img, []
    cropped_faces_gray = []
    for detection in results.detections:
        location = detection.location_data
        rbb = location.relative_bounding_box # relative bounding box
        if rbb is None:
            continue
        startpoint = npc(rbb.xmin,rbb.ymin,n,m)
        endpoint = npc(rbb.xmin+rbb.width,rbb.ymin+rbb.height,n,m)
        if (startpoint is None) or (endpoint is None):
            continue
        x1,y1 = npc(rbb.xmin,rbb.ymin,n,m)
        x1,y1 = startpoint
        x2,y2 = endpoint
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cropped_faces_gray.append(gray[y1:y2,x1:x2])
    return img, cropped_faces_gray

def detect_faces(f_cascade, img, scaleFactor = 1.2):  
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          
   faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
   cropped_faces_gray = []
   for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
      cropped_faces_gray.append(gray[y:y+w,x:x+h])
   return img, cropped_faces_gray

def predict(f_recognizer, test_img, subjects):
    img = test_img.copy()
    label, confidence = f_recognizer.predict(img)
    label_text = subjects[label]
    if confidence < 50:
        print(confidence)
        print(label_text, "Present")

flag = True
def check_input():
    global flag
    key=input()
    while (key != "q"):
        key=input()
    flag=False

def main_func():
    global flag

    face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("Trained_Model.yml")

    subjects = ["Salman Khan","Shahrukh Khan","Akshay Kumar","Sandeep","Harsha"]

    video_capture = cv2.VideoCapture(0)
    
    while flag:
        _, img = video_capture.read()
        cv2.imshow("Recognize", img)
        cv2.waitKey(100)
        # img2,faces = detect_faces(face_cascade,img.copy(),1.2)
        img2, faces = detect_faces_mediapipe(img.copy())
        for face in faces:
            face = cv2.equalizeHist(face)
            predict(face_recognizer, face, subjects)
    
    video_capture.release()
    cv2.destroyAllWindows()

n=threading.Thread(target=main_func)
i=threading.Thread(target=check_input)
n.start()
i.start()