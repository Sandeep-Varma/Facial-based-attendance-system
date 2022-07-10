import cv2
import threading
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as npc

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, # model selection
    min_detection_confidence=0.5 # confidence threshold
)

def detect_faces_mediapipe(img):
    m,n,_ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = mp_face.process(rgb_img)
    if results.detections is None:
        return [], []
    cropped_faces_gray = []
    r = []
    for detection in results.detections:
        location = detection.location_data
        rbb = location.relative_bounding_box # relative bounding box
        if rbb is None:
            continue
        startpoint = npc(rbb.xmin,rbb.ymin,n,m)
        endpoint = npc(rbb.xmin+rbb.width,rbb.ymin+rbb.height,n,m)
        if (startpoint is None) or (endpoint is None):
            continue
        (x1,y1) = startpoint
        (x2,y2) = endpoint
        cropped_faces_gray.append(gray[y1:y2,x1:x2])
        r.append([x1, y1, x2, y2])
    return cropped_faces_gray, r

def predict(f_recognizer, test_img, subjects):
    img = test_img.copy()
    label, confidence = f_recognizer.predict(img)
    if label < 0:
        return "Unknown",-1
    # if confidence < 50:
    label_text = subjects[label]
    return label_text, confidence

flag = True
def check_input():
    global flag
    key=input()
    while (key != "q"):
        key=input()
    flag=False

def main_func():
    global flag

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trained_model.yml")

    subjects = ["Salman Khan","Shahrukh Khan","Akshay Kumar","Sandeep"]

    video_capture = cv2.VideoCapture(0)
    
    while flag:
        _, img = video_capture.read()
        faces, r = detect_faces_mediapipe(img)
        for i in range(len(r)):
            faces[i] = cv2.equalizeHist(faces[i])
            label_text, confidence = predict(face_recognizer, faces[i], subjects)
            (x1,y1,x2,y2) = r[i]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if confidence>=0:
                # print(label_text,confidence)
                cv2.putText(
                    img,
                    label_text+" "+str(int(confidence)),
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    img,
                    "Unknown",
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
        cv2.imshow("Recognize", img)
        cv2.waitKey(100)
    
    video_capture.release()
    cv2.destroyAllWindows()

n=threading.Thread(target=main_func)
i=threading.Thread(target=check_input)
i.start()
n.start()