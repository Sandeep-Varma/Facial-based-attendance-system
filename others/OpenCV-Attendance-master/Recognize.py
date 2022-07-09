import cv2
import threading

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
    # if confidence < 50:
    label_text = subjects[label]
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

    face_cascade = cv2.CascadeClassifier("ffd.xml")

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trained_model.yml")

    subjects = ["Salman Khan","Shahrukh Khan","Akshay Kumar","Sandeep"]

    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('http://192.168.1.146:4747/video?640x480')
    # video_capture.set(3,1920)
    # video_capture.set(4,1080)
    
    while flag:
        for i in range(10):
            _, img = video_capture.read()
        cv2.imshow("Recognize", img)
        cv2.waitKey(100)
        img2,faces = detect_faces(face_cascade,img.copy(),1.2)
        for face in faces:
            face = cv2.equalizeHist(face)
            predict(face_recognizer, face, subjects)
    
    video_capture.release()
    cv2.destroyAllWindows()

n=threading.Thread(target=main_func)
i=threading.Thread(target=check_input)
n.start()
i.start()