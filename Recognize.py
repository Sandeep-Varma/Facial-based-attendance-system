import os
import cv2
import threading
import face_recognition as fr

def detect_faces(f_cascade, img, scaleFactor = 1.2):  
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          
   faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
   cropped_faces_gray = []
   r = []
   for (x, y, w, h) in faces:
      cropped_faces_gray.append(gray[y:y+h,x:x+w])
      r.append([x, y, w, h])
   return cropped_faces_gray, r

def predict(f_recognizer, img, subjects):
    label, confidence = f_recognizer.predict(img)
    if label < 0 or confidence >= 50:
        return "Unknown",-1,-1
    label_text = subjects[label]
    return label_text, confidence, label

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

    subjects = ["Salman Khan","Shahrukh Khan","Akshay Kumar","Sandeep","Harsha"]

    video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture('http://192.168.1.146:4747/video?640x480')
    # video_capture.set(3,1920)
    # video_capture.set(4,1080)
    prev_label = -1
    count = 0;
    while flag:
        for i in range(10):
            _, img = video_capture.read()
        faces, r = detect_faces(face_cascade,img,1.2)
        for i in range(len(r)):
            faces[i] = cv2.equalizeHist(faces[i])
            label_text, confidence, label = predict(face_recognizer, faces[i], subjects)
            (x,y,w,h) = r[i]
            if prev_label == label:
                count=count+1
            else:
                prev_label = label
                count=1
            if count == 20 and prev_label>=0:
                count = 0
                i1 = fr.face_encodings(cv2.cvtColor(img[y:y+h,x:x+w],cv2.COLOR_BGR2RGB))[0]
                i = []
                path = "training-data/s"+str(label)
                for f in os.listdir(path):
                    i2 = fr.face_encodings(cv2.cvtColor(fr.load_image_file(path+"/"+f),cv2.COLOR_BGR2RGB))
                    if len(i2) == 1:
                        i.append(i2[0])
                match_count = 0
                for comp in fr.compare_faces(i,i1):
                    if comp:
                        match_count = match_count + 1
                if match_count > 6:
                    print(label_text)
                else:
                    print("mismatch")
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if confidence>=0:
                # print(label_text,confidence)
                cv2.putText(
                    img,
                    label_text+" "+str(int(confidence)),
                    (x, y - 4),
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
                    (x, y - 4),
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
n.start()
i.start()