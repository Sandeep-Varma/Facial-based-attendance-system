import cv2,os
import numpy as np
import pandas as pd
import face_recognition

img_bgr = face_recognition.load_image_file('Hello4.jpeg')
img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

#----------Finding face Location for drawing bounding boxes-------
faces = face_recognition.face_locations(img, model="hog")

#-------------------Drawing the Rectangle-------------------------
#for x in faces:
    #cv2.rectangle(img, (x[3], x[0]),(x[1], x[2]), (255,0,255), 2)


#cv2.imshow('image', img)

students=pd.read_csv('students.csv',header=None).to_numpy()
print(students)
#train_encodings = face_recognition.face_encodings(img)


# lets test an image
test_encodings = face_recognition.face_encodings(img)
#print(len(test_encoding))
for y in range(len(test_encodings)):
    print("Hello")
    for [s] in students:
        images=os.listdir(s)
        d=[]
        match=True
        for i in images:
            train_encoding = face_recognition.face_encodings(cv2.cvtColor(face_recognition.load_image_file(s+'/'+i),cv2.COLOR_BGR2RGB))
            #print(face_recognition.face_distance([train_encoding],y))
            print(face_recognition.compare_faces([train_encoding[0]],test_encodings[y]))
            match=match and face_recognition.compare_faces([train_encoding[0]],test_encodings[y])[0]
            # if(face_recognition.compare_faces([train_encoding[0]],test_encodings[y])[0]):
            #     cv2.rectangle(img, (faces[y][3], faces[y][0]),(faces[y][1], faces[y][2]), (255,0,255), 2)
        if match:
            cv2.rectangle(img, (faces[y][3], faces[y][0]),(faces[y][1], faces[y][2]), (255,0,255), 2)
            break

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()