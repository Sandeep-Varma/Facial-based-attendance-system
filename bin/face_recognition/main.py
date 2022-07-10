import cv2,os
import numpy as np
import pandas as pd
import face_recognition as fr

img_bgr = fr.load_image_file('test_image.jpeg')
img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

#----------Finding face Location for drawing bounding boxes-------
faces = fr.face_locations(img)

#-------------------Drawing the Rectangle-------------------------
#for x in faces:
    #cv2.rectangle(img, (x[3], x[0]),(x[1], x[2]), (255,0,255), 2)


#cv2.imshow('image', img)

students=pd.read_csv('dataset/students.csv',header=None).to_numpy()
print(students)
#train_encodings = fr.face_encodings(img)


# lets test an image
test_encodings = fr.face_encodings(img)
#print(len(test_encoding))
for y in range(len(test_encodings)):
    print("Hello")
    for [s] in students:
        images=os.listdir("dataset/"+s)
        d=[]
        match=True
        for i in images:
            train_encoding = fr.face_encodings(cv2.cvtColor(fr.load_image_file("dataset/"+s+'/'+i),cv2.COLOR_BGR2RGB))
            #print(fr.face_distance([train_encoding],y))
            print(fr.compare_faces([train_encoding[0]],test_encodings[y]))
            match=match and fr.compare_faces([train_encoding[0]],test_encodings[y])[0]
            # if(fr.compare_faces([train_encoding[0]],test_encodings[y])[0]):
            #     cv2.rectangle(img, (faces[y][3], faces[y][0]),(faces[y][1], faces[y][2]), (255,0,255), 2)
        if match:
            print(s)
            cv2.rectangle(img, (faces[y][3], faces[y][0]),(faces[y][1], faces[y][2]), (255,0,255), 2)
            break

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()