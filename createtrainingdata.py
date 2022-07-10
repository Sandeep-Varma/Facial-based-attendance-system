import os,cv2
import pandas as pd
from paramenters import *
from studentslist import update_students_list
from facelocate import face_detect_n_locate

def create_training_data():
	update_students_list()
	students = pd.read_csv(students_list_path,header=None).to_numpy()
	for i in range(len(students)):
		j = 0
		for image in os.listdir(input_dataset_path+students[i][0]):
			img = cv2.imread(input_dataset_path+students[i][0]+"/"+image)
			faces = face_detect_n_locate(img)
			if len(faces) == 0:
				print("Could not detect complete face from",input_dataset_path+students[i][0]+"/"+image)
				continue
			if len(faces) > 1:
				print("Detected multiple faces from",input_dataset_path+students[i][0]+"/"+image)
				continue
			(x,y,w,h) = faces[0]
			cv2.imwrite(training_dataset_path+str(i)+"_"+str(j),img[y:y+h,x:x+w])
			j=j+1
	print("Training data created successfully")

create_training_data()