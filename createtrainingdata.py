import os,cv2,shutil
import pandas as pd
from parameters import *
from studentslist import update_students_list
from facelocate import face_detect_n_locate

def create_training_data():
	update_students_list()
	if os.path.isdir(training_dataset_path):
		shutil.rmtree(training_dataset_path)
	os.mkdir(training_dataset_path)
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
			img2 = img[y:y+h,x:x+w]
			if h < cropped_face_res:
				print("Low image resolution or face not clear:",input_dataset_path+students[i][0]+"/"+image)
				continue
			w = (int)(w*cropped_face_res/h)
			h = cropped_face_res
			img2 = cv2.resize(img2,(w,h),interpolation=cv2.INTER_AREA)
			if cv2.imwrite(training_dataset_path+str(i)+"_"+str(j)+file_format,img2):
				print(training_dataset_path+str(i)+"_"+str(j)+file_format)
			else:
				print("Failed saving image:",input_dataset_path+students[i][0]+"/"+image)
			j=j+1
	print("Training data created successfully")

create_training_data()