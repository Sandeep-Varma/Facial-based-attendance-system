import os,cv2
import numpy as np
from paramenters import *

def train_model():
	if not os.path.isdir(training_dataset_path):
		print("Create training data first to train model.\nRun createtrainingdata.py to do that.")
		return
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	faces = []
	labels = []
	for img in os.listdir(training_dataset_path):
		gray_img = cv2.cvtColor(cv2.imread(training_dataset_path+img),cv2.COLOR_BGR2GRAY)
		faces.append(gray_img)
		labels.append(int(img.split('_')[0]))
	face_recognizer.train(faces,np.array(labels))
	face_recognizer.write(trained_model_path)
	print("Trained model successfully")

train_model()