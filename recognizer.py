import cv2,threading,csv
import pandas as pd
from datetime import datetime
from parameters import *
from facelocate import face_detect_n_locate

flag = True
def check_input():
    global flag
    key=input()
    while (key != "quit"):
        key=input()
    flag=False

def predict(f_recognizer, img):
    label, confidence = f_recognizer.predict(img)
    if label < 0 or confidence > 50:
        return -1,-1
    return label, confidence

def main_func():
	global flag

	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.read(trained_model_path)
	students = pd.read_csv(students_list_path,header=None).values.tolist()
	count = 0
	prev_label = -1
	video_capture = cv2.VideoCapture(video_capture_input)
	if not capture_default_resolution:
		video_capture.set(3,non_default_resolution_width)
		video_capture.set(4,non_default_resolution_height)
	while flag:
		_, img = video_capture.read()
		if img is None:
			print("Could not capture image from camera")
			continue
		r = face_detect_n_locate(img)
		for face_pos in r:
			(x,y,w,h) = face_pos
			face = cv2.cvtColor(img[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
			face = cv2.equalizeHist(face)
			label, confidence = predict(face_recognizer,face)
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			if label >=0 and confidence>=0:
				if prev_label == label:
					count=count+1
				else:
					prev_label = label
					count = 1
				if count == 10:
					print(students[label][0],datetime.today().strftime(time_format))
					if students[label][-1] != datetime.today().strftime(time_format):
						students[label].append(datetime.today().strftime(time_format))
					count = 0
				cv2.putText(img,students[label][0]+" "+str(int(confidence)),
					(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),1,cv2.LINE_AA,)
			else:
				cv2.putText(img,"Unknown",
					(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 0, 0),1,cv2.LINE_AA,)
		cv2.imshow("Facial Recognizer", img)
		cv2.waitKey(100)

	video_capture.release()
	cv2.destroyAllWindows()
	for i in range(len(students)):
		if (len(students[i])>1):
			students[i].insert(1,"PRESENT")
		else:
			students[i].append("ABSENT")
	with open(attendance_file_path_prefix+datetime.today().strftime(date_format)+".csv",'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerows(students)

n=threading.Thread(target=main_func)
i=threading.Thread(target=check_input)
i.start()
n.start()