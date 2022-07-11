attendance_file_path_prefix = "./attendance-"
date_format = "%Y-%m-%d"
time_format = "%I:%M %p"
# attendance file name = attendance_file_path_prefix+date_format+".csv"

students_list_path = "./students.csv"
input_dataset_path = "./Input-Dataset/"
training_dataset_path = "./Training-Data/"
trained_model_path = "./students-model.yml"

file_format = ".jpg"
# This is just the file format used for training.
# We can give any type of file in dataset. The program converts it to .jpg

opencv_face_detector = "./opencv-face-detectors/haarcascade_frontalface_default.xml"
# can use any preferrable face detector, this works only when the face_detection_method is 0

cropped_face_res = 200
# increasing this increases accuracy but program load increases on cpu and training time increases

accuracy = 10
# increasing this increases accuracy and decreases speed

face_detection_method = ["opencv","mediapipe","face_recognition"][0]
# opencv method gives more accuracy
# mediapipe method detects faces well

video_capture_input = 0
# 0 for default webcam input, or any other if more cameras installed
# Live video-feed link can also be put (example: droidcam)

capture_default_resolution = True
# When this is True, below resolution is not applicable
# By setting this to False, live capture takes place with below resolution

non_default_resolution_width = 640
non_default_resolution_height = 480
