import cv2
from parameters import face_detection_method

def face_detect_n_locate(img):
	if face_detection_method == "opencv":
		from parameters import opencv_face_detector
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		face_cascade = cv2.CascadeClassifier(opencv_face_detector)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))
		return faces
	elif face_detection_method == "mediapipe":
		import mediapipe as mp
		from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as npc
		model_selection=1;min_detection_confidence=0.5
		mp_face = mp.solutions.face_detection.FaceDetection(model_selection,min_detection_confidence)
		m,n,_ = img.shape
		rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = mp_face.process(rgb_img)
		if results.detections is None:
			return []
		faces = []
		for detection in results.detections:
			location = detection.location_data
			rbb = location.relative_bounding_box # relative bounding box
			if rbb is None:
				continue
			startpoint = npc(rbb.xmin,rbb.ymin,n,m)
			endpoint = npc(rbb.xmin+rbb.width,rbb.ymin+rbb.height,n,m)
			if (startpoint is None) or (endpoint is None):
				continue
			x1,y1 = startpoint
			x2,y2 = endpoint
			faces.append([x1,y1,x2-x1,y2-y1])
		return faces
	elif face_detection_method == "face_recognition":
		import face_recognition as fr
		faces1 =  fr.face_locations(img)
		faces = []
		for x in faces1:
			faces.append([x[3],x[0],x[1]-x[3],x[2]-x[0]])
		return faces
	else:
		print("Face Detection Method",face_detection_method,"Not Available")
		return None
