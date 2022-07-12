# Face Recognition Attendance system

	Authors:
		Sandeep Varma Dendukuri
		Harshavardhan Dwarapudi
		Vinay Kumar Cheerla
		Saivardhan Annam

# Directions to use:

Place a set of images for each student in a folder. The program takes these as input and detects faces in them. It eliminates those images with a low resolution of the face, crops the rest, and modifies all images to the same resolution height. It also generates a list of students in a CSV file. Now it trains the cropped images to generate a trained model. With this trained model, it recognizes faces (can recognize multiple faces at a time) from a real-time webcam feed and marks attendance with good accuracy. This program supports three methods of face detection - opencv, Mediapipe, and face_recognition each having its own advantages. Any of these options can be selected by us. Also, this program provides many customizations to the user that he/her can change according to their convenience. All the customizations can be done in the parameters file. To run the program, one can see the commands in run.sh file or directly run it.