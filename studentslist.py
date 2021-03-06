import os,csv
from parameters import input_dataset_path
from parameters import students_list_path

def update_students_list():
	students_list = []
	if not os.path.isdir(input_dataset_path):
		print(input_dataset_path,"directory not found.")
		return
	if len(os.listdir(input_dataset_path)) == 0:
		print(input_dataset_path,"directory is empty.")
		return
	for name in os.listdir(input_dataset_path):
		if os.path.isdir(input_dataset_path+name):
			students_list.append([name])
		else:
			print("Files are not to be placed in this directory:",input_dataset_path,'\n',"Put only directories of students images")
	students_list.sort()
	with open(students_list_path,'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerows(students_list)
	print("Students list successfully updated")
