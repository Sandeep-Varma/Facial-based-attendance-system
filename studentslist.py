import os
import csv
from paramenters import input_dataset_path
from paramenters import students_list_path

def update_students_list():
	students_list = []
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
