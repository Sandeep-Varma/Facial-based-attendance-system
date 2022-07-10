#!/bin/bash

# before running this, make sure you place input dataset in "Input-Dataset/"

python3.8 createtrainingdata.py
python3.8 trainmodel.py
python3.8 recognizer.py
