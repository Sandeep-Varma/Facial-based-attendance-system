#!/bin/bash

python3.8 createtrainingdata.py
python3.8 trainmodel.py
python3.8 recognizer.py
