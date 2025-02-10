#!/bin/bash

source activate py312
pip install tqdm opencv-python-headless albumentations imutils scikit-learn easydict --upgrade
python main.py