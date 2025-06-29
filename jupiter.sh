#!/bin/bash

source activate py312
pip install tqdm opencv-python-headless albumentations imutils scikit-learn easydict papermill --upgrade
papermill notebooks/analyze_breaks.ipynb notebooks/output.ipynb