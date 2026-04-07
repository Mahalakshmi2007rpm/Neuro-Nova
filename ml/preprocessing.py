import cv2
import numpy as np

def preprocess_input_image(img_path, target_size=(224, 224)):
    """Read and normalize image"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img