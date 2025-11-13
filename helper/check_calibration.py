# Source - https://stackoverflow.com/a
# Posted by N.S
# Retrieved 2025-11-06, License - CC BY-SA 4.0

import numpy as np
from numpy import load
import cv2

data = load('../calibration.npz')
lst = data.files
for item in lst:
    print(item)
    if  item == "camera_matrix" or item == "rvecs" or item == "tvecs":
        print(data[item].shape)
    if item == "camera_matrix":
        print(data[item])
    
    if item == "rvecs":
        rvecs = data[item]
    elif item == "tvecs":
        tvecs = data[item]
    elif item == "camera_matrix":
        intrinsic_camera_matrix = data[item]


