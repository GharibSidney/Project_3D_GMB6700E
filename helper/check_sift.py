import numpy as np
import cv2

img = cv2.imread('../image_calibration/IMG_5724.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
key_points = sift.detect(gray, None)

img=cv2.drawKeypoints(gray,key_points,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift_keypoints.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()