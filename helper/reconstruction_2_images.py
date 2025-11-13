import numpy as np
from numpy import load
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = load('calibration.npz')
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
    elif item == "dist_coeffs":
        dist_coeffs = data[item]
# camera_matrix: (3,3) from calibration
# dist_coeffs: (k,) from calibration
# pts1, pts2: Nx2 matched keypoints between image1 and image2
img1 = cv2.imread('image_calibration/IMG_5724.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('image_calibration/IMG_5725.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Convert keypoints to Nx2 numpy arrays
pts1 = np.array([kp.pt for kp in kp1], dtype=np.float32)
pts2 = np.array([kp.pt for kp in kp2], dtype=np.float32)

# Example: if you want to match them first (recommended)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Take matched points
pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

# undistort matched points
pts1_ud = cv2.undistortPoints(pts1.reshape(-1, 1, 2), intrinsic_camera_matrix, dist_coeffs)
pts2_ud = cv2.undistortPoints(pts2.reshape(-1, 1, 2), intrinsic_camera_matrix, dist_coeffs)
# Essential matrix (because intrinsics known)
E, mask = cv2.findEssentialMat(pts1_ud, pts2_ud, intrinsic_camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# recover relative pose
_, R, t, mask_pose = cv2.recoverPose(E, pts1_ud, pts2_ud, intrinsic_camera_matrix)

# projection matrices for triangulation (camera1 = I|0)
P1 = np.hstack((np.eye(3), np.zeros((3,1))))
P2 = np.hstack((R, t))

# convert to 3x4 with intrinsics
P1 = intrinsic_camera_matrix @ P1
P2 = intrinsic_camera_matrix @ P2

# triangulate
pts4d_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # shape (4, N)
pts3d = (pts4d_h[:3] / pts4d_h[3]).T  # (N,3)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], s=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Reconstruction (Sparse Point Cloud)')
plt.show()