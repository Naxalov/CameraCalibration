import numpy as np
import cv2
import os
import time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# squear

DIR_INPUT = 'samples/photo'

# input image in order calibration
PATH = os.path.join(os.getcwd(), DIR_INPUT)

images = os.listdir(PATH)
img = cv2.imread(os.path.join(PATH, images[0]))
# get shape of image
W, H, _ = img.shape
# W, H = 320 * 2, 240 * 2
for fname in images:
    img = cv2.imread(os.path.join(PATH, fname))
    img = cv2.resize(img, (H // 2, W // 2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 7), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        # time.sleep(3)

cv2.destroyAllWindows()
print(img.shape)

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(dist)
print(mtx)

#  320 * 2, 240 * 2

# dist = [[-0.02357899  0.32398194 -0.01049933  0.00349942 -0.2899601 ]]
# mtx =[[551.35925517   0.         248.33988363]
#  [  0.         551.84937767 309.15983624]
#  [  0.           0.           1.        ]]


# 0.35788135489015593
