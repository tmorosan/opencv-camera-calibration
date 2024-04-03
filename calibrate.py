#!/usr/bin/env python3
import json

import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('./data/calibration_aaa/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lowerb = np.array([0, 0, 196])
    upperb = np.array([226, 43, 255])
    frame_threshold = cv2.inRange(hsv, lowerb, upperb)

    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(frame_threshold, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, frame_threshold)

    res = np.uint8(res)
    ret, corners = cv2.findChessboardCorners(res, (6, 9),
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_FAST_CHECK +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        print(corners)
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    else:
        print(f'No checkboard found in: {fname}')
    # cv2.imshow('img', img)
    # cv2.imshow("thresh", res)
    # cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


mtx.dump("./calibration/mtx")
dist.dump("./calibration/dist")

with open("./calibration/image.json", "w") as info:
    json.dump({"h": h, "w": w}, info)
