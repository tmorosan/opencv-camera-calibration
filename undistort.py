import json

import cv2
import numpy as np

if __name__ == "__main__":
    mtx = np.load("./calibration/mtx", allow_pickle=True)
    dist = np.load("./calibration/dist", allow_pickle=True)

    with open("./calibration/image.json") as input:
        info = json.load(input)
        h = info.get("h")
        w = info.get("w")

    img = cv2.imread("./data/carlos1/carlos_L.jpg")

    # Refining the camera matrix using parameters obtained by calibration
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    print(roi[2], roi[3])
    # Method 1 to undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Method 2 to undistort the image
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (roi[2], roi[3]), 5)

    dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Displaying the undistorted image
    cv2.imshow("original", img)
    cv2.imshow("undistorted image", dst)
    cv2.imshow("undistorted 2", dst2)
    cv2.waitKey(0)
