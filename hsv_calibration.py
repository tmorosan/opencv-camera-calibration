#!/usr/bin/env python3

import cv2
import numpy as np

WINDOWW_DETECTION_NAME = "detection"
WINDOWW_CAPTURE_NAME = "capture"
MAX_VALUE = 255

if __name__ == "__main__":
    image = cv2.imread("./data/calibration_aaa/IMG_6103.jpg")

    cv2.namedWindow(WINDOWW_DETECTION_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOWW_CAPTURE_NAME, cv2.WINDOW_NORMAL)

    # v2
    # Lowerb: (0, 0, 97)
    # Upperb: (255, 255, 255)

    low_H = 0
    high_H = 255

    low_S = 0
    high_S = 255

    # low_V = 97
    low_V = 72
    high_V = 255


    def on_low_h_thresh_trackbar(val):
        global low_H
        low_H = val
        # cv2.setTrackbarPos("low_H_name", WINDOWW_DETECTION_NAME, low_H)


    def on_high_h_thresh_trackbar(val):
        global high_H
        high_H = val
        # cv2.setTrackbarPos("high_H_name", WINDOWW_DETECTION_NAME, high_H)


    def on_low_S_thresh_trackbar(val):
        global low_S
        low_S = val


    def on_high_S_thresh_trackbar(val):
        global high_S
        high_S = val


    def on_low_V_thresh_trackbar(val):
        global low_V
        low_V = val


    def on_high_V_thresh_trackbar(val):
        global high_V
        high_V = val


    def nothing(x):
        pass


    while True:
        cv2.createTrackbar("low_H_name", WINDOWW_DETECTION_NAME, low_H, MAX_VALUE, on_low_h_thresh_trackbar)
        cv2.createTrackbar("high_H_name", WINDOWW_DETECTION_NAME, high_H, MAX_VALUE, on_high_h_thresh_trackbar)
        cv2.createTrackbar("low_S_name", WINDOWW_DETECTION_NAME, low_S, MAX_VALUE, on_low_S_thresh_trackbar)
        cv2.createTrackbar("high_S_name", WINDOWW_DETECTION_NAME, high_S, MAX_VALUE, on_high_S_thresh_trackbar)
        cv2.createTrackbar("low_V_name", WINDOWW_DETECTION_NAME, low_V, MAX_VALUE, on_low_V_thresh_trackbar)
        cv2.createTrackbar("high_V_name", WINDOWW_DETECTION_NAME, high_V, MAX_VALUE, on_high_V_thresh_trackbar)

        frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        print("Lowerb: ", (low_H, low_S, low_V))
        print("Upperb: ", (high_H, high_S, high_V))
        frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

        krn = cv2.getStructuringElement(cv2.MORPH_DILATE, (50, 50))
        # krn = np.ones((9, 9), np.uint8)
        dlt = cv2.dilate(frame_threshold, krn, iterations=5)
        res = 255 - cv2.bitwise_and(dlt, frame_threshold)

        cv2.imshow(WINDOWW_CAPTURE_NAME, image)
        cv2.imshow(WINDOWW_DETECTION_NAME, frame_threshold)
        # cv2.imshow("mask", res)

        res = np.uint8(res)
        ret, corners = cv2.findChessboardCorners(res, (6, 9),
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                       cv2.CALIB_CB_FAST_CHECK +
                                                       cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            print(corners)
            fnl = cv2.drawChessboardCorners(image, (6, 9), corners, ret)
            cv2.imshow("fnl", fnl)
        else:
            print("No Checkerboard Found")

        cv2.waitKey(0)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
