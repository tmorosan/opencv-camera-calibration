import cv2
import numpy as np


def compute_disparity(image_l, image_r):
    stereo = cv2.StereoBM.create()

    # 272 15 0 9 25 6 3 2 36 25 5
    # 224 23 0 31 4 12 22 23 6 0 5
    numDisparities = 16
    blockSize = 9
    minDisparity = 5
    preFilterSize = 2
    specleWindowSize = 12
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities * 16)
    stereo.setBlockSize(blockSize * 2 + 5)
    stereo.setPreFilterType(0)
    stereo.setPreFilterSize(preFilterSize * 2 + 5)
    stereo.setPreFilterCap(25)
    stereo.setTextureThreshold(6)
    stereo.setUniquenessRatio(3)
    stereo.setSpeckleRange(2)
    stereo.setSpeckleWindowSize(specleWindowSize * 2)
    stereo.setDisp12MaxDiff(25)
    stereo.setMinDisparity(minDisparity)

    disparity = stereo.compute(image_l, image_r)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.
    # Converting to float32
    disparity = disparity.astype(np.float32)
    min = np.min(disparity)
    max = np.max(disparity)
    print(f"Min: {min}, Max: {max}")
    # Scaling down the disparity values and normalizing them
    # disparity = (disparity / 16.0) / (numDisparities * 16)

    # clamp to 0-255
    disparity = np.uint8(255 * (disparity - min) / (max - min))
    return disparity
