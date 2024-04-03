import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

VOXEL_SIZE = 1


def init_window():
    cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp', 600, 600)

    def nothing(x):
        pass

    cv2.createTrackbar('numDisparities', 'disp', 16, 17, nothing)
    cv2.createTrackbar('blockSize', 'disp', 5, 50, nothing)
    cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
    cv2.createTrackbar('preFilterSize', 'disp', 4, 25, nothing)
    cv2.createTrackbar('preFilterCap', 'disp', 50, 62, nothing)
    cv2.createTrackbar('textureThreshold', 'disp', 40, 100, nothing)
    cv2.createTrackbar('uniquenessRatio', 'disp', 5, 100, nothing)
    cv2.createTrackbar('speckleRange', 'disp', 20, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'disp', 20, 25, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'disp', 11, 25, nothing)
    cv2.createTrackbar('minDisparity', 'disp', 16, 25, nothing)


def disparity_to_3d_occupancy(depth_map):
    # Compute the depth map
    # depth_map = (disparity_map * VOXEL_SIZE).astype(np.uint16)

    # Define the size of the 3D grid
    height, width = depth_map.shape
    depth = 1
    grid_shape = (height, width, depth)

    # Initialize the 3D occupancy grid
    grid = np.zeros(grid_shape, dtype=np.uint16)

    # Set the depth values in the grid
    grid[:, :, 0] = depth_map

    return grid


def compute_disparity(stereo, image_l, image_r):
    img_left_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(0)
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(img_left_gray, img_right_gray)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it
    # is essential to convert it to CV_32F and scale it down 16 times.
    # Converting to float32
    disparity = disparity.astype(np.float32)

    # Scaling down the disparity values and normalizing them
    disparity = (disparity / 16.0 - minDisparity) / numDisparities

    # Displaying the disparity map
    cv2.imshow("disp", disparity)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # parser = argparse.ArgumentParser("main")
    # parser.add_argument("--img")
    #
    # args = parser.parse_args()

    image_l = cv2.imread("data/artroom1/im0.png")
    image_r = cv2.imread("data/artroom1/im1.png")

    init_window()
    stereo = cv2.StereoBM.create()
    compute_disparity(stereo, image_l, image_r)


if __name__ == '__main__':
    main()
