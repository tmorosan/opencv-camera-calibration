import time
from utils import get_image_arr
import numpy as np
import cv2

from pre_process import get_region_map
from disparities import get_disparity_map
from triangulation import get_distance_map
from cloud_generation import get_points
from post_process import get_processed
from cv2_disparity import compute_disparity

from config import SCALE, SET, OFFSET


def main(file_out="./generated-cloud"):
    left_image_arr = get_image_arr("im0", greyscale=True)
    right_image_arr = get_image_arr("im1", greyscale=True)

    left_image_arr_color = get_image_arr("im0", greyscale=False)

    start_time = time.time()
    disparity_map = compute_disparity(left_image_arr, right_image_arr)
    print(f"Min normalize: {np.min(disparity_map)}, Max normalize: {np.max(disparity_map)}")

    cv2.filterSpeckles(disparity_map, newVal=0, maxSpeckleSize=200, maxDiff=5)
    distance_map = get_distance_map(disparity_map)
    # processed_distance_map = get_processed(distance_map, left_image_arr_color, iterations=3)
    points = get_points(distance_map, left_image_arr_color)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    cv2.imshow("Disparity", disparity_map)
    # cv2.imshow("Distance", distance_map)
    cv2.waitKey(0)

    np.savetxt(f"{file_out}.txt", points)
    np.save(file_out, points)


main()
