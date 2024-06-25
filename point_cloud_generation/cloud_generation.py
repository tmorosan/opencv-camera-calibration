import math

from config import SCALE, OFFSET, THRESHOLD_DISTANCE_OUTLIER


def get_points(distances, image_arr):
    points = []
    # Iterate through every pixel.
    for row in range(OFFSET, distances.shape[0] - OFFSET):
        for col in range(OFFSET, distances.shape[1] - OFFSET):
            [r, g, b] = image_arr[row, col]
            [X, Y, Z] = distances[row, col]
            # Filter out weird math overflows and points marked as deleted by denoiser.
            if Z is None:
                continue
            if math.isnan(X) or math.isnan(Y) or math.isnan(Z):
                continue
            if abs(X) > THRESHOLD_DISTANCE_OUTLIER:
                continue
            if abs(Y) > THRESHOLD_DISTANCE_OUTLIER:
                continue
            if abs(Z) > THRESHOLD_DISTANCE_OUTLIER:
                continue
            if math.isinf(X) or math.isinf(Y) or math.isinf(Z):
                continue

            # Add point.
            print(X, Y, Z, r, g, b)
            points.append([X, Y, Z, r, g, b])
    return points
