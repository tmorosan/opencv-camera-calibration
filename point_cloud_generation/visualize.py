import open3d as o3d
import numpy as np


def load_and_visualize_point_cloud(file_path):
    # Load the point cloud data from a .npy file
    data = np.load(file_path)

    # Extract points (X, Y, Z) and colors (r, g, b)
    # Assuming colors are stored in the range 0-255
    points = data[:, 0:3]  # First three columns are X, Y, Z
    np_points = np.random.rand(100, 3)
    colors = data[:, 3:6] / 255.0  # Last three columns are r, g, b and need normalization

    print(points.shape)
    print(points.max())
    print(points.min())
    points = points + abs(points.min())
    print("===========")
    print(np_points.shape)
    print(np_points.max())
    print(np_points.min())

    # Create a PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud], mesh_show_wireframe=True)


if __name__ == "__main__":
    file_path = "generated-cloud.npy"  # Replace with your actual file path
    load_and_visualize_point_cloud(file_path)
