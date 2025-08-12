import os
import re
import argparse
import numpy as np
import open3d as o3d  # pip install open3d
import matplotlib.pyplot as plt  # pip install matplotlib
from glob import glob

# Get the script's directory
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
TAKEN_IMAGES_DIR = os.path.join(SCRIPT_DIR, "takenImages")

def get_latest_folder():
    """Finds the latest timestamped folder inside takenImages based on folder name."""
    if not os.path.exists(TAKEN_IMAGES_DIR):
        print("[ERROR] No takenImages directory found!")
        return None

    # Regular expression for timestamp folder format: YYYY-MM-DD_HH-MM-SS
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')

    subdirs = [
        os.path.join(TAKEN_IMAGES_DIR, d)
        for d in os.listdir(TAKEN_IMAGES_DIR)
        if os.path.isdir(os.path.join(TAKEN_IMAGES_DIR, d)) and timestamp_pattern.match(d)
    ]

    if not subdirs:
        print("[ERROR] No timestamped folders found in takenImages!")
        return None

    latest_folder = max(subdirs, key=lambda d: os.path.basename(d))
    print(f"[INFO] Latest takenImages folder: {latest_folder}")
    return latest_folder

def get_folder(folder_specified=None):
    """
    Returns the folder to be used:
    - If folder_specified is provided and exists, that folder is used.
    - Otherwise, it falls back to get_latest_folder().
    """
    if folder_specified:
        full_path = os.path.join(TAKEN_IMAGES_DIR, folder_specified)
        if os.path.exists(full_path) and os.path.isdir(full_path):
            print(f"[INFO] Using specified folder: {full_path}")
            return full_path
        else:
            print(f"[ERROR] The specified folder {full_path} does not exist or is not a directory.")
            return None
    else:
        return get_latest_folder()

def get_latest_ply_file(folder_specified=None):
    """Finds the specified PLY file (helios_3D.ply) inside the folder."""
    folder = get_folder(folder_specified)
    if not folder:
        return None

    # Look for the specific file "helios_3D.ply"
    specific_ply_file = os.path.join(folder, "helios_3D.ply")
    if os.path.exists(specific_ply_file):
        print(f"[INFO] Using specified PLY file: {specific_ply_file}")
        return specific_ply_file

    print(f"[ERROR] The file 'helios_3D.ply' was not found in {folder}!")
    return None

def read_ply_file(folder_specified=None):
    """Reads and displays the latest PLY file from the specified folder."""
    ply_file = get_latest_ply_file(folder_specified)
    if not ply_file:
        print("[ERROR] No PLY file available for reading.")
        return

    print(f"[INFO] Loading PLY file: {ply_file}")
    pcd = o3d.io.read_point_cloud(ply_file)

    # Check if the file is correctly loaded
    if not pcd.has_points():
        print("[ERROR] Failed to load point cloud.")
        return

    # Remove outliers using statistical outlier removal
    print("[INFO] Removing outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[INFO] Outliers removed. Remaining points: {len(pcd.points)}")

    # Convert Open3D point cloud to NumPy arrays
    points = np.asarray(pcd.points)  # XYZ coordinates

    # Print some statistics
    print(f"[INFO] Number of points: {len(points)}")

    # Display the point cloud with Open3D
    visualize_point_cloud(pcd)

    return pcd
    
def visualize_point_cloud(pcd):
    """Visualizes the 3D point cloud with a colormap based on the Z-coordinate."""
    # Convert Open3D point cloud to NumPy arrays
    points = np.asarray(pcd.points)  # XYZ coordinates

    # Normalize the Z-coordinate to range [0, 1] for colormap
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    normalized_z = (points[:, 2] - z_min) / (z_max - z_min)

    # Apply a colormap (e.g., "jet") to the normalized Z values
    cmap = plt.get_cmap("jet")
    colors = cmap(normalized_z)[:, :3]  # Extract RGB values (ignore alpha channel)

    # Set the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud with Colormap") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specified timestamped folder inside takenImages.")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Specify the timestamped folder name (e.g., '2023-03-15_12-30-00'). If not provided, the latest folder will be used."
    )
    args = parser.parse_args()
    read_ply_file(args.folder)