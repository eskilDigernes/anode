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
    """Finds the latest PLY file inside the specified folder (or latest folder if none specified)."""
    folder = get_folder(folder_specified)
    if not folder:
        return None

    ply_files = glob(os.path.join(folder, "*.ply"))
    if not ply_files:
        print(f"[ERROR] No PLY file found in {folder}!")
        return None

    latest_ply = max(ply_files, key=os.path.getmtime)  # Get the most recent PLY file
    print(f"[INFO] Using latest PLY file: {latest_ply}")
    return latest_ply

def read_ply_file(folder_specified=None):
    """Reads and displays the latest PLY file from the specified folder and extracts temperature data."""
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

    # Convert Open3D point cloud to NumPy arrays
    points = np.asarray(pcd.points)  # XYZ coordinates
    colors = np.asarray(pcd.colors)  # RGB colors (assumed to store temperature data)

    if colors.size == 0:
        print("[WARNING] No color data found in PLY! Temperature mapping might be unavailable.")
        return

    # Convert RGB colors to grayscale intensity (assuming grayscale represents temperature)
    temperature_data = np.mean(colors, axis=1)  # Take the average of R, G, and B
    
    # Scale the grayscale intensity to temperature (°C)
    # temp_min, temp_max = 20.0, 100.0  # 
    # temp_min, temp_max = 10.0, 50.0   # 
    
    # 6/3
    # temp_min, temp_max = 175, 600.0   #  12:17
    # temp_min, temp_max = 175, 500.0   #  12:22
    temp_min, temp_max = 175, 450.0   #  12:25
    # temp_min, temp_max = 10, 175.0    #  12:39
    # temp_min, temp_max = 40, 175.0    #  12:43
    # temp_min, temp_max = 175, 600.0    #  15:40 uke gammel
    # temp_min, temp_max = 40, 500.0      #  15:50 

    # # 7/3
    # temp_min, temp_max = 51, 500.0      #  12:14
    # temp_min, temp_max = 30, 175.0      #  12:16     
    # temp_min, temp_max = 20, 175.0      #  12:19 
    # temp_min, temp_max = 36, 180.0      #  12:39 
    # temp_min, temp_max = 36, 180.0      #  12:41
   
   # 8/3
    # temp_min, temp_max = 20, 175.0      #  12:41

    # 9/3
    # temp_min, temp_max = 20, 40.0      #  12:41
    temperatures = temp_min + (temperature_data * (temp_max - temp_min))

    # Print some statistics
    print(f"[INFO] Number of points: {len(points)}")
    print(f"[INFO] Number of temperature values: {len(temperatures)}")

    # Plot interactive temperature heatmap
    plot_interactive_temperature_heatmap(points, temperatures)

    # Display the point cloud with Open3D
    visualize_point_cloud_with_temperature(pcd, temperatures)

    return pcd

def plot_interactive_temperature_heatmap(points, temperatures):
    """Plots a 2D heatmap of temperature values with interactive cursor."""
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=temperatures, cmap="jet", s=5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Temperature Distribution in Point Cloud")

    def on_hover(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                distances = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
                index = np.argmin(distances)
                temp_value = temperatures[index]
                ax.set_title(f"Temperature: {temp_value:.2f}°C (X: {x:.2f}, Y: {y:.2f})")
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    plt.show()
    
def visualize_point_cloud_with_temperature(pcd, temperatures):
    """Visualizes the 3D point cloud with temperature-based coloring and removes outliers."""
    # Remove outliers using statistical outlier removal
    print("[INFO] Removing outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[INFO] Outliers removed. Remaining points: {len(pcd.points)}")

    # Filter the temperature values to match the inlier indices
    temperatures = temperatures[ind]

    # Normalize temperature values for colormap
    temp_min, temp_max = np.min(temperatures), np.max(temperatures)
    normalized_temp = (temperatures - temp_min) / (temp_max - temp_min)

    # Apply colormap (e.g., "jet") to the normalized temperature values
    cmap = plt.get_cmap("jet")
    temp_colors = cmap(normalized_temp)[:, :3]  # Extract RGB values (ignore alpha channel)

    # Set the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(temp_colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="3D Thermal Point Cloud")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specified timestamped folder inside taken_images.")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Specify the timestamped folder name (e.g., '2023-03-15_12-30-00'). If not provided, the latest folder will be used."
    )
    args = parser.parse_args()
    read_ply_file(args.folder)
