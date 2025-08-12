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
    """Finds the latest PLY file inside the specified folder (or the latest folder if none specified)."""
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

def read_ply_file(folder_specified=None, x_min=None, x_max=None, y_min=None, y_max=None, z=None, tilt=0.0):
    """
    Reads and displays the latest PLY file from the specified folder,
    extracts temperature data, and then visualizes the data along with a plane.
    Spatial coordinate parameters (x_min, x_max, y_min, y_max, z) and an optional tilt (in degrees)
    can be specified for creating the plane.
    """
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

    # Convert the RGB colors to grayscale intensity (assuming the average represents temperature)
    temperature_data = np.mean(colors, axis=1)
    
    # Scale the grayscale intensity to temperature (°C)
    temp_min, temp_max = 10.0, 50.0
    temperatures = temp_min + (temperature_data * (temp_max - temp_min))

    # Print some statistics
    print(f"[INFO] Number of points: {len(points)}")
    print(f"[INFO] Number of temperature values: {len(temperatures)}")

    # Plot an interactive temperature heatmap
    plot_interactive_temperature_heatmap(points, temperatures)

    # Visualize the point cloud with temperature-based coloring and a custom plane
    visualize_point_cloud_with_temperature(pcd, temperatures,
                                             x_min=x_min, x_max=x_max,
                                             y_min=y_min, y_max=y_max, z=z,
                                             tilt=tilt)
    return pcd

def plot_interactive_temperature_heatmap(points, temperatures):
    """Plots a 2D heatmap of temperature values with an interactive cursor."""
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
    
def visualize_point_cloud_with_temperature(pcd, temperatures, x_min=None, x_max=None, y_min=None, y_max=None, z=None, tilt=0.0):
    """
    Visualizes the 3D point cloud with temperature-based coloring.
    Also adds a plane positioned above the anode surface using the provided spatial parameters.
    The plane is horizontal by default (tilt = 0) but can be rotated by specifying tilt (in degrees).
    """
    print("[INFO] Removing outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[INFO] Outliers removed. Remaining points: {len(pcd.points)}")

    # Filter temperature values to match the remaining (inlier) point cloud.
    temperatures = temperatures[ind]

    # Normalize temperature values for the colormap
    temp_min_val, temp_max_val = np.min(temperatures), np.max(temperatures)
    normalized_temp = (temperatures - temp_min_val) / (temp_max_val - temp_min_val)

    # Apply a colormap (e.g., "jet") to the normalized temperature values.
    cmap = plt.get_cmap("jet")
    temp_colors = cmap(normalized_temp)[:, :3]  # Extract RGB channels

    # Set the point cloud colors based on the temperature mapping.
    pcd.colors = o3d.utility.Vector3dVector(temp_colors)

    # Create a plane mesh with the custom spatial parameters.
    # By default, this plane will be horizontal.
    plane = create_plane(pcd, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z=z)

    # If a tilt is specified (nonzero), apply a rotation about the x-axis.
    if tilt != 0.0:
        angle_rad = np.deg2rad(tilt)
        # Create a rotation matrix for rotation around the x-axis.
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                    [0, np.sin(angle_rad),  np.cos(angle_rad)]])
        center = plane.get_center()
        plane.rotate(rotation_matrix, center=center)
        print(f"[INFO] Applied a tilt of {tilt}° to the plane.")

    # Create a coordinate frame for reference.
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # Visualize the point cloud, the plane, and the axes.
    o3d.visualization.draw_geometries([pcd, plane, axes], window_name="3D Thermal Point Cloud")

def create_plane(pcd, x_min=None, x_max=None, y_min=None, y_max=None, z=None, shift=None, rotation_matrix=None):
    """
    Creates a rectangular plane mesh using the specified spatial coordinates.
    If x_min, x_max, y_min, or y_max are not provided, the point cloud's bounding box values are used.
    If z is not provided, the plane is positioned at (min z + 0.5) of the point cloud.
    Duplicates triangles with reversed winding to render both sides.
    
    Optional parameters:
    - shift: a 3-element list/array [dx, dy, dz] to translate the plane.
    - rotation_matrix: a 3x3 NumPy array to rotate the plane around its center.
    """
    aabb = pcd.get_axis_aligned_bounding_box()

    # Set default bounds if not provided.
    if x_min is None:
        x_min = aabb.min_bound[0]
    if y_min is None:
        y_min = aabb.min_bound[1]
    if x_max is None:
        x_max = aabb.max_bound[0]
    if y_max is None:
        y_max = aabb.max_bound[1]
    if z is None:
        z = aabb.min_bound[2] + 0.5  # Default base plane height.

    # Define the four vertices for the plane.
    vertices = np.array([
        [x_min, y_min, z],
        [x_max, y_min, z],
        [x_max, y_max, z],
        [x_min, y_max, z]
    ])

    # Create two triangles to form the plane (front side).
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    # Duplicate triangles with reversed winding for backface rendering.
    back_triangles = triangles[:, ::-1]
    all_triangles = np.vstack((triangles, back_triangles))
    
    # Create and set up the plane mesh.
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector(all_triangles)
    plane.paint_uniform_color([1.0, 0, 0])
    plane.compute_vertex_normals()

    # Apply optional rotation if provided.
    if rotation_matrix is not None:
        center = plane.get_center()
        plane.rotate(rotation_matrix, center=center)
    # Apply optional translation (shift) if provided.
    if shift is not None:
        plane.translate(shift)
    return plane

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a specified timestamped folder inside takenImages and create a plane above the anode surface."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Specify the timestamped folder name (e.g., '2023-03-15_12-30-00'). If not provided, the latest folder is used."
    )
    # Command-line arguments for spatial coordinates of the plane.
    parser.add_argument("--x_min", type=float, default=None, help="Minimum x coordinate for the plane.")
    parser.add_argument("--x_max", type=float, default=None, help="Maximum x coordinate for the plane.")
    parser.add_argument("--y_min", type=float, default=None, help="Minimum y coordinate for the plane.")
    parser.add_argument("--y_max", type=float, default=None, help="Maximum y coordinate for the plane.")
    parser.add_argument("--z", type=float, default=None, help="Z coordinate for the plane (height above anode).")
    # Set tilt (in degrees) to 90 by default to make the plane vertical.
    parser.add_argument("--tilt", type=float, default=115.0, help="Tilt angle (in degrees) to rotate the plane about the x-axis.")
    
    args = parser.parse_args()
    read_ply_file(args.folder, x_min=args.x_min, x_max=args.x_max,
                  y_min=args.y_min, y_max=args.y_max, z=args.z, tilt=args.tilt)
