import os
import numpy as np
import cv2
import open3d as o3d  # pip install open3d
from glob import glob

# Get the script's directory
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
TAKEN_IMAGES_DIR = os.path.join(SCRIPT_DIR, "takenImages")

# Make sure these two files point to the same calibration results
# you produced in Parts 1 & 2 (the scaled "thermal" intrinsics, and the extrinsic file).
FLIR_INTRINSICS_FILE = "flir_a50_intrinsics.yml"
ORIENTATION_FILE = "orientation.yml"

"""
HELPER FUNCTIONS
"""

def get_latest_folder():
    """Finds the latest timestamped folder inside takenImages."""
    if not os.path.exists(TAKEN_IMAGES_DIR):
        print("[ERROR] No takenImages directory found!")
        return None

    subdirs = [
        os.path.join(TAKEN_IMAGES_DIR, d)
        for d in os.listdir(TAKEN_IMAGES_DIR)
        if os.path.isdir(os.path.join(TAKEN_IMAGES_DIR, d))
    ]
    if not subdirs:
        print("[ERROR] No timestamped folders found in takenImages!")
        return None

    latest_folder = max(subdirs, key=os.path.getmtime)
    print(f"[INFO] Latest takenImages folder: {latest_folder}")
    return latest_folder

def get_latest_ply_file():
    """Finds the latest Helios 3D PLY file in the newest timestamped folder."""
    latest_folder = get_latest_folder()
    if not latest_folder:
        return None

    ply_files = glob(os.path.join(latest_folder, "*helios_3D.ply"))
    if not ply_files:
        print(f"[ERROR] No PLY file found in {latest_folder}!")
        return None

    ply_files.sort(key=os.path.getmtime, reverse=True)
    latest_ply = ply_files[0]
    print(f"[INFO] Using latest PLY file: {latest_ply}")
    return latest_ply, latest_folder

def get_latest_thermal_image():
    """Finds the latest thermal image file (e.g. '*thermal*.png') in the newest folder."""
    latest_folder = get_latest_folder()
    if not latest_folder:
        return None

    thermal_images = glob(os.path.join(latest_folder, "*thermal*.png"))
    if not thermal_images:
        print(f"[ERROR] No thermal image found in {latest_folder}!")
        return None

    thermal_images.sort(key=os.path.getmtime, reverse=True)
    latest_thermal_image = thermal_images[0]
    print(f"[INFO] Using thermal image: {latest_thermal_image}")
    return latest_thermal_image

def load_calibration():
    """Loads camera intrinsics (thermal) and extrinsics from YAML files."""
    if not os.path.exists(FLIR_INTRINSICS_FILE) or not os.path.exists(ORIENTATION_FILE):
        print("[ERROR] Calibration files missing.")
        return None, None, None, None

    # Load FLIR intrinsics
    fs = cv2.FileStorage(FLIR_INTRINSICS_FILE, cv2.FILE_STORAGE_READ)

    # Try to read "cameraMatrix_msx"; fallback to "cameraMatrix" if not available.
    cam_mtx_node = fs.getNode("cameraMatrix_msx")
    if not cam_mtx_node.empty():
        cam_mtx = cam_mtx_node.mat()
        print("[INFO] Using cameraMatrix_msx from YAML.")
    else:
        print("[WARNING] cameraMatrix_msx not found; using cameraMatrix.")
        cam_mtx = fs.getNode("cameraMatrix").mat()

    dist_coeffs = fs.getNode("distCoeffs").mat()
    fs.release()

    # Load extrinsics
    fs = cv2.FileStorage(ORIENTATION_FILE, cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rotationVector").mat()
    tvec = fs.getNode("translationVector").mat()
    fs.release()

    if cam_mtx is None or dist_coeffs is None or rvec is None or tvec is None:
        print("[ERROR] One or more calibration matrices are missing or invalid!")
        return None, None, None, None

    return cam_mtx, dist_coeffs, rvec, tvec

def project_3D_to_thermal(xyz_points, rvec, tvec, cam_mtx, dist_coeffs):
    """Projects Nx3 3D points into the 2D thermal image plane using cv2.projectPoints."""
    projected_points, _ = cv2.projectPoints(xyz_points, rvec, tvec, cam_mtx, dist_coeffs)
    return projected_points.reshape(-1, 2)

# def bilinear_interpolate(image, x, y):
#     """
#     Bilinear interpolation for a single (x, y) in 'image'.
#     'image' is assumed to be 2D float32, shape (rows, cols).
#     """
#     h, w = image.shape
#     if x < 0 or x > w - 1 or y < 0 or y > h - 1:
#         return 0.0
#     x0 = int(np.floor(x))
#     y0 = int(np.floor(y))
#     x1 = min(x0 + 1, w - 1)
#     y1 = min(y0 + 1, h - 1)

#     dx = x - x0
#     dy = y - y0

#     I00 = image[y0, x0]
#     I01 = image[y0, x1]
#     I10 = image[y1, x0]
#     I11 = image[y1, x1]

#     return ((1 - dx) * (1 - dy) * I00 +
#             dx * (1 - dy) * I01 +
#             (1 - dx) * dy * I10 +
#             dx * dy * I11)


def bilinear_interpolate(img: np.ndarray, x, y, *, clamp: bool = True):
    """
    Bilinear interpolation on a single‑channel 2‑D image.

    Parameters
    ----------
    img   : (H, W) float32 ndarray
    x, y  : float or ndarray of floats –  pixel coordinates in *column*,*row* order
    clamp : If True (default) clamp x,y to the valid range; otherwise return 0 for
            every sample that falls outside.

    Returns
    -------
    float or ndarray with the same shape as np.broadcast(x, y)
    """
    if img.ndim != 2:
        raise ValueError("img must be a single‑channel (H, W) array")

    h, w = img.shape
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if clamp:
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
    else:
        mask = (x < 0) | (x > w - 1) | (y < 0) | (y > h - 1)
        # pre‑allocate result, fill zeros where mask==True later
        out = np.zeros_like(x, dtype=np.float32)

        # only interpolate where inside
        x, y = np.where(mask, 0, x), np.where(mask, 0, y)

    # integer neighbours
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    # distances
    dx = x - x0
    dy = y - y0

    # four neighbours
    I00 = img[y0, x0]
    I10 = img[y0, x1]
    I01 = img[y1, x0]
    I11 = img[y1, x1]

    # textbook bilinear formula
    interp = (
        (1 - dx) * (1 - dy) * I00 +
        dx        * (1 - dy) * I10 +
        (1 - dx) *      dy  * I01 +
        dx        *      dy  * I11
    ).astype(np.float32)

    if clamp:
        return interp          # same shape as x,y
    else:
        out[~mask] = interp    # keep zeros (or NaNs) outside
        return out



def fuse_thermal_with_point_cloud():
    """
    Loads a Helios 3D PLY, loads the latest thermal image, applies extrinsics and intrinsics,
    and saves a new PLY with grayscale "thermal" color. Only points that project within the
    thermal image bounds are retained.
    """
    # --- 1) Find the latest PLY and thermal image
    result = get_latest_ply_file()
    if result is None:
        return
    ply_file, output_folder = result

    thermal_image_path = get_latest_thermal_image()
    if thermal_image_path is None:
        return

    # --- 2) Load the point cloud
    print("[INFO] Loading 3D point cloud...")
    pcd = o3d.io.read_point_cloud(ply_file)
    xyz_points = np.asarray(pcd.points)  # shape (N, 3)

    # --- 3) Load the thermal image
    print("[INFO] Loading thermal image...")
    thermal_image = cv2.imread(thermal_image_path, cv2.IMREAD_UNCHANGED)
    if thermal_image is None:
        print("[ERROR] Could not load thermal image.")
        return

    # If the thermal image is actually 16-bit, cast it to float32
    thermal_image = thermal_image.astype(np.float32)

    # --- 4) Load intrinsics & extrinsics
    print("[INFO] Loading calibration data...")
    cam_mtx, dist_coeffs, rvec, tvec = load_calibration()
    if cam_mtx is None:
        return

    # --- 5) Project each 3D point into the thermal image
    print("[INFO] Projecting 3D points onto thermal image...")
    projected_points = project_3D_to_thermal(xyz_points, rvec, tvec, cam_mtx, dist_coeffs)

    thermal_min, thermal_max = thermal_image.min(), thermal_image.max()
    delta = thermal_max - thermal_min

    valid_points = []
    valid_colors = []
    h_img, w_img = thermal_image.shape

    # Loop through each 3D point and check if its projection is within bounds.
    for i in range(xyz_points.shape[0]):
        x_2d, y_2d = projected_points[i, 0], projected_points[i, 1]
        if 0 <= x_2d < w_img and 0 <= y_2d < h_img:
            raw_val = bilinear_interpolate(thermal_image, x_2d, y_2d)
            mapped_val = ((raw_val - thermal_min) / delta) * 255.0 if delta > 1e-6 else 0.0
            mapped_val = np.clip(mapped_val, 0, 255)
            gray = mapped_val / 255.0
            valid_colors.append([gray, gray, gray])
            valid_points.append(xyz_points[i])
        # Points that project out-of-bounds are skipped.

    if not valid_points:
        print("[ERROR] No valid points found after projection!")
        return

    # --- 6) Update point cloud with filtered points and colors, then save
    print("[INFO] Updating point cloud with thermal data...")
    pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(valid_colors))

    output_ply_file = os.path.join(output_folder, "thermal_fused.ply")
    print(f"[INFO] Saving thermal-fused point cloud to {output_ply_file}...")
    o3d.io.write_point_cloud(output_ply_file, pcd)

    print("[INFO] Thermal overlay saved successfully!")

if __name__ == "__main__":
    fuse_thermal_with_point_cloud()
