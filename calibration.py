import PySpin
import cv2
import numpy as np
import sys
import time
import os
import math  # Added for math.sqrt, etc.
import csv

################################################################
# USER SETTINGS
################################################################
# Update these values to match your printed target if needed.
BOARD_SIZE = (11, 8)         # (width, height) inner corners
# BOARD_SIZE = (9, 6)         # (width, height) inner corners

SQUARE_SIZE_MM = 79.0        # physical size of one square (in mm)
# SQUARE_SIZE_MM = 29.3
# SQUARE_SIZE_MM = 39.3

CALIB_OUTPUT_FILE = "flir_a50_intrinsics.yml"
NUM_IMAGES_REQUIRED = 10     # number of valid captures required
WINDOW_NAME = "FLIR A50"

# Set the known resolutions. Could maybe be read directly from the camera.
VISUAL_RES = (640, 480)      # Visual camera resolution (width, height)
THERMAL_RES = (464, 348)     # Thermal camera resolution (width, height)

# Folder for saving calibration images
CALIB_IMAGES_FOLDER = "calibrationImages"
os.makedirs(CALIB_IMAGES_FOLDER, exist_ok=True)

################################################################
# HELPER FUNCTIONS
################################################################

def find_chessboard(frame_gray, board_size):
    """
    Searches for a chessboard pattern in the provided grayscale image.
    Returns a tuple (found, corners).
    """
    found, corners = cv2.findChessboardCorners(
        frame_gray,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        # Refine corner locations for better accuracy.
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), term)
    return found, corners

def prepare_3d_points(board_size, square_size):
    """
    Generate a grid of 3D object points for a single chessboard view.
    The returned points are arranged in a regular grid in the XY plane (Z=0).
    """
    w, h = board_size
    objp = np.zeros((w * h, 3), np.float32)
    # Create grid coordinates.
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp *= square_size
    return objp

def compute_reprojection_err(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Compute reprojection error and its total average value.
    Returns (total_avg_error, per_view_errors).
    """
    per_view_errors = [0] * len(object_points)
    total_points = 0
    total_error = 0

    for i in range(len(object_points)):
        image_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2)
        n = len(object_points[i])
        per_view_errors[i] = math.sqrt(error * error / n)
        total_error += error * error
        total_points += n

    return math.sqrt(total_error / total_points), per_view_errors

def calibrate_visual_camera(objpoints, imgpoints, image_size):
    """
    Run OpenCV's calibrateCamera with the collected object points and image points.
    The image_size parameter should be a tuple (width, height).
    Returns RMS error, camera matrix, distortion coefficients, rotation vectors, and translation vectors.
    """
    w, h = image_size
    ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (w, h),
        None,
        None
    )
    return ret, cam_mtx, dist, rvecs, tvecs

################################################################
# MAIN STREAM + CALIBRATION
################################################################

def run_intrinsic_calibration_stream():
    """
    1) Connects to the FLIR A50 using Spinnaker.
    2) Continuously grabs frames and displays a window with chessboard detection.
    3) Press 's' to save detected corners (and calibration image) and 'q' to quit.
    4) After collecting sufficient captures, calibrates the camera and saves the intrinsic parameters.
    5) Adjusts the intrinsic parameters to match the thermal resolution.
    6) Saves the reprojection error in a CSV file.
    """
    # --------------------- INIT SPINNAKER --------------------------
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_cams = cam_list.GetSize()

    if num_cams == 0:
        print("[ERROR] No cameras found.")
        cam_list.Clear()
        system.ReleaseInstance()
        return
    
    cam = cam_list[0]  # Use the first available camera.
    cam.Init()

    # Set up camera nodes.
    nodemap = cam.GetNodeMap()

    # 1) Set 'ImageMode' to 'Visual'
    node_image_mode = PySpin.CEnumerationPtr(nodemap.GetNode("ImageMode"))
    if PySpin.IsAvailable(node_image_mode) and PySpin.IsWritable(node_image_mode):
        node_visual = node_image_mode.GetEntryByName("Visual")
        if PySpin.IsAvailable(node_visual) and PySpin.IsReadable(node_visual):
            node_image_mode.SetIntValue(node_visual.GetValue())
            print("[INFO] Set ImageMode to 'Visual'")
        else:
            print("[WARNING] 'Visual' mode not available on this camera.")
    else:
        print("[WARNING] Unable to set 'ImageMode' node. It might not exist on this model.")

    # 2) Set PixelFormat to 'Mono8'
    node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
        node_mono8 = node_pixel_format.GetEntryByName("Mono8")
        if PySpin.IsAvailable(node_mono8) and PySpin.IsReadable(node_mono8):
            node_pixel_format.SetIntValue(node_mono8.GetValue())
            print("[INFO] Set PixelFormat to 'Mono8'")
        else:
            print("[WARNING] 'Mono8' pixel format not available.")
    else:
        print("[WARNING] Unable to set 'PixelFormat' node.")

    # 3) Set acquisition mode to 'Continuous'
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not node_acquisition_mode or not PySpin.IsWritable(node_acquisition_mode):
        print("[ERROR] Unable to set acquisition mode to Continuous.")
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        return

    node_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    node_acquisition_mode.SetIntValue(node_continuous.GetValue())

    # Start acquisition.
    cam.BeginAcquisition()
    print("[INFO] Camera acquisition started (Continuous mode).")
    print("[INFO] Press 's' to save corners (and calibration image), press 'q' to quit.")

    # Prepare calibration arrays.
    objp_single = prepare_3d_points(BOARD_SIZE, SQUARE_SIZE_MM)
    object_points = []  # List of 3D points for all captures.
    image_points = []   # List of corresponding 2D image points.

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    collected = 0
    image_size = None  # Will be set once a valid frame is captured.

    try:
        while True:
            try:
                # Grab a frame.
                image_result = cam.GetNextImage(500)
                if image_result.IsIncomplete():
                    print("[WARNING] Incomplete image. Skipping...")
                    image_result.Release()
                    continue
                
                frame_data = image_result.GetNDArray()
                image_result.Release()

            except PySpin.SpinnakerException as e:
                print("[ERROR] Spinnaker exception:", e)
                break

            if frame_data is None:
                print("[WARNING] No frame data. Skipping...")
                continue

            # Set image_size on the first valid capture.
            current_size = (frame_data.shape[1], frame_data.shape[0])
            if image_size is None:
                image_size = current_size
                print(f"[INFO] Image size set to {image_size}")
            elif image_size != current_size:
                print("[WARNING] Inconsistent image sizes detected. Using the first valid size:", image_size)

            # Attempt chessboard detection.
            found, corners = find_chessboard(frame_data, BOARD_SIZE)
            if found:
                # Convert grayscale to BGR for colored overlay.
                colored_frame = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(colored_frame, BOARD_SIZE, corners, found)
            else:
                colored_frame = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)

            # Show the frame.
            cv2.imshow(WINDOW_NAME, colored_frame)
            key = cv2.waitKey(1) & 0xFF

            # Handle key presses.
            if key == ord('q'):
                print("[INFO] User pressed 'q' -> quitting.")
                break
            elif key == ord('s'):
                if found and len(corners) == (BOARD_SIZE[0] * BOARD_SIZE[1]):
                    object_points.append(objp_single)
                    image_points.append(corners)
                    collected += 1
                    print(f"[INFO] Saved corners. (Total: {collected}/{NUM_IMAGES_REQUIRED})")
                    # Save the colored calibration image.
                    calib_filename = os.path.join(CALIB_IMAGES_FOLDER, f"calib_{collected:02d}.png")
                    cv2.imwrite(calib_filename, colored_frame)
                    print(f"[INFO] Calibration image saved to {calib_filename}")
                    if collected >= NUM_IMAGES_REQUIRED:
                        print("[INFO] Collected enough corners -> stopping capture.")
                        break
                else:
                    print("[WARNING] Chessboard not found. Adjust board position or lighting and try again.")
    finally:
        # Ensure proper cleanup regardless of errors.
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()

    if image_size is None:
        print("[ERROR] No valid images captured for calibration.")
        return

    if collected < 1:
        print("[INFO] No corners were saved. Exiting without calibration.")
        return

    # Run calibration.
    print("\n[INFO] Running calibration with", collected, "valid captures...")
    ret, cam_mtx, dist_coeffs, rvecs, tvecs = calibrate_visual_camera(
        object_points, image_points, image_size
    )
    print("[INFO] Calibration complete.")
    print("      -> RMS error (from calibrateCamera):", ret)
    print("      -> Camera Matrix (Visual Resolution):\n", cam_mtx)
    print("      -> Distortion Coeffs:", dist_coeffs.ravel())

    # --- Resolution Adjustment ---
    scale_x = THERMAL_RES[0] / VISUAL_RES[0]
    scale_y = THERMAL_RES[1] / VISUAL_RES[1]
    adjusted_cam_mtx = cam_mtx.copy()
    adjusted_cam_mtx[0, 0] *= scale_x
    adjusted_cam_mtx[0, 2] *= scale_x
    adjusted_cam_mtx[1, 1] *= scale_y
    adjusted_cam_mtx[1, 2] *= scale_y

    print("[INFO] Adjusted Camera Matrix for Thermal Resolution (464x348):")
    print(adjusted_cam_mtx)

    # Compute and print the reprojection error.
    total_avg_err, per_view_errs = compute_reprojection_err(object_points, image_points, rvecs, tvecs, cam_mtx, dist_coeffs)
    print("[INFO] Total reprojection error:", total_avg_err)
    print("[INFO] Per-view reprojection errors:", per_view_errs)

    # Save reprojection errors to CSV.
    error_csv = "calibration_errors.csv"
    with open(error_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Capture", "ReprojectionError"])
        for idx, err in enumerate(per_view_errs, start=1):
            writer.writerow([idx, err])
        writer.writerow(["Total", total_avg_err])
    print(f"[INFO] Reprojection errors saved to '{error_csv}'.")

    # Save calibration results, including the adjusted matrix.
    print(f"[INFO] Saving intrinsics to '{CALIB_OUTPUT_FILE}'...")
    fs = cv2.FileStorage(CALIB_OUTPUT_FILE, cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix_msx", cam_mtx)
    fs.write("cameraMatrix_thermal", adjusted_cam_mtx)
    fs.write("distCoeffs", dist_coeffs)
    if len(rvecs) > 0:
        rvecs_arr = np.array([rv.flatten() for rv in rvecs], dtype=np.float32)
        tvecs_arr = np.array([tv.flatten() for tv in tvecs], dtype=np.float32)
        fs.write("rvecs", rvecs_arr)
        fs.write("tvecs", tvecs_arr)
    fs.release()
    print("[INFO] Intrinsic calibration saved.")
    print("[INFO] Done.\n")

################################################################
# ENTRY POINT
################################################################

if __name__ == "__main__":
    run_intrinsic_calibration_stream()
