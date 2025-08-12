import os
import time
import numpy as np
import cv2
import csv

# ---------- Your Custom Imports -----------
from camera.helios import Helios       # Helios wrapper
from camera.flir import Flir           # FLIR wrapper
from camsyncer import Camsyncer        # Manages camera sync
from supportiveFeatures.readConfig import readConfig, writeTrainingSession
from supportiveFeatures.Timer import Timer

# ---------- Chessboard/Calibration Constants -----------
BOARD_SIZE = (11, 8)  # (columns, rows) inner corners
NUM_CORNERS = BOARD_SIZE[0] * BOARD_SIZE[1]

# Intrinsics file (from Part 1)
FLIR_INTRINSICS_FILE = "flir_a50_intrinsics.yml"

# Output file for extrinsic calibration
ORIENTATION_FILE = "orientation.yml"

# This should match the thermal resolution you used in Part 1
TARGET_WIDTH = 464
TARGET_HEIGHT = 348

# Folder for saving extrinsic calibration images
CALIB_IMAGES_FOLDER = "calibrationImages"
os.makedirs(CALIB_IMAGES_FOLDER, exist_ok=True)

def find_chessboard_16u(frame_16u):
    """
    Detects a chessboard pattern in a 16-bit image.
    Converts to 8-bit and returns (found, corners).
    """
    if frame_16u.ndim == 3:
        # Assume image is already grayscale; skip conversion
        frame_gray = cv2.cvtColor(frame_16u, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame_16u

    min_val, max_val, _, _ = cv2.minMaxLoc(frame_gray)
    if (max_val - min_val) < 1e-6:
        scaled = np.zeros_like(frame_gray, dtype=np.uint8)
    else:
        factor = 255.0 / (max_val - min_val)
        scaled = ((frame_gray - min_val) * factor).astype(np.uint8)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(scaled, BOARD_SIZE, flags)

    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(scaled, corners, (11, 11), (-1, -1), criteria)

    return found, corners

def interpolate_xyz(xyz_image, pt):
    """
    Interpolate a subpixel (x, y) location in the xyz_image (H x W x 3).
    Returns a [X, Y, Z] triple.
    """
    patch = cv2.getRectSubPix(xyz_image, (1, 1), (pt[0], pt[1]))
    return patch[0, 0]

def main():
    # 1) Setup paths/configuration
    parentPathStoringImages = CALIB_IMAGES_FOLDER
    folderName = time.strftime("%Y-%m-%d_%H-%M-%S") + os.sep
    savePath = os.path.join(parentPathStoringImages, folderName)
    os.makedirs(savePath, exist_ok=True)

    # Read configuration files
    ipAdressesConfigFile = "configs/config.json"
    ipConfig = readConfig(ipAdressesConfigFile)

    heliosCameraConfigFile = "configs/HeliosCameraConfig.json"
    heliosStreamConfigFile = "configs/HeliosStreamConfig.json"
    flirCameraConfigFile = "configs/FlirCameraConfig.json"
    flirStreamConfigFile = "configs/FlirStreamConfig.json"
    heliosColorConfigFile = "configs/HeliosColorLimits.json"

    heliosCameraConfig = readConfig(heliosCameraConfigFile)
    heliosStreamConfig = readConfig(heliosStreamConfigFile)
    heliosColorConfig  = readConfig(heliosColorConfigFile)
    flirCameraConfig   = readConfig(flirCameraConfigFile)
    flirStreamConfig   = readConfig(flirStreamConfigFile)

    # 2) Instantiate cameras
    hc = Helios(ipConfig["heliosIP"], "helios")
    fc = Flir(ipConfig["flirSN"], "flir")

    timer = Timer()
    sync = Camsyncer()
    sync.register_camera(hc)
    sync.register_camera(fc)
    sync.connect()

    streamDict = {hc.name: heliosStreamConfig, fc.name: flirStreamConfig}
    sync.setStreamConfig(streamDict)
    
    configDict = {hc.name: heliosCameraConfig, fc.name: flirCameraConfig}
    sync.setCameraConfig(configDict)

    colorDict = {hc.name: heliosColorConfig, fc.name: ""}
    sync.setColorConfig(colorDict)

    # 3) Capture images
    print("[INFO] Taking images...")
    timer.start()
    imageDict, timeStampDict = sync.get_image_serial()
    totalTime = timer.stop()
        
    if not imageDict:
        print("[ERROR] No images were captured.")
        sync.disconnect()
        return

    print("[INFO] Saving images to:", savePath)
    pathDict = {
        hc.name: ["helios_intensity.png", "helios_heatmap.png", "helios_3D.ply"],
        fc.name: ["flir_thermal.png", "flir_visual.png"]
    }
    sync.saveImage(savePath, pathDict, imageDict)

    # 4) Load frames
    helios_data = imageDict.get(hc.name, {})
    flir_data   = imageDict.get(fc.name, {})

    helios_intensity = helios_data.get("intensity", None)
    helios_xyz       = helios_data.get("xyz", None)
    flir_thermal     = flir_data.get("2D", None)

    if helios_intensity is None or helios_xyz is None or flir_thermal is None:
        print("[ERROR] Missing Helios or FLIR data.")
        sync.disconnect()
        return

    # 5) Detect chessboard in both images and save the grayscale calibration images.
    print("[INFO] Detecting chessboard in Helios intensity...")
    found_hlt, corners_hlt = find_chessboard_16u(helios_intensity)
    if not found_hlt or corners_hlt.shape[0] != NUM_CORNERS:
        print("[ERROR] Could not find chessboard corners in Helios image.")
        sync.disconnect()
        return

    print("[INFO] Detecting chessboard in FLIR thermal...")
    found_flir, corners_flir = find_chessboard_16u(flir_thermal)
    if not found_flir or corners_flir.shape[0] != NUM_CORNERS:
        print("[ERROR] Could not find chessboard corners in FLIR image.")
        sync.disconnect()
        return

    # Draw chessboard corners on the grayscale images.
    helios_with_corners = cv2.drawChessboardCorners(helios_intensity.copy(), BOARD_SIZE, corners_hlt, found_hlt)
    helios_calib_img = os.path.join(savePath, "helios_calib.png")
    cv2.imwrite(helios_calib_img, helios_with_corners)
    print(f"[INFO] Saved Helios calibration image: {helios_calib_img}")

    flir_with_corners = cv2.drawChessboardCorners(flir_thermal.copy(), BOARD_SIZE, corners_flir, found_flir)
    flir_calib_img = os.path.join(savePath, "flir_calib.png")
    cv2.imwrite(flir_calib_img, flir_with_corners)
    print(f"[INFO] Saved FLIR calibration image: {flir_calib_img}")

    # Keep corners in float for subpixel interpolation.
    corners_hlt = corners_hlt.reshape(-1, 2).astype(np.float32)
    corners_flir = corners_flir.reshape(-1, 2).astype(np.float32)

    # 6) Build 3D->2D correspondences.
    object_points_mm = []
    image_points_2d  = []
    h_xyz, w_xyz = helios_xyz.shape[:2]

    for i, pt in enumerate(corners_hlt):
        x, y = pt
        if x < 0 or y < 0 or x >= w_xyz or y >= h_xyz:
            print(f"[ERROR] Corner {i} at ({x:.2f}, {y:.2f}) out of Helios XYZ bounds.")
            sync.disconnect()
            return

        X, Y, Z = interpolate_xyz(helios_xyz, (x, y))
        object_points_mm.append([X, Y, Z])
        image_points_2d.append(corners_flir[i])

    object_points_mm = np.array(object_points_mm, dtype=np.float32)
    image_points_2d  = np.array(image_points_2d,  dtype=np.float32)

    # 7) Load intrinsics.
    if not os.path.exists(FLIR_INTRINSICS_FILE):
        print("[ERROR] Missing FLIR intrinsics file:", FLIR_INTRINSICS_FILE)
        sync.disconnect()
        return

    fs = cv2.FileStorage(FLIR_INTRINSICS_FILE, cv2.FILE_STORAGE_READ)
    cam_mtx_node = fs.getNode("cameraMatrix_msx")
    if not cam_mtx_node.empty():
        cam_mtx = cam_mtx_node.mat()
        print("[INFO] Using 'cameraMatrix_msx' from YAML.")
    else:
        print("[WARNING] 'cameraMatrix_msx' not found. Using 'cameraMatrix' instead.")
        cam_mtx = fs.getNode("cameraMatrix").mat()
    dist_coeff = fs.getNode("distCoeffs").mat()
    fs.release()

    # 8) SolvePnP (extrinsic calibration)
    print("[INFO] Running solvePnP (Helios -> FLIR).")
    success, rvec, tvec = cv2.solvePnP(
        object_points_mm,
        image_points_2d,
        cam_mtx,
        dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE
    )


    if not success:
        print("[ERROR] solvePnP failed to find a valid solution.")
        sync.disconnect()
        return

    print("[INFO] solvePnP succeeded.")
    print("  rotationVector (Rodrigues) =", rvec.ravel())
    print("  translationVector (mm)     =", tvec.ravel())

    # 9) Compute reprojection error and save errors to CSV.
    projected_points, _ = cv2.projectPoints(object_points_mm, rvec, tvec, cam_mtx, dist_coeff)
    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(image_points_2d - projected_points, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    
    error_csv = os.path.join(savePath, "extrinsic_errors.csv")
    with open(error_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["PointIndex", "ReprojectionError"])
        for idx, err in enumerate(errors, start=1):
            writer.writerow([idx, err])
        writer.writerow(["RMSE", rmse])
    print(f"[INFO] Extrinsic reprojection errors saved to '{error_csv}'.")

    # 10) Save extrinsics.
    print("[INFO] Saving extrinsics to", ORIENTATION_FILE)
    fs = cv2.FileStorage(ORIENTATION_FILE, cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix", cam_mtx)
    fs.write("distCoeffs", dist_coeff)
    fs.write("rotationVector", rvec)
    fs.write("translationVector", tvec)
    fs.release()

    # 11) Cleanup.
    timeStampDict["Total time taking images"] = totalTime
    writeTrainingSession(os.path.join(savePath, "timestamp.json"), timeStampDict)

    sync.disconnect()
    print("[INFO] Done. Extrinsic calibration saved.")

if __name__ == '__main__':
    main()
