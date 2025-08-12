"""
This script captures images from Helios and FLIR cameras, then saves the results and timestamps.
It uses Camsyncer to synchronize camera connections and image acquisitions.
"""

import time
import os
import cProfile

from camera.helios import Helios
from camera.flir import Flir
from camsyncer import Camsyncer
from supportiveFeatures.readConfig import readConfig, writeTrainingSession
from supportiveFeatures.Timer import Timer


def main():
    # ------------------------------------------
    # 1) Define paths for storing images & configs
    # ------------------------------------------
    parent_path_storing_images = "takenImages\\"
    ip_addresses_config_file = "configs/config.json"

    # Helios
    helios_color_config_file = "configs/HeliosColorLimits.json"
    helios_camera_config_file = "configs/HeliosCameraConfig.json"
    helios_stream_config_file = "configs/HeliosStreamConfig.json"

    # FLIR
    flir_camera_config_file = "configs/FlirCameraConfig.json"
    flir_stream_config_file = "configs/FlirStreamConfig.json"

    # ------------------------------------------
    # 2) Read configuration files
    # ------------------------------------------
    ip_config = readConfig(ip_addresses_config_file)

    helios_color_config = readConfig(helios_color_config_file)
    helios_camera_config = readConfig(helios_camera_config_file)
    helios_stream_config = readConfig(helios_stream_config_file)

    flir_camera_config = readConfig(flir_camera_config_file)
    flir_stream_config = readConfig(flir_stream_config_file)

    # ------------------------------------------
    # 3) Create camera devices
    # ------------------------------------------
    helios_camera = Helios(ip_config["heliosIP"], "helios")
    flir_camera = Flir(ip_config["flirSN"], "flir")

    # ------------------------------------------
    # 4) Initialize and configure cameras
    # ------------------------------------------
    sync = Camsyncer()
    sync.register_camera(helios_camera)
    sync.register_camera(flir_camera)
    sync.connect()

    # Stream Config
    stream_dict = {
        helios_camera.name: helios_stream_config,
        flir_camera.name: flir_stream_config
    }
    sync.setStreamConfig(stream_dict)

    # Camera Config
    config_dict = {
        helios_camera.name: helios_camera_config,
        flir_camera.name: flir_camera_config
    }
    sync.setCameraConfig(config_dict)

    # Color Config (only Helios has color limits in this setup)
    color_dict = {
        helios_camera.name: helios_color_config,
        flir_camera.name: ""
    }
    sync.setColorConfig(color_dict)

    # ------------------------------------------
    # 5) Capture images
    # ------------------------------------------
    print("Capturing images...")
    timer = Timer()
    timer.start()
    image_dict, timestamp_dict = sync.get_image_serial()
    total_time = timer.stop()

    if not image_dict:
        print("[ERROR] No images were returned by the cameras.")
    else:
        print("[INFO] Images captured successfully.")

    # ------------------------------------------
    # 6) Create output folder
    # ------------------------------------------
    folder_name = time.strftime("%Y-%m-%d_%H-%M-%S") + "\\"
    save_path = os.path.join(parent_path_storing_images, folder_name)
    try:
        os.makedirs(save_path, exist_ok=True)
    except OSError as e:
        print(f"[WARNING] Could not create directory {save_path}: {e}")

    # ------------------------------------------
    # 7) Save images and timestamps
    # ------------------------------------------
    path_dict = {
        helios_camera.name: [
            "helios_intensity.png",
            "helios_heatmap.png",
            "helios_3D.ply"
        ],
        flir_camera.name: [
            "flir_thermal.png",
            "flir_visual.png"
        ]
    }

    print("[INFO] Saving images...")
    sync.saveImage(save_path, path_dict, image_dict)

    timestamp_dict["Total time taking images"] = total_time
    writeTrainingSession(os.path.join(save_path, "timestamp.json"), timestamp_dict)

    # ------------------------------------------
    # 8) Disconnect and clean up
    # ------------------------------------------
    sync.disconnect()
    print("[INFO] Capture session complete.")


if __name__ == '__main__':
    main()
