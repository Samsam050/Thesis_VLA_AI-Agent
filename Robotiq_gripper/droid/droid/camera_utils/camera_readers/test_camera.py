import time

import cv2
import numpy as np

from droid.misc.parameters import hand_camera_id, varied_camera_1_id
from get_camera import OakCamera, RealSenseCamera

try:
    from zed_camera import gather_zed_cameras
except ImportError:
    gather_zed_cameras = None


def waiting_frame(width=640, height=480, text="WAITING FOR FRAME..."):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (30, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )
    return img


def main():
    print("Initializing cameras...")

    oak_cam = None
    try:
        oak_cam = OakCamera(hand_camera_id)
        oak_cam.set_calibration_mode()
    except Exception as e:
        print(f"Failed to initialize OAK camera: {e}")
        oak_cam = None

    rs_cam = RealSenseCamera(varied_camera_1_id)
    rs_cam.set_calibration_mode()

    zed_cam = None
    zed_debug_printed = False

    if gather_zed_cameras is None:
        print("ZED camera helper not available in get_camera.py")
    else:
        try:
            zed_cams = gather_zed_cameras()
            if len(zed_cams) == 0:
                print("No ZED cameras found.")
            else:
                zed_cam = zed_cams[0]
                print(f"Using ZED camera: {zed_cam.serial_number}")
                if len(zed_cams) > 1:
                    print(f"Found {len(zed_cams)} ZED cameras, using the first one.")

                zed_cam.set_reading_parameters(
                    image=True,
                    depth=False,
                    pointcloud=False,
                    concatenate_images=False,
                    resolution=(0, 0),
                    resize_func=None,
                )
                zed_cam.set_trajectory_mode()
        except Exception as e:
            print(f"Failed to initialize ZED camera: {e}")
            zed_cam = None

    time.sleep(2)
    print("Streaming... Press 'q' to quit.")

    if oak_cam is not None:
        cv2.namedWindow("OAK-D", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("OAK-D", 960, 540)

    cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RealSense", 960, 540)

    if zed_cam is not None:
        cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ZED", 960, 540)

    while True:
        if oak_cam is None:
            oak_left = waiting_frame(text="NO OAK FOUND...")
        else:
            oak_result = oak_cam.read_camera()
            if oak_result is None:
                oak_left = waiting_frame(text="WAITING FOR OAK...")
                oak_right = oak_left.copy()
            else:
                oak_data, _ = oak_result
                oak_left = oak_data["image"][oak_cam.serial_number + "_left"]

        rs_result = rs_cam.read_camera()
        if rs_result is None:
            rs_frame = waiting_frame(text="WAITING FOR REALSENSE...")
        else:
            rs_data, _ = rs_result
            rs_frame = rs_data["image"][rs_cam.serial_number + "_left"]

        if zed_cam is None:
            zed_frame = waiting_frame(text="NO ZED FOUND...")
        else:
            zed_result = zed_cam.read_camera()
            if zed_result is None:
                zed_frame = waiting_frame(text="WAITING FOR ZED...")
            else:
                zed_data, _ = zed_result
                zed_left_key = zed_cam.serial_number + "_left"

                if zed_left_key in zed_data["image"]:
                    zed_frame = zed_data["image"][zed_left_key]
                else:
                    zed_frame = next(iter(zed_data["image"].values()))

                if not zed_debug_printed:
                    intrinsics = zed_cam.get_intrinsics()
                    print("ZED intrinsics keys:", intrinsics.keys())
                    print("ZED first cameraMatrix:\n", next(iter(intrinsics.values()))["cameraMatrix"])
                    print("ZED image keys:", zed_data["image"].keys())
                    zed_debug_printed = True

        if oak_cam is not None:
            cv2.imshow("OAK-D", oak_left)
        cv2.imshow("RealSense", rs_frame)

        if zed_cam is not None:
            cv2.imshow("ZED", zed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if oak_cam is not None:
        oak_cam.disable_camera()
    rs_cam.disable_camera()

    if zed_cam is not None:
        zed_cam.disable_camera()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()