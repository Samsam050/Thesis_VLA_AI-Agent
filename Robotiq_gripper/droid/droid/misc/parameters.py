import os
from cv2 import aruco

# Robot Params #
nuc_ip = "192.168.1.2"
robot_ip = "172.16.0.101"
laptop_ip = ""
sudo_password = "frankanuc01"
robot_type = "panda"  # 'panda' or 'fr3'
robot_serial_number = ""

# Camera ID's #
hand_camera_id = "19443010513EA12E00"
#hand_camera_id = "19892622"
#hand_camera_id= ""
varied_camera_1_id = "242422301956"
varied_camera_2_id = "19443010C14F9D2E00"

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = ""

# Code Version [DONT CHANGE] #
droid_version = "1.3"

