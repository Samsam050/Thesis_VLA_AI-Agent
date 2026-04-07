import depthai as dai
import pyrealsense2 as rs

# 1. Get OAK-D Serial (Wrist?)
print("--- OAK-D DEVICES ---")
for device in dai.Device.getAllAvailableDevices():
    print(f"ID: {device.getMxId()}  (State: {device.state})")

# 2. Get RealSense Serial (Side?)
print("\n--- REALSENSE DEVICES ---")
ctx = rs.context()
for dev in ctx.query_devices():
    print(f"Name: {dev.get_info(rs.camera_info.name)}")
    print(f"Serial: {dev.get_info(rs.camera_info.serial_number)}")