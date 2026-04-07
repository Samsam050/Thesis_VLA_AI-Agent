import cv2
import cv2.aruco as aruco

# --- DROID PARAMETERS (From your config) ---
# Rows and Columns
rows = 9
cols = 14

# Sizes in meters (converted to drawing scale)
square_size = 0.020
marker_size = 0.016

# The specific dictionary DROID uses
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# --- GENERATE IMAGE ---
print(f"Generating {cols}x{rows} Charuco Board...")
board = aruco.CharucoBoard_create(cols, rows, square_size, marker_size, dictionary)

# Draw it at high resolution for printing (2000x1400 pixels)
img = board.draw((2000, 1400)) 

# Save
filename = "droid_charuco_9x14.png"
cv2.imwrite(filename, img)
print(f"Done! Saved to {filename}")
