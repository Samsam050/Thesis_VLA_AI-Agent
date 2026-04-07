import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
col=7
row=11
objp = np.zeros((row*col,3), np.float32)
objp[:,:2] = np.mgrid[0:col,0:row].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    flags = (cv.CALIB_CB_ADAPTIVE_THRESH +
             cv.CALIB_CB_NORMALIZE_IMAGE +
             cv.CALIB_CB_FAST_CHECK)

    ret, corners = cv.findChessboardCorners(gray, (col, row), flags)

    debug = gray.copy()

    cv.drawChessboardCorners(debug, (col, row), corners, ret)

    cv.imshow("debug", debug)
    cv.waitKey(0)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (col,row), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print("error")

cv.destroyAllWindows()