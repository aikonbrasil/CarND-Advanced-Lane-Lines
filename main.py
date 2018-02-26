import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

###############################################################################
def cal_undistort(img, objpoints, imgpoints):
    img1 = np.copy(img)
    #gray = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    #ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    #img1 = cv2.drawChessboardCorners(img1, (9,6), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img1, mtx, dist, None, mtx)
    return undist

###############################################################################

###############################################################################
# CAMERA CALIBRATION - script
##############################################################################
# Prepare object points
nx = 9 # Number of inside corners in x
ny = 6 # Number of inside corners in y

# Read in and make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
#fname = 'camera_cal/calibration3.jpg'
#img = cv2.imread(fname)

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... ,(7,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x, y coordinates

for fname in images:
    # Read each images
    img = mpimg.imread(fname)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        #print('ENTROU NO IF')
        # Appending data for imgpoints and objpoints arrays
        imgpoints.append(corners)
        objpoints.append(objp)

        # draw and display the corners
        #img_corners_detected = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        #plt.imshow(img_corners_detected)
        #plt.show()

img = mpimg.imread('camera_cal/calibration5.jpg')
undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

###############################################################################
#  PIPELINE (Test images) - Provide an example of a distortion-corrected images
##############################################################################

# Loading a test image
img = mpimg.imread('test_images/straight_lines2.jpg')
undistorted = cal_undistort(img, objpoints, imgpoints)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
