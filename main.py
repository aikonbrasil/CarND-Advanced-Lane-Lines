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
flag_plot = False
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
if flag_plot == True:
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
if flag_plot == True:
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

###############################################################################
# PIPELINE (thresholding) Describe how (and identify where in your code) you
# used color transforms, gradients or other methods to create a thresholded
# binary image. Provide an example of a binary image result.
###############################################################################

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), d_thresh=(0, np.pi/2), mag_thresh=(0,255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # ********** Technique 1  - Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1


    # ********* Technique 2 - Threshold Direction
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
    abs_sobely = np.absolute(sobely) # Absolute y derivative to accentuate lines away from horizontal
    # Using np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    # Threshold direction gradient
    sxbinary_direction = np.zeros_like(absgraddir)
    sxbinary_direction[(absgraddir >= d_thresh[0]) & (absgraddir <= d_thresh[1])] = 1
    sxbinary_direction = np.uint8(sxbinary_direction)

    # ************Technique 3 -   Threshold color  S channel
    s_binary = np.zeros_like(s_channel)
    s_binary = np.uint8(s_binary)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # **********Technique 4 - Abs_sobelxy
    #  Calculate the magnitude
    abs_sobelxy = np.sqrt(np.square(sobelx)+np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    sxybinary = np.zeros_like(scaled_sobelxy)
    sxybinary[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1


    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary[(s_binary == 1) | (sxbinary == 1) & (sxbinary_direction == 1)] = 1
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    #combined_binary[(s_binary == 1) | (sxbinary == 1) |  (sxybinary == 1)] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = np.dstack((sxybinary, sxbinary, s_binary)) * 255
    return color_binary, combined_binary


if flag_plot == True:
    image = mpimg.imread('test_images/test2.jpg')
    result_mixcolor, result_binary  = pipeline(image, s_thresh=(170, 255), sx_thresh=(20, 100), d_thresh = (0.75, 1.3), mag_thresh=(58,110))

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result_mixcolor)
    ax2.set_title('Color Binary', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    ax3.imshow(result_binary,cmap='gray')
    ax3.set_title('Binary Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()


###############################################################################
# PIPELINE (Perspective Transform) Describe how (and identify where in your code)
# you performed a perspective transform and provide an example of a transformed
# image.
###############################################################################
def corners_unwarp(img1, objpoints, imgpoints, src, dst):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img1, mtx, dist, None, mtx)
    M = cv2.getPerspectiveTransform(src, dst)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
    img_size = gray.shape[::-1]
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    #delete the next two lines
    #M = None
    #warped = np.copy(img1)
    return warped, M

point_0 = [587,452]
point_1 = [691,452]
point_2 = [200,718]
point_3 = [1120,718]
src = np.float32([point_0,point_1,point_2,point_3])
dst = np.float32([[465,0],[920,0],[465,700],[920,700]])
img = mpimg.imread('test_images/straight_lines1.jpg')
undistorted = cal_undistort(img, objpoints, imgpoints)
cv2.line(undistorted, tuple(point_2), tuple(point_0), color=[255,0,0], thickness=2)
cv2.line(undistorted, tuple(point_0), tuple(point_1), color=[255,0,0], thickness=2)
cv2.line(undistorted, tuple(point_1), tuple(point_3), color=[255,0,0], thickness=2)
#plt.imshow(image)
#plt.show()
top_down, perspective_M = corners_unwarp(undistorted, objpoints, imgpoints, src, dst)
if flag_plot == True:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(undistorted)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(top_down)
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
