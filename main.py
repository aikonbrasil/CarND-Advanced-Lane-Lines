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
    img = mpimg.imread('test_images/test2.jpg')
    image = cal_undistort(img, objpoints, imgpoints)
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
images = glob.glob('test_images/*.jpg')
for frame in images:
    img = mpimg.imread(frame)
#img = mpimg.imread('test_images/straight_lines1.jpg')
    undistorted = cal_undistort(img, objpoints, imgpoints)
    cv2.line(undistorted, tuple(point_2), tuple(point_0), color=[255,0,0], thickness=2)
    cv2.line(undistorted, tuple(point_0), tuple(point_1), color=[255,0,0], thickness=1)
    cv2.line(undistorted, tuple(point_1), tuple(point_3), color=[255,0,0], thickness=2)
    cv2.line(undistorted, tuple(point_2), tuple(point_3), color=[255,0,0], thickness=2)
    #plt.imshow(image)
    #plt.show()
    top_down, perspective_M = corners_unwarp(undistorted, objpoints, imgpoints, src, dst)
    #flag_plot = True
    if flag_plot == True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undistorted)
        ax1.set_title('Undistorted Image', fontsize=30)
        ax2.imshow(top_down)
        ax2.set_title('Warped Image - bird´s eye view', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

###############################################################################
# PIPELINE (lane line pixels) Describe how (and identify where in your code) you
# identified lane-line pixels and fit their positions with a polynomial?
###############################################################################
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def list_of_peaks(hist, maxrange):
    #Convert histogram to simple list
    #hist = [val[0] for val in hist];

    #Generate a list of indices
    indices = list(range(0, maxrange));

    #Descending sort-by-key with histogram value as key
    s = [(x,y) for y,x in sorted(zip(hist,indices), reverse=True)]

    #Index of highest peak in histogram
    # Peaks over a threshold
    index_of_highest_peak = s[0][0]
    for ii in range(0,maxrange):
        if (s[ii][1] > 100) & (index_of_highest_peak - s[ii][0] > 20):
            print('pontos com maior pico')
            print(s[ii][0])
    index_of_highest_peak = s[0][0]
    counter=0
    diferencia = 0
    #while diferencia < 100:
    #    counter = counter +1
    #    diferencia = s[0][1] - s[counter][1]
    #    print(diferencia)


    #Index of second highest peak in histogram
    #index_of_second_highest_peak = s[1][0];
    index_of_second_highest_peak = s[counter][0]
    #print(s)

    return index_of_highest_peak, index_of_second_highest_peak

def findng_lines(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    plt.plot(histogram)
    plt.show()
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    print('Pico raw izquerda')
    print(leftx_base)

    peak_1, peak_2 = list_of_peaks(histogram[:midpoint],midpoint)
    print('1er pico:')
    print(peak_1)
    print('2nd pico:')
    print(peak_2)


    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_lane_inds, right_lane_inds, left_fit, right_fit, out_img, nonzeroy, nonzerox

images = glob.glob('test_images/*.jpg')
#images = glob.glob('test_images/test5.jpg')
for frame in images:
    img = mpimg.imread(frame)
    image = cal_undistort(img, objpoints, imgpoints)

    point_0 = [587,452]
    point_1 = [691,452]
    point_2 = [200,718]
    point_3 = [1120,718]

    #vertices = np.array([[(200-100,718),(587-50, 452), (691+50, 452), (1120+100,718)]], dtype=np.int32)
    #masked_edges = region_of_interest(image, vertices)
    #image = masked_edges
    print(frame)
    #image = mpimg.imread('test_images/test1.jpg')
    # Applying Threshold
    result_mixcolor, result_binary  = pipeline(image, s_thresh=(170, 255), sx_thresh=(20, 100), d_thresh = (0.75, 1.3), mag_thresh=(58,110))
    #result_mixcolor, result_binary  = pipeline(image, s_thresh=(150, 255), sx_thresh=(20, 90), d_thresh = (0.75, 1.3), mag_thresh=(58,110))
    plt.imshow(result_binary,cmap='gray')
    plt.show()
    plt.imshow(result_mixcolor)
    plt.show()
    # Applying unwarp function
    binary_warped, perspective_M = corners_unwarp(result_binary, objpoints, imgpoints, src, dst)
    plt.imshow(binary_warped,cmap='gray')
    plt.show()
    # Applying sliding windows and Fit a Polynomial
    left_lane_inds, right_lane_inds, left_fit, right_fit, out_img, nonzeroy, nonzerox = findng_lines(binary_warped)

    # ***********visualization
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
