## Writeup Project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1_distort]: ./output_images/calibration_result_1.png "Undistorted_1"
[image2_distort]: ./output_images/calibration_result_2.png "Undistorted_1"
[image1_undistort]: ./output_images/distortion_corrected_0.png "Test Distortion Corrected 1"
[image2_undistort]: ./output_images/distortion_corrected_1.png "Test Distortion Corrected 2"
[image1_threshold]: ./output_images/gradient_threshold_0.png "Threshold result"
[image1_warped]: ./output_images/wraped_0.png "Warped 1"
[image2_warped]: ./output_images/wraped_1.png "Warped 2"
[image3_warped]: ./output_images/wraped_2.png "Warped 3"
[image4_warped]: ./output_images/wraped_3.png "Warped 4"
[image5_warped]: ./output_images/wraped_4.png "Warped 5"
[image6_warped]: ./output_images/wraped_5.png "Warped 6"
[image7_warped]: ./output_images/wraped_6.png "Warped 7"
[image1_lane_line]: ./output_images/lane_line_pixels_1.png "Lane line Example 1"
[image2_lane_line]: ./output_images/lane_line_pixels_2.png "Lane line Example 2"
[image3_lane_line]: ./output_images/lane_line_pixels_3.png "Lane line Example 3"
[image4_lane_line]: ./output_images/lane_line_pixels_4.png "Lane line Example 4"
[image5_lane_line]: ./output_images/lane_line_pixels_5.png "Lane line Example 5"
[image6_lane_line]: ./output_images/lane_line_pixels_6.png "Lane line Example 6"
[image7_lane_line]: ./output_images/lane_line_pixels_7.png "Lane line Example 7"
[image8_lane_line]: ./output_images/lane_line_pixels_8.png "Lane line Example 8"
[video1]: ./outputvideo.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 22 through 71 of the file called `main.py`.  

I star by defining the number of corners of calibration images, in this case nx=9 and ny=6. After that I read in and make a list of calibration images. The next step was to create a two empty arrays, objpoints and imgpoints to append "object points" (3D points in real world space) and "image points" (2D points in image plane) respectively.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. These corners are defined using `cv2.findChessboardCorners` function, this function also used a gray scale image of each calibration chessboard images.

I then created and used the function `cal_undistort` that is based on `cv2.calibrateCamera()` and `cv2.undistort() ` functions.  I applied this distortion correction to the test image and obtained this result:

![alt text][image1_distort]

In the next Figure is easily to check the calibration effect on the chessboard image.

![alt text][image2_distort]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I used the arrays `imgpoints` and `objpoints` obtained in the Camera Calibration step. I also reuse the function `cal_undistort`. This step could be identified in between lines 78 and 87 in the file called `main.py`. The result of one of the test images looks like this one:
![alt text][image1_undistort]

Specifically focusing on the image edge, in which the effect of the distortion is more evident.
![alt text][image2_undistort]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used created some binaries images based Sobel X gradient , Sobel Y gradient , Magnitude gradient, Gradient direction (using the `arctan2` function), Threshold color on S Channel from HLS color space (Channel L and S used in this case), and combinations of the techniques described before. After try many combinations, the best choose was the combined binary of Threshold channel S color binary + Sobel X gradient binary. The code used to apply the threshold techniques could be localized since the line 96 until 168 in `main.py`.

In the following image is easily to distinguish three differente images, the first one is the original figure that was previousbly undistorted, Over this image were applied the Threshold techniques. In the Color Binary is possible to identify the threshold contribution of three differente binary images (Red for Magnitude gradient, Sobel x gradient, and the S color space). The Binary result shows the final result of applying techniques described before.

![alt text][image1_threshold]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `corners_warper()`, which appears in lines 176 through 215 in the file `main.py` (output_images/examples/example.py).  The `corners_warper()` function takes as inputs an image (`img1`), as well as Object_Points(`objpoints`), Image_points (`imgpoints`), source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 587, 452      | 465, 0        |
| 691, 452      | 920, 0      |
| 200, 718     | 465, 700      |
| 1120, 718      | 920, 700        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. It was tested in all test images provided in the `test_images` folder

![alt text][image1_warped]
![alt text][image2_warped]
![alt text][image3_warped]
![alt text][image4_warped]
![alt text][image5_warped]
![alt text][image6_warped]
![alt text][image7_warped]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
Since the line 293 begins the code of `findng_lines_slidingwindow()` function. In this function we calculate the histogram in axis=0 (X) and calculated the main peak of the first half of the first 640 pixels in axis x and the main peak of the second half of histogram. After an analysis we observed that the Peaks that are interested for this project are that peaks that are near the center in both cases. So, I decided to restring the localization of these peaks between 400 and 1060 pixels (check line 304 and 305 of main.py file). After defined the peaks, we define initial point to search pixels in left and right side. To perform pixels search, I used the Udacity code, in this code is defined a window with margin and minpix parameters which are used to define the size of the window and the criteria to look for a new direction, respectively. Other parameters is the number of windows to use in vertical direction. Every window check the pixels that are inside it, the search is performed vertically, defining a line based on the pixels that were localized. These informations are concatenated in numpy array called `left_lane_inds` for left line and `right_lane_inds` for right line.  Based on these information the code uses the `polyfit` function to define the parameters of the 2nd order curve defined by A, B, and C in the polynomial equation: `(f(y)=A y**2 + B y + C)`.

As suggested by Udacity I generated other function called `finding_lines_targeted()` (since line 402 in main.py file), in which I take advantage of the information of the windowing solution to find lines without apply any histogram, just using the previous information to localize lines (an optimized search of lines).

Described images were shared in section 6.
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
 As complement the radius were calculated using the relation defined by `L_curve =  = ((1 + (2*A*y+ B)**2)**1.5) / abs(2*A)`. It was useful extracting values of vectors: `left_fit` and `right_fit` with its respective value in meters. Vectors mentioned before contain A, B, and C parameters. It could be checked in the line 392 of the file `main.py`

 The localization of the vehicle with respect to center was done using code lines 618 until 623 of `main.py` file. For this pourpose we calculated the mean values of indexes of `left_lane_inds` and `right_lane_inds` vectors, everyone representing the position of left and right lines. This positions were summed and divided by 2 in order to obtain the center calculated based on line find technique. The previous value was compared with the center of the image that is equal to 640, The difference between both values described before represents the offset in pixels. Finally I applied the conversions to meters using the rate 3.7/700.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Since the line code number 234 until 534 in `main.py` file, I implemented functions that helped to get lane finding algorithm. Here I defined the next functions: `list_of_peaks()`, `histogram_analise`, `findng_lines_slidingwindow()`, `findng_lines_targeted`. Here is an example of my result on a test image:

![alt text][image1_lane_line]
.
.
.

![alt text][image2_lane_line]
.
.
.

![alt text][image3_lane_line]
.
.
.

![alt text][image4_lane_line]
.
.
.

![alt text][image5_lane_line]
.
.
.

![alt text][image6_lane_line]
.
.
.

![alt text][image7_lane_line]
.
.
.

![alt text][image8_lane_line]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I generated the output video requested in this project. I added the text with radius and offset distance of the car to the center. If you check, It is possible to check the lines detected in red color for left lines and blue color for the right lines. The Radius is extrapolated when the car is in zones without curves. In curve zones, the radius is approximately 1000m.

Here's a [link to my video saved in Youtube](https://youtu.be/9C8vRQcIHxo)
It could be also acceced in the repository, it was saved with the name `outputvideo.mp4`


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have inserted some global variables to perform some adaptative configuration parameters in threshold algorithm code. I mean, in specific scenarios these values changed based on distribution of pixels in the histogram. It is possible that my solution might fail in night conditions.
