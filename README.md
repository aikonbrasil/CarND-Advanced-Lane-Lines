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
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
