This program is for road detection and tracking model for an autonomous aerial vehicle.
The original code was for line tracking for a self-driving car and I modified it into road detection and tracking model for a aerial vehicle. 
By using this model we can put an UAV inside the area of a road and not go outside into a private property.

# Camera calibration
Calibration is a process that compare the devices measurement to standard measurements and understand the acuracy of the device. 
In this model, we are using check board images to calibrate the camera module. 
In ths calbration1.jpg, we can see how a camera infront of a vehicle will capture the road.
in the real view, we can say that the z=0 in x.y.z matrix.
so, we must calibrate the camera view just as the real view.

# combined threshold 
this file is about the combination of colour and gradient thresholds for generating binary images.
(line 81 there is a debug at the moment. will try to solve it)

# perspective transform
this python is created to execute the function of perspective transform.

# generating example images
this python script is to generate example images to illustrate different pipeline stages outputs.
The source and destination points were hardcoded in the following manner:

`
src = np.float32([[180, img.shape[0]], [575, 460], [705, 460], [1150, img.shape[0]]])
`

`
dst = np.float32([[320, img.shape[0]], [320, 0], [960, 0], [960, img.shape[0]]])
`
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 180, 720      | 320, 720      | 
| 575, 460      | 320, 0        |
| 705, 460      | 960, 0        |
| 1150, 720     | 960, 720      |

Draw the `src` and `dst` points onto a test image and its warped counterpart to verify that perspective transform works as expected.

# Line
line is the python script to track the road line in a live or recorded video.
this will be implementtd in to testing data. 
n -window size of moving average.

# line fit 
line fit is the main python script for finding and fit lanes in a testing video.