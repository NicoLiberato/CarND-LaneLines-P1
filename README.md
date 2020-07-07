
### **Finding Lane Lines on the Road**

The goals / steps of this project are the following:

The first project or challenge for the Udacity Nanodegree Self Driving Car engineer consist on writing a simple
pipeline to find lanes on the road. The pipeline need to assess the lines finding algorithm on a series of
images provided and then, after running the pipeline on some pics taken from a camera, run the find lanes algorithm
for producing a real video. The project is a big opportunity to learn some basic concepts about image processing,
lines and segments extrapolation, so forth. The main steps of the pipeline are designed in the challenge.py script
included in this repository.

### 1. Annotaded Frame description. Reflection

The pipeline is using the function annotatedFrame in the Python script ; that script performs
all the steps needed to elaborate the frame/image and then filter and recognize the lanes on the road.

My first iteration  of the pipeline consist of the following steps:

a) Filter the images or the frame taken using HSL color space filtering. This is particulary important
   to filter the yellow and white lines from the "outside" world.
b) Convert the image / frame produced to grayscale in order to filter even better the lanes.
c) Apply the Canny algorithm to the image
d) Apply and configure and tune  the Hough transform to the frame produced in the steps a), b) and c)
e) Select the region of interest for that filtering process (we can image to select a region of interest
   coherent with the image taken from a camera mounted on the top of the car).
f) Draw the lines filtered after the application of the steps explained and then merge the images; the result will be a "sum" of the original image and the image with the lines produced by the algorithm.

On this first iteration of the pipeline, the draw_lines method was modified to better raster the lines on
the screen. I followed an original approach for that problem and I decided to use the Bresenham algorithm to
draw the lines:  imported in the Python script a generator for producing a tuple of
points accordlyng to the Bresenham algorithm, the lines are then drawn using a polyfit from Matplotlib.




[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Potential shortcomings within the  current pipeline

Important shortcomings regards the improvement of the lanes filtering and fine tuning of the
Canny and Hough algorithms for isolate the lanes in all possible weather conditions:
when the images are taken in rainy days or not in sunny conditions, the lanes finding could
be hard and create some side effects (as result , we can have strange lines drawn on the screen
or not coherent with the road lines contours).

The draw lines actually implemented in this first iteration is actually time to time
producing some additional lines not aligned with the contours of the lines road.


### 3. Possible improvements to your pipeline

A first improvement  to the pipeline, in a next iteration of the script,  would be a better implementation of the draw_lines function : this is order to select and delete all the not coherent lines drawn. Another possible 
improvement migth be to choose a different polinomial approximation of the lines made by the Matplotlib polyfit (change or modify the algorithms used, increase the degrees, so on).





[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Potential shortcomings within the  current pipeline

Important shortcomings regards the improvement of the lanes filtering and fine tuning of the
Canny and Hough algorithms for isolate the lanes in all possible weather conditions:
when the images are taken in rainy days or not in sunny conditions, the lanes finding could
be hard and create some side effects (as result , we can have strange lines drawn on the screen
or not coherent with the road lines contours).

The draw lines actually implemented in this first iteration is actually time to time
producing some additional lines not aligned with the contours of the lines road.


### 3. Possible improvements to your pipeline

A first improvement  to the pipeline, in a next iteration of the script,  would be a better implementation of the draw_lines function : this is order to select and delete all the not coherent lines drawn. Another possible 
improvement migth be to choose a different polinomial approximation of the lines made by the Matplotlib polyfit (change or modify the algorithms used, increase the degrees, so on).


# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).The code file is called P1.ipynb and the writeup template is writeup_template.md 

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)


Creating a Great Writeup
---
For this project, a great writeup should provide a detailed response to the "Reflection" section of the [project rubric](https://review.udacity.com/#!/rubrics/322/view). There are three parts to the reflection:

1. Describe the pipeline

2. Identify any shortcomings

3. Suggest possible improvements

We encourage using images in your writeup to demonstrate how your pipeline works.  

All that said, please be concise!  We're not looking for you to write a book here: just a brief description.

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup. Here is a link to a [writeup template file](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md). 


The Project
---

## If you have already installed the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) you should be good to go!   If not, you should install the starter kit to get started on this project. ##

**Step 1:** Set up the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) if you haven't already.

**Step 2:** Open the code in a Jupyter Notebook

You will complete the project code in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out [Udacity's free course on Anaconda and Jupyter Notebooks](https://classroom.udacity.com/courses/ud1111) to get started.

Jupyter is an Ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, use terminal to navigate to your project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) installation instructions!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  

**Step 3:** Complete the project and submit both the Ipython notebook and the project writeup

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

