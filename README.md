
### **Finding Lane Lines on the Road**

The goals / steps of this project are the following:

The first project or challenge of the Udacity Nanodegree Self Driving Car engineer is about  writing a simple
software pipeline to find lanes on the road. The pipeline need to assess the lines finding algorithm created on a series of images provided and, after running the pipeline on some pics taken from a camera, run the find lanes algorithm pipeline for producing a real video with the annotated lines. The project is a great opportunity to learn some basic concepts about image processing,lines and segments extrapolation/interpolation, so forth. The main steps of the pipeline are written in the challenge.py script that is included in this repository.

### 1. Annotaded Frame description. Reflection

The pipeline is mostly based on the function annotatedFrame in the Python script challenge.py;
that script performs all the steps needed to elaborate the frame/image and filter and select
the lanes on the road.

In brief, the first  iteration  of the pipeline consist of the following steps:

1. Filter the images or the frame taken using HSL color space filtering. This is particulary important
   to distinguish and select the yellow and white lines from the "outside" world.
2.  Convert the image / frame produced to grayscale in order to filter even better the lanes.
3.  Apply the Canny algorithm to the image
4.  Apply and configure and tune  the Hough transform to the frame produced in the steps 1, 2 and 3
5.  Select the region of interest for that filtering process (we can image to select a region of interest
   coherent with the image taken from a camera mounted on the top of the car).
6.  Draw the lines filtered after the application of the steps explained and then merge the images; the result will be a "sum" of the original image and the image with the lines produced by the algorithm.

On this first iteration of the pipeline, the draw_lines method was modified to better raster the lines on
the screen. I followed an original approach for that problem and I decided to use the Bresenham algorithm to
draw the lines:  imported in the Python script a generator for producing a tuple of
points accordlyng to the Bresenham algorithm, the lines are then drawn in the end using a polyfit from Matplotlib.

---


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]

---

### 2. Potential shortcomings with the current pipeline

Important shortcomings regards the improvement of the lanes filtering and fine tuning of the
Canny and Hough algorithms for isolate the lanes in all possible weather conditions:
when the images are taken in rainy days or not in sunny conditions, the lanes finding could
be hard and create some side effects (as result , we can have strange lines drawn on the screen
or not coherent with the road lines contours).

The draw lines actually implemented in this first iteration is at now  time to time
producing some additional lines not coherent with the contours of the lines road.

---

### 3. Possible improvements to the pipeline

A first improvement  to the pipeline, in a next iteration of the script,  would be a better implementation of the draw_lines function : this is order to select and delete all the not coherent lines . Other hands , another possible 
improvement migth be to choose a different polinomial approximation of the lines drawn by the Matplotlib polyfit (change or modify the algorithms used, increase the degrees, so on). Better fine tuning of the parameter of Hough transform and Canny algorithms can be also taken in account to improve the line finding in general. 

---

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]

