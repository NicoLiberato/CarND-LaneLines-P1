#!/usr/bin/env python3

import os
import statistics
from statistics import mean 
import logging
import datetime
import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from bresenham import bresenham
from numpy.polynomial import Chebyshev as T

def filterLanes(img):    
    retImg = np.copy(img)
    cv2.cvtColor(retImg,cv2.COLOR_RGB2HLS)
    w_filter = cv2.inRange(retImg, np.uint8([  0, 205,   0]), np.uint8([255, 255, 255]))
    y_filter = cv2.inRange(retImg, np.uint8([ 12,   0, 100]),  np.uint8([ 35, 255, 255]))
    final_mask = cv2.bitwise_or(w_filter, y_filter)
    return cv2.bitwise_and(retImg, retImg, mask = final_mask)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size), 0)
    return cv2.cvtColor(blur_gray, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
     lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
     return line_img
 
def region_of_interest(img, vertices):
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
 
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
     # Iterate over the output "lines" and draw lines on a blank imag  
     for line in lines:
         for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(233,233,0),10) 

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
     return cv2.addWeighted(initial_img, α, img, β, γ)
     
def draw_lines_inter(img, lines, color=[255, 0, 0], thickness=2):
     # Iterate over the output "lines" and draw lines on a blank imag  
    x=[]
    y=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            a = list(bresenham(x1, y1, x2, y2)) 
            xb = [x for (x,y) in a ]        
            yb = [y for (x,y) in a ]             
            x += xb
            y += yb 
            #z = T.fit(xb, yb, 4)
            z = np.polyfit(xb, yb, 1)
            f = np.poly1d(z)
            m, b = z
                           
            for i in range(min(x),max(x)):
               cv2.line(img,(0,int(b)),(int(i), int(i * m + b)),(233,233,0),8)    

def annotadedFrame(image):
    img = mpimg.imread(image)
    copyImg = filterLanes(img)
    gray = grayscale(copyImg)
    edges = canny(gray,112,255) 
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(copyImg)*0 # creating a blank to draw lines on
   
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                             min_line_length, max_line_gap)
        
    line_img = np.zeros((copyImg.shape[0], copyImg.shape[1], 3), dtype=np.uint8) 
         
    color_edges = np.dstack((edges, edges, edges)) 
       
    draw_lines_inter(color_edges,lines,(255,233,0),6)
        
    imshape = copyImg.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    selected = region_of_interest(color_edges,vertices)
    result = weighted_img(img,selected,0.6,1,0.0)
    return result

def annotadedFrameVideo(img):
    copyImg = filterLanes(img)
    gray = grayscale(copyImg)
    edges = canny(gray,112,255) 
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(copyImg)*0 # creating a blank to draw lines on
   
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                             min_line_length, max_line_gap)
        
    line_img = np.zeros((copyImg.shape[0], copyImg.shape[1], 3), dtype=np.uint8) 
         
    color_edges = np.dstack((edges, edges, edges)) 
       
    draw_lines_inter(color_edges,lines,(255,233,0),6)
        
    imshape = copyImg.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    selected = region_of_interest(color_edges,vertices)
    result = weighted_img(img,selected,0.6,1,0.0)
    return result

def main():
    print(" --- find lane roads")
    # List all subdirectories using os.listdir
    # basepath = 'test_images/'
    # images = []
    # for filename in os.listdir(basepath):
    #     img = annotadedFrame(os.path.join(basepath, filename))
    #     if img is not None:
    #         images.append(img)
    #         #use this for show the result after applyng the hough
    # for img in images:
    #     mgplot = plt.imshow(img)
    #     plt.show()
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('test_videos_output/output.mp4', fourcc, 20.0, (490,290))
    cap = cv2.VideoCapture('test_videos/solidYellowLeft.mp4')
    
    while(cap.isOpened()):
          ret, frame = cap.read()
          newFrame = annotadedFrameVideo(frame) 
          wFrame = cv2.resize(newFrame, (490, 290), fx = 0, fy = 0, 
                         interpolation = cv2.INTER_CUBIC)
          cv2.imshow('frame', wFrame)
          #frame = cv2.flip(frame,0)
          out.write(wFrame)
         
          if cv2.waitKey(1) & 0xFF == ord('q'):
               break

    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == "__main__":
    main()

