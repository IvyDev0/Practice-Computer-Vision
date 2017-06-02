# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:42:51 2016

@author: Gaurav.Hegde
"""

import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


cap = cv2.VideoCapture('test.mov')
cap.set(0, 0)
"""while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (ret==1):
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()"""
# take first frame of the video
ret, frame = cap.read()


# setup initial location of window
"""points = open("points_detect_"+str(1)+".txt","r")
all_points = points.read().split(',')
for i in range(len(all_points)-1) :
 all_points[i]=int(all_points[i])"""


# setup initial location of window
# c,r,w,h = all_points[0],all_points[1],all_points[2],all_points[3]  #
# simply hardcoded the values - CHANGE TO DETECTED VALUES
width = frame.shape[1]
height = frame.shape[0]
#c, r, w, h = 100, 200, 20, 46
#c, r, w, h = 40, 110, 20, 30
c, r, w, h = 366, 126, 18, 35
#c, r, w, h = 50, 200, 20, 46
#c, r, w, h = 149, 200, 20, 46
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


img = cv2.rectangle(frame, (c, r), (c + w, r + h), 255, 2)
# imgray = cv2.cvtColor(hsv_roi,cv2.COLOR_BGR2GRAY)

# mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
#                   np.array((100., 255., 255.)))
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                   np.array((30., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image

        x, y, w, h = track_window
        # print x, y, w, h
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        # What is the above line doin ????????

        cv2.imshow('img2', frame)


# Why is the below code needed ?
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".png", img2)
    else:
        break

cap.release()

# cv2.destroyAllWindows()
