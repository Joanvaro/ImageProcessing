#!/usr/bin/python 

import cv2 as cv
import numpy as np

img = cv.imread('img11.jpg')
res = cv.resize(img,None,fx=0.125,fy=0.125,interpolation=cv.INTER_CUBIC)

cv.imwrite('img11_1.jpg',res)
