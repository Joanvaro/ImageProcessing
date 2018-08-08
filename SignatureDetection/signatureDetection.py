#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

def gettingBinaryImage(path) :

    # Open the image in gray scale TODO Applying filters in order to get a clean image  
    image = cv2.imread(path, 0)

    # Applying thresholding
    ret, thresholedImage = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Drawing Contours
    im2, contours, hierarchy = cv2.findContours(thresholedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresholedImage, contours, -1, (0,255,0), 3)

    # Inverting the Binary image
    thresholedImage = cv2.bitwise_not(thresholedImage)

    '''
    Applying SIFT in order to obtain the key points of the signature
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints = sift.detect(thresholedImage,None)

    img = cv2.drawKeypoints(thresholedImage,keyPoints,image)

    return img


# TODO k means implementation but first check that this code is working 

image = "signature.png"
thresholedImageObtained = gettingBinaryImage(image)

cv2.imshow('image',thresholedImageObtained)
cv2.waitKey(0)
cv2.destroyAllWindows()

