#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

def gettingBinaryImage(path) :

    # Open the image in gray scale
    #image = cv2.imread(path, CV_LOAD_IMAGE_GRAYSCALE)
    image = cv2.imread(path, 0)

    # Applying thresholding
    ret, thresholedImage = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #return thresholdingImage
    '''
    This must be removed after the debugging
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints = sift.detect(thresholedImage,None)

    img = cv2.drawKeypoints(thresholedImage,keyPoints,image)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gettingKeyPoints(image) :

    # Applying SIFT to the binary image
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints = sift.detect(image,None)

    img = cv2.drawKeypoints(image,kp)

    return img

# TODO k means implementation but first check that this code is working 

image = "signature.png"
#thresholingImage = gettingBinaryImage(image)
gettingBinaryImage(image)
#img = gettingKeyPoints(thresholingImage)

