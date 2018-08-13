#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

def gettingBinaryImage(path) :

    # Open the image in gray scale TODO Applying filters in order to get a clean image  
    image = cv2.imread(path, 0)

    # Applying thresholding
    ret, thresholedImage = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Inverting the Binary image
    thresholedImage = cv2.bitwise_not(thresholedImage)

    # Applying Morphological Operations
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresholedImage, cv2.MORPH_OPEN, kernel)
    #opening = cv2.morphologyEx(thresholedImage, cv2.MORPH_CLOSE, kernel)

    dil = cv2.dilate(opening,kernel,iterations = 2)

    '''
    Applying SIFT in order to obtain the key points of the signature
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints = sift.detect(dil,None)

    img = cv2.drawKeypoints(dil,keyPoints,image)

    return img

image = "signature.png"
thresholedImageObtained = gettingBinaryImage(image)

cv2.imshow('image',thresholedImageObtained)
cv2.waitKey(0)
cv2.destroyAllWindows()

