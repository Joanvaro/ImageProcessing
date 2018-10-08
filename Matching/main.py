#!/usr/bin/python

import cv2
import sys
import numpy as np

def harris_corner(image):
    image = np.float32(image)
    dst = cv2.cornerHarris(image,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    #define criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image,np.float32(centroids),(5,5),(-1,-1),criteria)

    return corners, centroids

def feature_extractor(image, featureType):
    if( featureType == 'sift' ):
        feature = cv2.xfeatures2d.SIFT_create()
    elif( featureType == 'surf' ):
        feature = cv2.xfeatures2d.SURF_create()
    elif( featureType == 'orb' ):
        feature = cv2.ORB_create()
    elif( featureType == 'harris' ):
        return harris_corner(image)
    else:
        exit(1)

    return feature.detectAndCompute(image, None)

def matching_image(img1, kp1, kp2, desc1, desc2, outputImage):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)

    matching_result = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None, flags=2)

    cv2.imwrite(outputImage, matching_result)
    cv2.imshow('Image',matching_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imageName1 = str("img"+sys.argv[1]+".jpg")
imageName2 = str("img"+sys.argv[2]+".jpg")

img1 = cv2.imread(imageName1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(imageName2, cv2.IMREAD_GRAYSCALE)

kp1,desc1 = feature_extractor(img1, sys.argv[3])
kp2,desc2 = feature_extractor(img2, sys.argv[3])

outputFile = str(sys.argv[1]+"_vs_"+sys.argv[2]+"_"+sys.argv[3]+".jpg")
print outputFile
matching_image(img1,kp1,kp2,desc1,desc2,outputFile)

