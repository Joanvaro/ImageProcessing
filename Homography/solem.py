#!/usr/bin/python

from PIL import Image
from pylab import *
from scipy.ndimage import filters
from scipy import ndimage

def normalize(points):
    for row in points:
        row /= points[-1]
    return point

def make_homog(points):
    return vstack((points,ones((1,points.shape[1]))))

def H_from_points(fp,tp):
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp = dot(C1,fp)

    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    fp = dot(C2,fp)

    nbr_correspondeces = fp.shape[1]
    A = zeros((2*nbr_correspondeces,9))
    for i in range(nbr_correspondeces):
        A[2*i] = [-fp[0][i], -fp[1][i], -1,0,0,0,
                tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
        A[2*i+1] = [0,0,0,-fp[0][i], -fp[1][i],1,
                tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
        U,S,V = linalg.svd(A)
        H = V[8].reshape((3,3))
        
        # decognition
        H = dot(linalg.inv(C2),dot(H,C1))

        # normalize and return
        return H / H[2,2]

def Haffine_from_points(fp,tp):
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[0]/maxstd
    fp_cond = dot(C1,fp)

    m = mean(tp[:2], axis=1)
    C2 = C1.copy()
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)

    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)

    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
    H = vstack((tmp2,[0,0,1]))

    H = dot(linalg.inv(C2),dot(H,C1))

    return H / H[2,2]

def image_in_image(im1,im2,tp):
    # points to wrap from 
    m,n = im1.shape[:2]
    fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

    # compute affine transform and apply
    H = Haffine_from_points(tp,fp)
    im1_t = ndimage.affine_transform(im1,H[:2,:2],
            (H[0,2],H[1,2]),im2.shape[:2])
    alpha = (im1_t > 0)

    return (1-alpha)*im2 + alpha*im1_t

def alpha_for_triangle(points,m,n):
    alpha = zeros((m,n))
    for i in range(min(points[0]),max(points[0])):
        for j in range(min(points[1]),max(points[1])):
            x = linalg.solve(points,[i,j,1])
            if min(x) > 0:
                alpha[i,j] = 1
    return alpha

im1 = array(Image.open('imagen.jpg').convert('L'))
im2 = array(Image.open('img1.jpg').convert('L'))
#tp = array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])
tp = array([[270,971,935,288],[434,312,6,205],[1,1,1,1]])

im3 = image_in_image(im2,im1,tp)

figure()
gray()
imshow(im3)
axis('equal')
axis('off')
show()
'''
im2 = array(Image.open('imagen.jpg').convert('L'))
im1 = array(Image.open('img1.jpg').convert('L'))

#tp = array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])
#tp = array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])
tp = array([[285,930,975,267],[212,7,314,431],[1,1,1,1]])

# set from point to corners of im1
m,n = im1.shape[:2]
fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

# first triangle
tp2 = tp[:,:3]
fp2 = fp[:,:3]

# compute H
H = Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
            (H[0,2],H[1,2]),im2.shape[:2])

# alpha for triangle
alpha = alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im3 = (1 - alpha)*im2 + alpha*im1_t

# second triangle
tp2 = tp[:,[0,2,3]]
fp2 = fp[:,[0,2,3]]

# compute H 
H = Haffine_from_points(tp2,fp2)
im1_t = ndimage.affine_transform(im1,H[:2,:2],
        (H[0,2],H[1,2]),im2.shape[:2])

# alpha for triangle
alpha = alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
im4 = (1 - alpha) * im3 + alpha*im1_t

figure()
gray()
imshow(im4)
axis('equal')
axis('off')
show()
'''
