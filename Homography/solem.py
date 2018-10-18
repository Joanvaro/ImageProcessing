#!/usr/bin/python

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




