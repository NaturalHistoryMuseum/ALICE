import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyflow
import skimage
from skimage.measure import ransac
import skimage.transform
import time


from warping import SimilarAsPossible, bidirectional_similarity


sift = cv2.SIFT()


def align_homography(image, target):
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(target, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = [m for m, n in flann.knnMatch(des1, des2, k=2) if m.distance < 0.9 * n.distance]

    src_points = [kp1[m.queryIdx].pt for m in matches]
    dst_points = [kp2[m.trainIdx].pt for m in matches]

    plt.imshow(np.concatenate((image, target), axis=1))
    for b, a in zip(src_points, dst_points):
        plt.plot([a[0] + image.shape[1], b[0]], [a[1], b[1]], 'o-')
    plt.show()

    src = np.array(src_points, dtype=np.float)
    dst = np.array(dst_points, dtype=np.float)
    H, inliers = ransac([src, dst], skimage.transform.ProjectiveTransform, min_samples=8, residual_threshold=2, max_trials=400)

    return skimage.transform.warp(image, H.inverse)


def align_optical_flow(image, target):
    target = skimage.img_as_float(target)
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        target, image, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
    e = time.time()

    print('Time Taken: {:.2f} seconds for image of size ({:d}, {:d}, {:d})'.format(e - s, *image.shape))
    return im2W, u, v


def align_regularised_flow(image, target):
    S = SimilarAsPossible((12, 12), (40, 40))

    _, u, v = align_optical_flow(image, target)
    _, u_reverse, v_reverse = align_optical_flow(target, image)

    Y, X = np.mgrid[:image.shape[0], :image.shape[1]]

    u = u[10:-10]
    v = v[10:-10]
    u_reverse = u_reverse[10:-10]
    v_reverse = v_reverse[10:-10]
    Y = Y[10:-10]
    X = X[10:-10]

    BDS = bidirectional_similarity(u, v, u_reverse, v_reverse)

    P = np.stack((X.flatten(), Y.flatten()), axis=1) + 0.5
    P_hat = np.stack(((X - u).flatten(), (Y - v).flatten()), axis=1) + 0.5

    height, width = image.shape[:2]
    valid = (BDS < 2).flatten()

    P = P[valid]
    P_hat = P_hat[valid]

    S.fit(P, P_hat, 2)
    return skimage.transform.warp(image, S.transformation.inverse)
