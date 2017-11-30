import matplotlib.pyplot as plt
import numpy as np
import pyflow
import skimage
from skimage.feature import match_descriptors, ORB
from skimage.measure import ransac
import skimage.transform
import time


from warping import SimilarAsPossible, bidirectional_similarity


def align_homography(image, target):
    detector_extractor1 = ORB()
    detector_extractor1.detect_and_extract(image[..., 1])
    detector_extractor2 = ORB()
    detector_extractor2.detect_and_extract(target[..., 1])
    matches = match_descriptors(detector_extractor1.descriptors, detector_extractor2.descriptors, cross_check=True)
    src = detector_extractor1.keypoints[matches[:, 0], ::-1]
    dst = detector_extractor2.keypoints[matches[:, 1], ::-1]

    # plt.imshow(np.concatenate((image, target), axis=1))
    # for b, a in zip(src, dst):
    #     plt.plot([a[0] + image.shape[1], b[0]], [a[1], b[1]], 'o-')
    # plt.show()

    H, inliers = ransac([src, dst], skimage.transform.ProjectiveTransform,
                        min_samples=8, residual_threshold=2, max_trials=400)

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
    S = SimilarAsPossible((int(np.ceil(image.shape[0] / 40)), int(np.ceil(image.shape[1] / 40))), (40, 40))

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

    output_image = skimage.transform.warp(image, S.transformation.inverse, cval=np.nan)
    mask = np.isnan(output_image)
    output_image[mask] = 0
    return output_image, mask
