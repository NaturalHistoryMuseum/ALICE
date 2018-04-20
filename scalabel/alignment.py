import numpy as np
from scalabel import pyflow
import skimage
from skimage.feature import match_descriptors, ORB
from skimage.measure import ransac
import skimage.transform


from scalabel.warping import SimilarAsPossible, bidirectional_similarity


def align_homography(image, target):
    detector_extractor1 = ORB()
    detector_extractor1.detect_and_extract(image[..., 1])
    detector_extractor2 = ORB()
    detector_extractor2.detect_and_extract(target[..., 1])
    matches = match_descriptors(detector_extractor1.descriptors, detector_extractor2.descriptors, cross_check=True)
    src = detector_extractor1.keypoints[matches[:, 0], ::-1]
    dst = detector_extractor2.keypoints[matches[:, 1], ::-1]

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

    u, v, im2W = pyflow.coarse2fine_flow(
        target.copy(order='C'), image.copy(order='C'),
        alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)

    return np.stack((u, v), axis=2)


def align_regularised_flow(image, target):
    flow = align_optical_flow(image, target)
    flow_reverse = align_optical_flow(target, image)

    height, width = image.shape[:2]
    grid = np.stack(np.mgrid[:height, :width][::-1], axis=2)

    flow = flow[10:-10]
    flow_reverse = flow_reverse[10:-10]
    grid = grid[10:-10]

    P = grid.reshape(-1, 2) + 0.5
    P_hat = (grid - flow).reshape(-1, 2) + 0.5

    valid = (bidirectional_similarity(flow, flow_reverse) < 2).flatten()

    S = SimilarAsPossible(shape=image.shape, grid_separation=(40, 40)).fit(P[valid], P_hat[valid], alpha=2)

    return np.ma.masked_invalid(skimage.transform.warp(image, S.transformation.inverse, cval=np.nan))
