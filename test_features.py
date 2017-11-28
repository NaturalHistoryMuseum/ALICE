import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import skimage
import skimage.transform
from skimage.measure import ransac

images = []
data_dir = '.'
do_show = 0
for filename in ['out2label0.png', 'out2label1.png', 'out2label2.png', 'out2label3.png']:
    im = cv2.imread(os.path.join(data_dir, filename))
    factor = 1
    im = cv2.resize(im, (im.shape[1] // factor, im.shape[0] // factor))

    if do_show:
        cv2.imshow("main", im)
        cv2.waitKey(0)
    images.append(im)

img2 = images[3]
height, width = img2.shape[:2]
Y, X = np.mgrid[:height, :width]
distance = (height / 2)**2 + (width / 2)**2 - ((Y - height / 2)**2.0 + (X - width / 2)**2.0)
weight = distance / distance.sum()
aligned = [skimage.img_as_float(img2)]
aligned_weights = [weight]

for index, img1 in enumerate(images[:3]):
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    newimg = np.concatenate((img1, img2), axis=1)
    cv2.imwrite('image_pair_{}.png'.format(index), newimg)

    src_points = []
    dst_points = []
    for i in range(min(len(pts1), len(pts2))):
        pt_a = (int(pts1[i][0]), int(pts1[i][1]))
        pt_b = (int(pts2[i][0] + img1.shape[0]), int(pts2[i][1]))
        # # XXX manually select single label hack.
        # # replace with clustering to group similar matching vectors
        # if pt_a[0] < 800 or pt_a[1] > 750 or pt_a[1] < 600 or\
        #    pt_b[0] < 800 or pt_b[1] > 830:
        #     continue
        src_points.append(pt_a)
        dst_points.append((pt_b[0] - img1.shape[0], pt_b[1]))

        cv2.circle(newimg, pt_a, 2, (0, 255, 0), -1)
        cv2.circle(newimg, pt_b, 2, (0, 255, 0), -1)
        # cv2.circle(newimg, pt_a, 4, (255, 0, 0))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))

    src = np.array(src_points, dtype=np.float)
    dst = np.array(dst_points, dtype=np.float)

    np.savez('matches_{}'.format(index), src=src, dst=dst)

    H, inliers = ransac([src, dst], skimage.transform.ProjectiveTransform, min_samples=8, residual_threshold=1, max_trials=400)

    # for (x, y), is_good in zip(src, inliers):
    #     cv2.circle(img1, (int(x), int(y)), 2, (0, 255, 0) if is_good else (0, 0, 255), -1)

    warped_image = skimage.transform.warp(img1, H.inverse)
    cv2.imwrite('warped_{}.png'.format(index), warped_image * 256)
    cv2.imwrite('siftmatch_{}.png'.format(index), newimg)
    aligned.append(warped_image)
    aligned_weights.append(skimage.transform.warp(weight, H.inverse))

cv2.imwrite('composite_median.png', np.median(np.stack(aligned), axis=0) * 256)
cv2.imwrite('composite_mean.png', np.mean(np.stack(aligned), axis=0) * 256)

highest_weight = np.argmax(np.stack(aligned_weights), axis=0)
initial_weights = [1.0 * (highest_weight == i) for i in range(4)]
composite = (np.stack(aligned) * np.stack(aligned_weights)[..., np.newaxis]).sum(axis=0) / np.stack(aligned_weights).sum(axis=0)[..., np.newaxis]
cv2.imwrite('composite_highest.png', composite * 256)
