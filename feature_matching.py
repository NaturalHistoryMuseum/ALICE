import skimage.io
import skimage.transform
from skimage.feature import ORB, match_descriptors, plot_matches, corner_harris, corner_peaks
from skimage.measure import ransac
import matplotlib.pyplot as plt
import numpy as np

img1 = skimage.transform.rescale(skimage.io.imread('out2label0.png'), 1)[..., 1]
img2 = skimage.transform.rescale(skimage.io.imread('out2label1.png'), 1)[..., 1]

feature_extractor1 = ORB(n_keypoints=100j, n_scales=4)
feature_extractor2 = ORB(n_keypoints=100j, n_scales=4)

feature_extractor1.detect_and_extract(img1)
feature_extractor2.detect_and_extract(img2)

matches = match_descriptors(feature_extractor1.descriptors, feature_extractor2.descriptors)

src = feature_extractor1.keypoints[matches[:, 0], ::-1]
dst = feature_extractor2.keypoints[matches[:, 1], ::-1]
model, inliers = ransac((src, dst), skimage.transform.ProjectiveTransform, min_samples=8, residual_threshold=3, max_trials=500)

# src2 = feature_extractor3.keypoints[matches2[:, 0], ::-1]
# dst2 = feature_extractor2.keypoints[matches2[:, 1], ::-1]
# model2, inliers2 = ransac((src2, dst2), skimage.transform.ProjectiveTransform, min_samples=8, residual_threshold=3, max_trials=500)

# db = MeanShift().fit(src[:, ::-1])
# for i in range(-1, db.labels_.max() + 1):
#     points = src[:, ::-1][db.labels_ == i]
#     plt.plot(points[:, 1], points[:, 0], '+')
# plt.imshow(img1)
# plt.show()

# ica = FastICA()
# ica.fit(src)
# for axis in ica.mixing_:
#     axis /= axis.std()
#     x_axis, y_axis = axis
#     plt.quiver(120, 55, x_axis, y_axis)
# plt.plot(src[:, 0], src[:, 1], '+')
# plt.imshow(img1)
# plt.show()
# #
# # src_ica = ica.transform(src)
# # src_ica /= src_ica.std(axis=0)
# # plt.plot(src_ica[:, 0], src_ica[:, 1], '+')
# # plt.show()
#

# img1 = skimage.io.imread('data/Test 3_Flies_GreyFoam/ALICE3_0004.JPG')
# img2 = skimage.io.imread('data/Test 3_Flies_GreyFoam/ALICE5_0004.JPG')
# img3 = skimage.io.imread('data/Test 3_Flies_GreyFoam/ALICE6_0004.JPG')
# img1 = img1[1500:2100, 1800:3000]
# img2 = img2[1600:2200, 2750:3750]
# img3 = img3[1280:1825, 3300:4250]

# warped = skimage.transform.warp(img1, model.inverse, output_shape=img2.shape)
# warped2 = skimage.transform.warp(img3, model2.inverse, output_shape=img2.shape)
# # plt.imshow(warped)
# # plt.imshow(img2)
# skimage.io.imsave('img2_original.png', img2)
# skimage.io.imsave('img1_warped.png', warped)
# skimage.io.imsave('img3_warped.png', warped2)
# # plt.plot(model(src)[:, 0], model(src)[:, 1], '+')
# plt.show()

# matches = matches[inliers]
ax = plt.axes()
plot_matches(ax, img1, img2, feature_extractor1.keypoints, feature_extractor2.keypoints, matches, matches_color='r')
plt.show()
# matches2 = matches2[inliers2]
# ax = plt.axes()
# plot_matches(ax, img3, img2, feature_extractor3.keypoints, feature_extractor2.keypoints, matches2, matches_color='r')
# plt.show()
