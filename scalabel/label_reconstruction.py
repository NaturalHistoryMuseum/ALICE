import numpy as np
import skimage
import skimage.filters
import skimage.io


from alignment import align_homography, align_optical_flow, align_regularised_flow
from features import global_matches
from labels import separate_labels
from post_processing import improve_contrast


images = [skimage.io.imread('outlabel{}.png'.format(i)) for i in range(4)]

points = global_matches(images)

label_images = separate_labels(images, points, visualise=True)

for l, label in enumerate(label_images):
    i = len(label) - 1
    warped = []
    aligned = [skimage.img_as_float(label[i])]
    flows = np.zeros((4, label[0].shape[0], label[0].shape[1], 2))

    for j in range(4):
        warped.append(align_homography(label[j], label[i]))
    skimage.io.imsave('warped_{}.png'.format(l), np.concatenate(warped, axis=1))

    for j in range(4):
        if j != i:
            warped_flow, mask = align_regularised_flow(warped[j], warped[i])
            aligned.append(warped_flow)
    skimage.io.imsave('warped_flow_{}.png'.format(l), np.concatenate(aligned, axis=1))

    skimage.io.imsave('composite_flow_median_{}_.png'.format(l), improve_contrast(np.median(np.stack(aligned), axis=0)))
