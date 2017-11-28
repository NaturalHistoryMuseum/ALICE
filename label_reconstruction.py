import numpy as np
import skimage
import skimage.filters
import skimage.io


from alignment import align_homography, align_regularised_flow


images = [skimage.io.imread('outlabel{}.png'.format(i)) for i in range(4)]

i = 3
warped = []
aligned = [skimage.img_as_float(images[i])]
flows = np.zeros((4, images[0].shape[0], images[0].shape[1], 2))

for j in range(4):
    warped.append(align_homography(images[j], images[i]))
    skimage.io.imsave('warped_{}.png'.format(j), warped[j])

for j in range(4):
    if j != i:
        warped_flow = align_regularised_flow(warped[j], warped[i])
        skimage.io.imsave('warped_flow_{}.png'.format(j), warped_flow)
        aligned.append(warped_flow)

skimage.io.imsave('composite_flow_median.png'.format(i), np.median(np.stack(aligned), axis=0))
