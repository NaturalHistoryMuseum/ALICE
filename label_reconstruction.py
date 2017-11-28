import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.filters


from alignment import align_homography, align_optical_flow, align_regularised_flow


images = [cv2.imread('out2label{}.png'.format(i)) for i in range(4)]

i = 3
warped = []
aligned = [skimage.img_as_float(images[i])]
flows = np.zeros((4, images[0].shape[0], images[0].shape[1], 2))

for j in range(4):
    warped.append(align_homography(images[j], images[i]))
    cv2.imwrite('warped_{}.png'.format(j), warped[j] * 255)

# for j in range(4):
#     if j != i:
#         warped_flow = align_regularised_flow(warped[j], warped[i])
#         cv2.imwrite('warped_flow_{}.png'.format(j), warped_flow * 255)
#         aligned.append(warped_flow)
#
# cv2.imwrite('composite_flow_median.png'.format(i), np.median(np.stack(aligned), axis=0) * 255)
