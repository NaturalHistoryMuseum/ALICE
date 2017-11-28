import numpy as np
import os
import skimage.io
import skimage.transform


data_dir = 'data/Test 3_Flies_GreyFoam'
specimen_prefix = "0004"

scale = 0.25

filenames = sorted(filename for filename in os.listdir('data/Test 3_Flies_GreyFoam') if specimen_prefix in filename)[2:]
images = [skimage.transform.rescale(skimage.io.imread(os.path.join(data_dir, name)), scale) for name in filenames]

# points representing squares on the calibration grid
point_list = []
point_list.append([(3752, 3840), (2919, 1092), (1712, 2540), (2415, 5139)])  # ALICE3_0002.JPG
point_list.append([(2661, 4082), (3321, 1940), (2386, 614), (1788, 2672)])  # ALICE4_0002.JPG
point_list.append([(2024, 2313), (2521, 4635), (3720, 3510), (3115, 1057)])  # ALICE5_0002.JPG
point_list.append([(2361, 1906), (1727, 4115), (2794, 5573), (3505, 3260)])  # ALICE6_0002.JPG

point_list = np.array(point_list, dtype=np.float) * scale
point_list = point_list[..., ::-1]

box_normal = np.array([[0, 100], [0, 0], [100, 0], [100, 100]])
trans = [skimage.transform.estimate_transform('projective', box_normal, points) for points in point_list]
height, width = images[0].shape[:2]
box_image = np.array([[0, height], [0, 0], [width, 0], [width, height]])
bounds = [transformation.inverse(box_image) for transformation in trans]
offset = np.stack(bounds).min(axis=0).min(axis=0)
shape = np.stack(bounds).max(axis=0).max(axis=0) - offset
scale = shape / np.array([4000, 4000])

trans = [(skimage.transform.AffineTransform(scale=scale) + skimage.transform.AffineTransform(translation=offset) + transformation) for transformation in trans]

for i in range(4):
    display = np.array(images[i])
    warped = skimage.transform.warp(display, trans[i], output_shape=(4000, 4000))

    skimage.io.imsave('outlabel{:d}.png'.format(i), warped[1000:-1000, 1000:-1000])
    skimage.io.imsave('image{:d}.png'.format(i), display)
