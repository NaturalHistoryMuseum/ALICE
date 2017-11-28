import numpy as np
import os
import cv2
import itertools
from matplotlib import pyplot as plt
import skimage.transform
from transform import four_point_transform

points = []

def on_mouse(event, x, y, flags, param):
    """
    Mouse event callback.

    Args:
        event (int): Opencv type of event.
        x, y (int): Coordinates of event.
        flags (int): Opencv flags.
        param (int): Opencv params.
    """
    if event == cv2.cv.CV_EVENT_LBUTTONDOWN:
        print(points)
        if len(points) < 4:
            points.append((x,y))
            print (x, y)


data_dir = 'data/Test 3_Flies_GreyFoam'
specimen_prefix = "0004"

filenames = []
for filename in os.listdir(data_dir):
    if specimen_prefix in filename and 'ALICE1' not in filename \
    and 'ALICE2' not in filename :
        filenames.append(filename)
filenames.sort()
print(filenames)

do_show = 0
images = []
print(filenames)
for filename in filenames:
    im = cv2.imread(os.path.join(data_dir, filename))
    # resizing factor of image
    factor = 4
    # factor = 8
    # factor = 16
    im = cv2.resize(im, (im.shape[1] // factor, im.shape[0] // factor))
    if do_show:
        cv2.imshow("main", im)
        cv2.waitKey(0)
    images.append(im)

# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_mouse)

# points representing squares on the calibration grid
point_list = []
point_list.append([(1590, 2492), (3410, 996), (5889, 1626), (4300, 3283)]) #ALICE3_0002.JPG
point_list.append([(527, 1640), (3098, 914), (4898, 1956), (2216, 2778)])  #ALICE4_0002.JPG
point_list.append([(440, 2018), (2018, 798), (4818, 1342), (3465, 2722)])  #ALICE5_0002.JPG
point_list.append([(352, 1898), (2558, 1292), (3844, 2288), (1540, 2948)]) #ALICE6_0002.JPG

point_list = []
point_list.append([(3752, 3840), (2919, 1092), (1712, 2540), (2415, 5139)]) #ALICE3_0002.JPG
point_list.append([(2661, 4082), (3321, 1940), (2386, 614), (1788, 2672)])  #ALICE4_0002.JPG
point_list.append([(2024, 2313), (2521, 4635), (3720, 3510), (3115, 1057)])  #ALICE5_0002.JPG
point_list.append([(2361, 1906), (1727, 4115), (2794, 5573), (3505, 3260)]) #ALICE6_0002.JPG

point_list = np.array(point_list, dtype=np.float) / factor
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
    c = 0
    points = point_list[i]
    # for point in points:
    #     x0, y0 = point
    #     cv2.circle(display, tuple(point.astype(np.int)), 5, (0, 0, 255), -1)
    warped = skimage.transform.warp(display, trans[i], output_shape=(4000, 4000))

    cv2.imwrite("out2label%d.png" % i, warped[1000:-1000, 1000:-1000] * 255)
    cv2.imwrite("image%d.png" % i, display)
