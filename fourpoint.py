import numpy as np
import os
import cv2
import itertools
from matplotlib import pyplot as plt
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
        print points 
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
print filenames

do_show = 0 
images = []
print filenames
for filename in filenames:
    im = cv2.imread(os.path.join(data_dir, filename))
    # resizing factor of image
    factor = 4 
    # factor = 8 
    # factor = 16 
    im = cv2.resize(im, (im.shape[1] / factor, im.shape[0] / factor))
    if do_show:
        cv2.imshow("main", im)
        cv2.waitKey(0)
    images.append(im)

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_mouse)

# points representing squares on the calibration grid
point_list = []
point_list.append([(1590, 2492), (3410, 996), (5889, 1626), (4300, 3283)]) #ALICE3_0002.JPG
point_list.append([(527, 1640), (3098, 914), (4898, 1956), (2216, 2778)])  #ALICE4_0002.JPG
point_list.append([(440, 2018), (2018, 798), (4818, 1342), (3465, 2722)])  #ALICE5_0002.JPG
point_list.append([(352, 1898), (2558, 1292), (3844, 2288), (1540, 2948)]) #ALICE6_0002.JPG
point_list = np.array(point_list, dtype=np.float)

# normalize squares by the square diagonal
for i, p in enumerate(point_list):
    fact = float(np.max([np.sqrt(np.sum((p[2] - p[0]) ** 2)), np.sqrt(np.sum((p[3] - p[1]) ** 2))]) / 300)
    point_list[i, :] /= fact

angles = [0, 90 + 180, 180, 270 + 180]
# adjust the calibration grid to be centred
adjust = np.array([(60, 190), (100, 200), (220, 220), (290, 110)]) / factor * 8

for i, a in enumerate(adjust):
    point_list[i, :, 0] += a[0]
    point_list[i, :, 1] += a[1]

while 1:
    for i in range(4):
        display = np.array(images[i])
        c = 0 
        points = point_list[i]
        for point in points:
            x0, y0 = point 
            cv2.circle(display, tuple(point.astype(np.int)), 5, (0, 0, 255), -1)
        if len(points) == 4:
            generated = True
            warped = four_point_transform(display, points, angles[i])
            dx, dy = warped.shape[0] / 3, warped.shape[1] / 3
            sx, sy = warped.shape[0], warped.shape[1]
            warped = warped[dy:sy-dy, dx:sx-dx, :]
            cv2.imshow("warp", warped)
            cv2.imwrite("outlabel%d.png" % i, warped)
            cv2.imwrite("image%d.png" % i, display)

        cv2.imshow("image", display)
        cv2.waitKey(0)