import numpy as np
import cv2
import os
from skimage.transform import (hough_line, hough_line_peaks)
from segment import segment_grabcut
from matplotlib import pyplot as plt
from utils import peakdetect

# data_dir = "data/Test2_Grey_Flies/" 
data_dir = "data/Test 3_Flies_GreyFoam/" 

def resize(im, factor=8):
    im = cv2.resize(im, (im.shape[1] / factor, im.shape[0] / factor))
    return im

cv2.namedWindow("overview")
cv2.namedWindow("sideview")
cv2.namedWindow("main")
cv2.moveWindow("sideview", 0, 0)
cv2.moveWindow("overview", 800, 0)
cv2.moveWindow("main", 800, 600)

def find_lines(image):
    rows, cols = image.shape
    h, theta, d = hough_line(image)
    lines = []
    # min_distance=9, min_angle=10, threshold=None, num_peaks=inf)

    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=9, min_angle=30, threshold=0.1*np.max(h))):
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=19, min_angle=5, threshold=0.2*np.max(h))):
    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=5, min_angle=5)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
        line = ((0, y0), (cols, y1))
        # plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'r')
        lines.append(line)
    return lines

filenames = []
for root, dir, files in os.walk(data_dir):
    filenames.extend([os.path.join(root, f) for f in files if 'ALICE1' in f])

for location in filenames:
    # if "0011" not in location:
        # continue
    # print filename
    for i in range(1, 7):
        # location = os.path.join(data_dir, filename)
        sub_location = location.replace("ALICE1", "ALICE%d" % i)
        print sub_location
        im = resize(cv2.imread(sub_location))
        display = im.copy()
        factor = 8
        s = None
        if "ALICE1" in sub_location:
            win_name = "overview"
        elif "ALICE2" in sub_location:
            win_name = "sideview"
            im = cv2.flip(np.transpose(im, (1, 0, 2)), 1)
        else:
            win_name = "main"
            # _, display = segment_grabcut(im)
            # print display.shape, display.dtype
            # im[display == 0] = 0

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # gray = cv2.blur(gray, (3, 3));
            # gray = cv2.Canny(gray, 30, 10, 5);
            gray = cv2.Canny(gray, 50, 70, 3);
            # cv2.imshow("cann", gray)
            lines = find_lines(gray)
            for line in lines:
                # a = line[0][0], line[1]
                a = [int(x) for x in line[0]]
                b = [int(x) for x in line[1]]
                # pts = [(a, b) in zip(line[0], line[1])
                # cv2.line(im, tuple(a), tuple(b), color=255)

        if win_name == "sideview":
            _, display = segment_grabcut(im, background_bounds=True)
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            s = np.sum(gray, axis=1)
            peaks, _ = peakdetect(s, delta=0.1*max(s))
            for peak, value in peaks:
                y = int(peak)
                cv2.line(im, (0, y), (im.shape[1], y), color=(0, 0, 255))
            side_height = im.shape[0]
            cv2.imshow("mask", gray)
            cv2.waitKey(10)
            plt.plot(s)
            plt.plot(peaks[:, 0], peaks[:, 1], 'o')
            plt.ion()
            plt.show()
        elif win_name == "main":
            h, w = im.shape[:2]
            coloured = np.zeros((h, w), np.uint8)
            coloured = display.copy()
            # coloured[:] = 0
            # for fg_i in range(len(peaks)):
            for fg_i in [2]:
                h, w = im.shape[:2]
                initial = np.zeros((h, w), np.uint8)
                initial[:] = cv2.GC_PR_BGD
                for peak, value in peaks:
                    # y = int(peak)
                    offset = [420, 370, 390, 310]
                    hole_x = [320, 280, 390, 460]
                    y = int(im.shape[0] * (peak - 250) / (side_height - offset[i - 3]))
                    x = int(hole_x[i - 3])
                    cv2.line(display, (0, y), (im.shape[1], y), color=(0, 0, 255))
                    cv2.line(display, (x, 0), (x, im.shape[0]), color=(255, 0, 0))
                    cv2.circle(display, (x, y), int(2 * np.sqrt(value / 255)), color=(0, 0, 255), thickness=-1)

                    if peak == peaks[fg_i][0]: 
                        cv2.circle(initial, (x, y), int(3 * np.sqrt(value / 255)), color=int(cv2.GC_PR_FGD), thickness=-1)
                        cv2.circle(initial, (x, y), int(2 * np.sqrt(value / 255)), color=int(cv2.GC_FGD), thickness=-1)
                    else:
                        cv2.circle(initial, (x, y), int(2 * np.sqrt(value / 255)), color=int(cv2.GC_BGD), thickness=-1)

                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                rect = None
                mask = initial
                cv2.grabCut(im, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                coloured[mask2 == 0] = 0
            # coloured[:, :, 1] = 0
            # coloured[:, :, 2] = 0
            cv2.imshow("col", coloured)

        cv2.imshow(win_name, display)
        if win_name == "main":
            cv2.waitKey(0)

