import cv2
import numpy as np

img = cv2.imread("data/test_patterns/test2.jpg")

factor = 8 
img = cv2.resize(img, (img.shape[1] / factor, img.shape[0] / factor))
# im = im[380:800, 400:1000, :]

found, corners = cv2.findChessboardCorners(img, (8, 8))
print found, corners
if found:
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

if 0 and debug_dir:
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    path, name, ext = splitfn(fn)
    outfile = debug_dir + name + '_chess.png'
    cv2.imwrite(outfile, vis)
    if found:
        img_names_undistort.append(outfile)