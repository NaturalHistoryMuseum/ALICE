import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

images = []
data_dir = '.'
do_show = 0
for filename in ['outlabel0.png', 'outlabel1.png', 'outlabel2.png', 'outlabel3.png']:
    im = cv2.imread(os.path.join(data_dir, filename))
    factor = 1 
    im = cv2.resize(im, (im.shape[1] / factor, im.shape[0] / factor))

    if do_show:
        cv2.imshow("main", im)
        cv2.waitKey(0)
    images.append(im)

imgB = images[0]
num_found = []
for degree in [0]:
    imgA = np.array(images[3])
    rows, cols, _ = imgA.shape
    s = int(np.sqrt(cols ** 2 + rows ** 2))
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    (tx, ty) = ((s - cols) / 2, (s - rows) / 2)
    # third column of matrix holds translation, which takes effect after rotation.
    M[0, 2] += tx 
    M[1, 2] += ty
    img1 = cv2.warpAffine(imgA, M, (s, s))

    img2 = np.array(img1) 
    img2[:] = 0
    img2[:imgB.shape[0], :imgB.shape[1], :] = imgB

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
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x:x.distance)
    # matches = sorted(matches, key=lambda x:x.distance)
    # dists = [x.distance for x in matches]

    good = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    num_found.append(len(good))
    print degree, len(good)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = (h1 - h2) / 2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hdif:hdif + h2, :w2] = img1 
    newimg[:h1, w2:w1 + w2] = img2 

    src_points = []
    dst_points = []
    for i in range(min(len(pts1), len(pts2))):
        pt_a = (int(pts1[i][0]), int(pts1[i][1] + hdif))
        pt_b = (int(pts2[i][0] + w2), int(pts2[i][1]))
        # XXX manually select single label hack.
        # replace with clustering to group similar matching vectors 
        if pt_a[0] < 800 or pt_a[1] > 750 or pt_a[1] < 600 or\
           pt_b[0] < 800 or pt_b[1] > 830:
            continue
        src_points.append(pt_a)
        dst_points.append((pt_b[0] - w2, pt_b[1]))

        cv2.circle(newimg, pt_a, 2, (0, 255, 0), -1)
        cv2.circle(newimg, pt_b, 2, (0, 255, 0), -1)
        # cv2.circle(newimg, pt_a, 4, (255, 0, 0))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))

    src_points = np.array(src_points, dtype=np.float)
    dst_points = np.array(dst_points, dtype=np.float)
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 4)
    print H

    warped_image = cv2.warpPerspective(img1, H, tuple(img1.shape[:2]))
    dst = cv2.addWeighted(warped_image, 0.5, img2, 0.5, 0)
    cv2.imwrite("labelmerged.png", dst)
    # cv2.imshow("warped", (warped_image + img2) / 2)
    cv2.imshow("warped", dst)
    cv2.imshow("matches", newimg)
    cv2.imwrite('siftmatch.png', newimg)
    cv2.waitKey(10)

plt.plot(num_found)
plt.show()
