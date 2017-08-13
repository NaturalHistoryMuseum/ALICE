import numpy as np 
import cv2

# written by Adiran Rosebrock, used with some edits for IEEE @ UIUC Hackathon 2014 by Mohammad Saad

def order_points(pts):

    rect = np.zeros((4,2), dtype = "float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts, angle=None):
    # rect = order_points(pts)
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[0] - bl[0]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[0] - tl[0]) ** 2))

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[1] - bl[1]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # print maxWidth, maxHeight
    maxWidth = 200
    maxHeight = maxWidth 

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight]], dtype = 'float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    tx, ty = image.shape[1], image.shape[0] / 2
    translate = np.eye(3)
    translate[0, 2] = tx #third column of matrix holds translation, which takes effect after rotation.
    translate[1, 2] = ty
    M = np.dot(translate, M)
    # M[0, 2] = tx #third column of matrix holds translation, which takes effect after rotation.
    # M[1, 2] = ty
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    ones = np.ones(shape=(len(pts), 1))
    points_ones = np.hstack([pts, ones])
    warped = cv2.warpPerspective(image, M, tuple(2*np.array((image.shape[1], image.shape[0]))))
    p = M.dot(points_ones.T).T
    # transformed_points = [np.array([p[0] / p[2], p[1] / p[2]]) for p in M.dot(points_ones.T).T]
    # for p in transformed_points:
        # cv2.circle(warped, p, 15, (255, 0, 0), 1)


    if angle is not None:
        rows, cols, _ = warped.shape
        s = int(np.sqrt(cols ** 2 + rows ** 2))
        R = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        (tx, ty) = ((s - cols)/2, (s - rows)/2)
        # tx += cols / 2 - transformed_points[0][0]
        # ty += rows / 2 - transformed_points[0][1]

        p = np.array([(p[0] / p[2], p[1] / p[2], 1) for p in M.dot(points_ones.T).T])
        p = R.dot(p.T).T
        print p[:, 0]
        tx += cols / 2 - np.mean(p[:, 0])
        ty += rows / 2 - np.mean(p[:, 1])

        R[0, 2] += tx #third column of matrix holds translation, which takes effect after rotation.
        R[1, 2] += ty
        warped = cv2.warpAffine(warped, R, (s, s))

        # for pt in p:
            # cv2.circle(warped, tuple(pt.astype(np.int)), 15, (255, 0, 0), 1)


    return warped