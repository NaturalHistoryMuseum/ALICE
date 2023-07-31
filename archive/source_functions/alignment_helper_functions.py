from hashlib import blake2b
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from skimage import measure
import cv2
from collections import Counter
import imutils
import re


# GENERAL FUNCTIONS #
#####################


def removeReps(T, F):
    # Remove repeated points in a curve.
    n = len(T)
    newT = []
    newF = []
    newT.append(T[0])
    newF.append(F[0])
    k = 0
    for i in range(1, n):
        if (T[i] != newT[k]) or (F[i] != newF[k]):
            newT.append(T[i])
            newF.append(F[i])
            k = k + 1
    return newT, newF


def reparam(x, y, npoints):
    # This function reparametrizes the curve to have npoints.
    tst = np.zeros((len(x), 2))
    tst[:, 0] = x
    tst[:, 1] = y

    p = tst
    dp = np.diff(p, axis=0)
    pts = np.zeros(len(dp) + 1)
    pts[1:] = np.cumsum(np.sqrt(np.sum(dp * dp, axis=1)))
    newpts = np.linspace(0, pts[-1], int(npoints))
    newx = np.interp(newpts, pts, p[:, 0])
    newy = np.interp(newpts, pts, p[:, 1])

    return newx, newy


def reject_outliers(data, index, m=10):
    # Function used to remove outliers from arrays. This is based on the median of the data.
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    inds = np.where(s < m)[0]
    if len(index) != 0:
        return data[inds], index[inds]
    else:
        return data[inds]


def get_angle(line1_x, line1_y, line2_x, line2_y):
    # Angle between two intersecting lines.
    g1 = (line1_y[0] - line1_y[1]) / (line1_x[0] - line1_x[1])
    g2 = (line2_y[0] - line2_y[1]) / (line2_x[0] - line2_x[1])
    theta = np.rad2deg(np.arctan(abs((g2 - g1) / (1 + (g1 * g2)))))
    return theta


def round_pixel_colours(image, base=5):
    # Aim: round pixel colours to the nearest multiple of your choosing (default 5).

    new_img = np.zeros(np.shape(image), dtype="uint8")
    for i, pix in enumerate(image):
        new_img[i] = np.array(base * np.round(pix / base), dtype="uint8")

    return new_img


def ccw(A, B, C):
    return (C["y"] - A["y"]) * (B["x"] - A["x"]) > (B["y"] - A["y"]) * (C["x"] - A["x"])


def intersect(A, B, C, D):
    # Return true if line segments AB and CD intersect
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# SEGMENTATION / CONTOUR FUNCTIONS #
####################################

# Clear Non-label Area:
def segment_label(mask, img):
    # Function segments an image by transforming pixels outside of the given mask region into one colour.
    inds = np.where(mask == False)
    img_seg = deepcopy(img)
    img_seg[inds] = 1
    return img_seg


def find_label_contour_boundaries(mask):
    # Finds the contour on an image that describes the mask's outline.
    contours = measure.find_contours(mask, 0.8)
    total_contours = len(contours)
    k = np.argmax([len(contours[i][:, 0]) for i in range(total_contours)])
    x = contours[k][:, 1]
    y = contours[k][:, 0]

    mnx = min(x)
    mxx = max(x)
    mny = min(y)
    mxy = max(y)

    mnx_ = np.argmin(x)
    mxx_ = np.argmax(x)
    mny_ = np.argmin(y)
    mxy_ = np.argmax(y)

    return contours, x, y, mnx, mxx, mny, mxy, mnx_, mxx_, mny_, mxy_


def basic_threshold(img, p):
    # Global thresholding on a grey-scaled images based on a threshold value, p.
    if np.max(img) < 2:  # Assumes colour map is either between 0-1, or 0-255.
        mx = 1
    else:
        mx = 255
    thresh = deepcopy(img)
    thresh[np.where(img < p)] = 0
    thresh[np.where(img >= p)] = mx

    return thresh


def combine_masks(mask, contours):
    # Combines masks by filling-in greater boundary created by all the masks.
    """
    This function works by going through each individual vertical segment within a boundary and turning
    all pixels between the lowest and highest "True" pixel "True". The same is done for horizontal segments.
    """

    a, b = np.int_(
        [
            np.floor(min([min(c[:, 1]) for c in contours])),
            np.ceil(max([max(c[:, 1]) for c in contours])),
        ]
    )

    inds = np.int_(np.linspace(a, b, (b - a) + 1))

    mask_updt1 = deepcopy(mask)

    for v in inds:
        try:
            a, b = np.where(mask[:, v] == True)[0][[0, -1]]
            mask_updt1[a:b, v] = True
        except:
            pass

    a, b = np.int_(
        [
            np.floor(min([min(c[:, 0]) for c in contours])),
            np.ceil(max([max(c[:, 0]) for c in contours])),
        ]
    )

    inds = np.int_(np.linspace(a, b, (b - a) + 1))

    mask_updt2 = deepcopy(mask_updt1)

    for v in np.int_(inds):
        try:
            a, b = np.where(mask_updt1[v, :] == True)[0][[0, -1]]
            mask_updt2[v, a:b] = True
        except:
            pass

    return mask_updt2


# POINT-DISTANCE FUNCTIONS #
############################


def dist_two_points(x1, y1, x2, y2, ydist=True):
    # Finds distance between two points in R2.
    if ydist == False:
        d = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
    else:
        d = abs(y2 - y1)
    return d


def find_furthest_two_points(x, y, furthest=False, top=10):
    # Finds furthest two points on curve.
    dists = []
    pairs = []
    n = len(x)
    for i, xp1 in enumerate(x):
        yp1 = y[i]
        for j in range(i + 1, n):
            xp2 = x[j]
            yp2 = y[j]
            d = dist_two_points(xp1, yp1, xp2, yp2)
            dists.append(d)
            pairs.append([i, j])

    if furthest == True:
        a, b = pairs[np.argmax(dists)]
    else:
        # In this case, it looks at the top distances, and from that selection,
        # it chooses the pair with the smallest x-range.
        inds = np.argsort(dists)[-top:]
        xdiff = []
        for u in inds:
            a, b = pairs[u]
            xdiff.append(abs(x[a] - x[b]))

        a, b = pairs[inds[np.argmin(xdiff)]]

    return a, b


# AXES-DIRECTION FUNCTIONS #
############################


def line_rotation(x, y, cx, cy, unq_x, angles):
    # Rotates a line (x,y) using angles.
    # Inputs:
    # x,y - line based on furthest two points on original curve,
    # cx,cy - origin for rotation. Based on one of the furthest two points.
    # unq_x - sorted, unique x values in the original curve,
    # angles - angles to rotate line by, e.g. angles = np.linspace(0,120,13).
    coords = []

    for theta in angles:
        theta = np.deg2rad(theta)
        x_ = ((x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)) + cx
        y_ = (-(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)) + cy
        newx = np.sort(unq_x[np.where((unq_x > min(x_)) & (unq_x < max(x_)))[0]])
        y_2 = np.interp(newx, np.sort(x_), y_[np.argsort(x_)])

        if len(y_2) > 0:
            coords.append([newx, y_2])

    coords = [coord for coord in coords if len(coord[0]) > 0]

    return coords


def total_alignments(coord, X, Y, epsilon=10):
    # Find points of intersection between rotated line and original curve (X,Y).
    a, b = removeReps(np.round(coord[0]), np.round(coord[1]))

    aligned_points = []

    total = 0
    for i, xp in enumerate(a):
        yp = b[i]
        inds = np.where((np.array(X) < xp + epsilon) & (np.array(X) > xp - epsilon))[0]
        yvals = np.array(Y)[inds]
        l_ind = np.where((yvals >= yp - epsilon) & (yvals <= yp + epsilon))[0]
        l = len(l_ind)
        total = total + l

        if l > 0:
            aligned_points.append([X[inds[0]], yvals[l_ind[0]]])

    return total, aligned_points


def line_of_best_fit(xcurve, ycurve, x, y, t_, minmax="min"):
    # Approximates label side, to curve (xcurve,ycurve).

    if minmax == "max":
        k = np.argmin(xcurve)
        X = [xcurve[k]]
        Y = [ycurve[k]]
    else:
        k = np.argmax(xcurve)
        X = [x[-1]]
        Y = [y[-1]]

    X.extend(list(np.array(t_).T[0]))
    Y.extend(list(np.array(t_).T[1]))

    if minmax == "max":
        X.append(x[-1])
        Y.append(y[-1])
    else:
        X.append(xcurve[k])
        Y.append(ycurve[k])

    X = np.array(X)
    Y = np.array(Y)

    theta = np.polyfit(X, Y, 1)

    y_line = theta[1] + theta[0] * X

    return X, Y, y_line


def find_direction(mask, angleTotal=37, angleMax=180, epsilon=15):
    # Find label direction.
    # 1) Find mask contour:
    contours = find_label_contour_boundaries(mask)
    contours = contours[0]
    # 2) Reparametrize contour:
    contour_x, contour_y = reparam(contours[0][:, 1], contours[0][:, 0], 80)
    # 3) Find furthest two points on label contour:
    a, b = find_furthest_two_points(contour_x, contour_y)
    swapped = 0
    if contour_y[a] < contour_y[b]:
        swapped = 1
    x = [contour_x[a], contour_x[b]]
    y = [contour_y[a], contour_y[b]]
    # 4) Rotate lines:
    cx = contour_x[b]
    cy = contour_y[b]
    vals = np.sort(np.unique(np.round(contour_x)))

    angles = np.linspace(0, angleMax, angleTotal)[1:]
    coords1 = line_rotation(x, y, cx, cy, vals, angles)
    angles = np.linspace(-1 * angleMax, 0, angleTotal)[:-1]
    coords2 = line_rotation(x, y, cx, cy, vals, angles)
    # 5) Find optimally rotated lines (to find perpendicular lines):
    x_round, y_round = removeReps(np.round(contour_x), np.round(contour_y))
    tot1 = [
        total_alignments(coord, x_round, y_round, epsilon=epsilon)[0]
        for coord in coords1
    ]
    tot2 = [
        total_alignments(coord, x_round, y_round, epsilon=epsilon)[0]
        for coord in coords2
    ]
    k1 = np.argmax(tot1)
    k2 = np.argmax(tot2)
    # 6) Line approximation:
    new_lines = []
    _, t_ = total_alignments(coords2[k2], x_round, y_round, epsilon=epsilon)
    if swapped == 0:
        X, _, y_line = line_of_best_fit(x_round, y_round, x, y, t_, minmax="max")
        new_lines.append([X, y_line])
        _, t_ = total_alignments(coords1[k1], x_round, y_round, epsilon=epsilon)
        X, _, y_line = line_of_best_fit(x_round, y_round, x, y, t_, minmax="min")
    else:
        X, _, y_line = line_of_best_fit(x_round, y_round, x, y, t_, minmax="min")
        new_lines.append([X, y_line])
        _, t_ = total_alignments(coords1[k1], x_round, y_round, epsilon=epsilon)
        X, _, y_line = line_of_best_fit(x_round, y_round, x, y, t_, minmax="max")
    new_lines.append([X, y_line])

    return new_lines, [x, y], swapped, [contour_x, contour_y]


def find_label_corners(new_lines, contours, swapped=0, ydistance=True):
    # 1) Get known corners:

    dists = []
    pairs = []
    n = len(contours[0])
    if swapped == 1:
        xp1 = new_lines[0][0][0]
        yp1 = new_lines[0][1][0]
    else:
        xp1 = new_lines[1][0][0]
        yp1 = new_lines[1][1][0]

    for j in range(0, n):
        xp2 = contours[0][j]
        yp2 = contours[1][j]
        d = dist_two_points(xp1, yp1, xp2, yp2, ydist=ydistance)
        dists.append(d)
        pairs.append([xp2, yp2])

    x_, y_ = np.round(pairs[np.argmax(dists)], 3)

    if swapped == 1:
        newx = [xp1, new_lines[1][0][0], x_, new_lines[0][0][-1]]
        newy = [yp1, new_lines[1][1][0], y_, new_lines[0][1][-1]]
    else:
        newx = [new_lines[1][0][-1], x_, new_lines[0][0][0], new_lines[1][0][0]]
        newy = [new_lines[1][1][-1], y_, new_lines[0][1][0], new_lines[1][1][0]]

    corners_x = deepcopy(newx)
    corners_y = deepcopy(newy)

    # 2) - Find new (final) corner
    a = new_lines[0][0][0] - new_lines[1][0][0]
    b = new_lines[0][1][0] - new_lines[1][1][0]
    if swapped == 0:
        newx = new_lines[1][0] + a
        newy = new_lines[1][1] + b
    else:
        newx = new_lines[0][0] - a
        newy = new_lines[0][1] - b
    new_corner_x = deepcopy(newx[-1])
    new_corner_y = deepcopy(newy[-1])

    # 3) Pick best (final) corner
    d1 = dist_two_points(xp1, yp1, x_, y_)
    d2 = dist_two_points(xp1, yp1, new_corner_x, new_corner_y)
    k = np.argmax([d1, d2])
    if k == 0:
        best_corner_x = deepcopy(x_)
        best_corner_y = deepcopy(y_)
    else:
        best_corner_x = deepcopy(new_corner_x)
        best_corner_y = deepcopy(new_corner_y)
        if swapped == 1:
            corners_x[2] = best_corner_x
            corners_y[2] = best_corner_y
        else:
            corners_x[1] = best_corner_x
            corners_y[1] = best_corner_y

    return corners_x, corners_y


def find_label_corners_v2(new_lines, contours, angleMax=30, angleTotal=11, epsilon=15):

    # 1) Find the corner where the two lines join:
    A1 = [new_lines[0][0][0], new_lines[0][1][0]]
    B1 = [new_lines[0][0][-1], new_lines[0][1][-1]]
    A2 = [new_lines[1][0][0], new_lines[1][1][0]]
    B2 = [new_lines[1][0][-1], new_lines[1][1][-1]]
    # We do this by computing the distances between the four points.
    d1 = dist_two_points(A1[0], A1[1], A2[0], A2[1], ydist=False)
    d2 = dist_two_points(B1[0], B1[1], A2[0], A2[1], ydist=False)
    d3 = dist_two_points(A1[0], A1[1], B2[0], B2[1], ydist=False)
    d4 = dist_two_points(B1[0], B1[1], B2[0], B2[1], ydist=False)

    v = np.argmin([d1, d2, d3, d4])

    corners_ = [[A1, A2, B1, B2], [B1, A2, A1, B2], [A1, B2, B1, A2], [B1, B2, A1, A2]][
        v
    ]
    # 2) Create new corner by computing the average between the close points:
    avg_corner = (np.array(corners_[1]) + np.array(corners_[0])) / 2
    x1, y1 = avg_corner

    # 3) Find approximate fourth side by translating one side:
    xe, ye = avg_corner - np.array(corners_[2])
    x = [corners_[2][0], corners_[3][0] - xe]
    y = [corners_[2][1], corners_[3][1] - ye]

    # 4) Adjust the new side by rotating it until it is better aligned with label contour:
    # Specify origin.
    cx = corners_[2][0]
    cy = corners_[2][1]

    vals = np.sort(np.unique(np.round(contours[0])))

    # Create rotated lines.
    angles = np.linspace(0, angleMax, angleTotal)[1:]
    coords1 = line_rotation(x, y, cx, cy, vals, angles)
    angles = np.linspace(-1 * angleMax, 0, angleTotal)[:-1]
    coords2 = line_rotation(x, y, cx, cy, vals, angles)

    x_round, y_round = removeReps(np.round(contours[0]), np.round(contours[1]))

    # Count total aligned points.
    tot1 = [
        total_alignments(coord, x_round, y_round, epsilon=epsilon)[0]
        for coord in coords1
    ]
    tot2 = [
        total_alignments(coord, x_round, y_round, epsilon=epsilon)[0]
        for coord in coords2
    ]

    # 5) Select rotated line that is most aligned with original contour:
    if max(tot1) > max(tot2):
        m = np.argmax(tot1)
        new_line_ = coords1[m]
    else:
        m = np.argmax(tot2)
        new_line_ = coords2[m]

    # 6) Update list of four corners:

    d_1 = dist_two_points(
        new_line_[0][-1], new_line_[1][-1], corners_[2][0], corners_[2][1], ydist=False
    )
    d_2 = dist_two_points(
        new_line_[0][0], new_line_[1][0], corners_[2][0], corners_[2][1], ydist=False
    )

    if d_1 > d_2:
        corners_x2 = [corners_[3][0], new_line_[0][-1], new_line_[0][0], x1]
        corners_y2 = [corners_[3][1], new_line_[1][-1], new_line_[1][0], y1]
    else:
        corners_x2 = [corners_[3][0], new_line_[0][0], new_line_[0][-1], x1]
        corners_y2 = [corners_[3][1], new_line_[1][0], new_line_[1][-1], y1]

    return corners_x2, corners_y2


# TRANSFORMATION FUNCTIONS #
############################


def pers_transform(corners_x, corners_y, img_orig, eps=0.05, box_epsilon=1):
    dists = np.zeros((6, 3))

    # STEP 1: Identify longest / shortest sides.
    ###########################################

    # Find distances between corners:
    k = 0

    for i in range(0, 4):
        xp1 = corners_x[i]
        yp1 = corners_y[i]

        for j in range(i + 1, 4):

            xp2 = corners_x[j]
            yp2 = corners_y[j]

            d = dist_two_points(xp1, yp1, xp2, yp2, ydist=False)

            dists[k, :] = [i, j, np.round(d, 3)]
            k = k + 1

    all_pairs = [[0, 1], [1, 2], [2, 3], [0, 3]]

    a, b = np.argsort(dists[:, 2])[:2]

    short_inds = np.int_(dists[[a, b], :2]).tolist()

    long_inds = [v for v in all_pairs if v not in short_inds]

    # STEP 2: Create desired box.
    #############################

    c, d = [i for i, v in enumerate(np.int_(dists[:, :2]).tolist()) if v in long_inds]
    mx_long = max(np.int_(dists[[c, d], 2]))
    mx_short = max(np.int_(dists[[a, b], 2]))

    box_x = [0, 0, mx_long, mx_long, 0]
    box_y = [0, mx_short, mx_short, 0, 0]

    # STEP 3: Match corners.
    ########################

    # Match corners of desired box with current box:
    u, v = [short_inds[i][0] for i in range(0, 2)]
    p = np.argmin([corners_x[u], corners_x[v]])
    u, v = short_inds[p]

    corner_inds = {}

    i, j = np.argsort([corners_y[u], corners_y[v]])
    k = [u, v][i]
    corner_inds["bot_left"] = [corners_x[k], corners_y[k]]
    k = [u, v][j]
    corner_inds["top_left"] = [corners_x[k], corners_y[k]]

    p = (p + 1) % 2
    u, v = short_inds[p]

    i, j = np.argsort([corners_y[u], corners_y[v]])
    k = [u, v][i]
    corner_inds["bot_right"] = [corners_x[k], corners_y[k]]
    k = [u, v][j]
    corner_inds["top_right"] = [corners_x[k], corners_y[k]]

    order = ["bot_left", "top_left", "top_right", "bot_right"]
    pts1 = []
    pts2 = []

    mx = (max(corners_x) + min(corners_x)) / 2
    my = (max(corners_y) + min(corners_y)) / 2

    v = box_epsilon - 1
    a, b = np.array([mx, my]) * v

    for i in range(0, 4):
        pts1.append(
            [
                (corner_inds[order[i]][0] * box_epsilon) - a,
                (corner_inds[order[i]][1] * box_epsilon) - b,
            ]
        )
        pts2.append([box_x[i] * box_epsilon, box_y[i] * box_epsilon])

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    # STEP 4: Segment label.
    ########################

    box_x_orig = deepcopy(corners_x)
    box_x_orig.append(corners_x[0])
    box_y_orig = deepcopy(corners_y)
    box_y_orig.append(corners_y[0])

    mx = (max(box_x_orig) + min(box_x_orig)) / 2
    my = (max(box_y_orig) + min(box_y_orig)) / 2

    v = 0.15
    a, b = np.array([mx, my]) * v
    v = v + 1
    box_big_x = list((np.array(box_x_orig) * v) - a)
    box_big_y = list((np.array(box_y_orig) * v) - b)

    img_filled = deepcopy(img_orig)
    a3 = np.array([np.array([box_big_x, box_big_y]).T], dtype=np.int32)

    # Fill background (exterior of label contour) in one colour (fill_col)
    fill_col = [255, 0, 0]
    img_filled = cv2.fillPoly(img_filled, a3, fill_col)

    img_bin = deepcopy(img_orig)
    img_bin[np.where(img_filled != [255, 0, 0])[:2]] = 0

    # STEP 5: Perspective transformation.
    #####################################

    M = cv2.getPerspectiveTransform(pts1, pts2)
    # Slightly adjust boundary:
    ydim, xdim, _ = np.shape(img_bin)
    epsilon_x = xdim * eps
    epsilon_y = ydim * eps

    img_warped = cv2.warpPerspective(
        img_bin, M, (int(max(box_x) + epsilon_x), int(max(box_y) + epsilon_y))
    )

    return img_warped, box_x, box_y, img_bin


# ALIGNMENT FUNCTIONS #
#######################


def adjust_alignment(
    image_orig,
    angles=np.linspace(-10, 10, 21),
    percentile=15,
    min_contour_height=15,
    height_prop=30,
    return_col=False,
):

    k = 0

    opt_angle = 0
    max_len = 0
    max_ind = 0

    all_lens = []

    # Convert image to greyscale:
    image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    # Compute threshold value based on percentile (not including black pixels):
    bound = np.percentile(
        image.flatten()[np.where(image.flatten() != 0)[0]], percentile
    )
    # Global thresholding:
    image_thresh = basic_threshold(image, bound)

    for i, angle in enumerate(angles):
        # Rotate image based on angle:
        seg = imutils.rotate_bound(image_thresh, angle)
        # Find contours:
        contours = measure.find_contours(seg, 0.8)
        # Filter for contours whose number of points isn't an outlier. This step is required to remove
        # contours around the label hence the aim is to get the contours of characters only.
        lens_ = [len(c[:, 0]) for c in contours]
        _, inds = reject_outliers(
            np.array(lens_), np.int_(np.linspace(0, len(lens_) - 1, len(lens_)))
        )
        # Save y coordinates of filtered contours:
        min_height = min([min_contour_height, np.shape(image)[0] / height_prop])
        all_y = []
        for i in inds:
            c = contours[i]
            x = c[:, 1]
            y = c[:, 0]
            if (len(x) > 10) & (
                (max(y) - min(y)) > min_height
            ):  # We focus on contours > 10 points. This number was chosen arbitrarily to exclude illegible/"accidental" contours.
                # Remove repeated points in contour.
                x, y = removeReps(np.round(x), np.round(y))
                all_y.extend(y)
        # Find number of points from the contours that lie in each horizontal strip across the thresholded image.
        lens = [
            len(np.where(np.array(all_y) == p)[0]) for p in range(0, np.shape(seg)[1])
        ]
        # Compute maximum across the horizontal strips.
        v = np.argmax(lens)
        m = max(lens)
        all_lens.append(lens)
        if m > max_len:
            max_len = deepcopy(m)
            max_ind = deepcopy(v)
            opt_angle = deepcopy(angle)

    rotated_img = imutils.rotate_bound(image, opt_angle)
    if return_col == True:
        return imutils.rotate_bound(image_orig, opt_angle)
    else:
        return opt_angle, max_len, max_ind, all_lens, rotated_img


# PIN-REMOVAL FUNCTIONS #
#########################


def round_pixel_colours(image, base=5):

    new_img = np.zeros(np.shape(image), dtype="uint8")
    for i, pix in enumerate(image):
        new_img[i] = np.array(base * np.round(pix / base), dtype="uint8")

    return new_img


def highlight_pin(img_orig, pin_mask_paths, new_col=[255, 0, 0]):
    # Aim: combine pin masks and convert their pixels in the original image.

    msk_img = deepcopy(img_orig)

    for msk in pin_mask_paths:
        mask_pin = np.load(msk)
        msk_img[np.where(mask_pin == True)] = new_col

    return msk_img


def hide_pin(image, label_mask, pin_mask, epsilon=10):
    # Aim: Convert the pin to have the same colour as the label background colour.
    """
    Note to self: change this to avoid changing pixel cols, and just use binaries.
    """
    # 1) Get binary version of pin mask:
    mask_pin_bin = np.full((np.shape(label_mask)), False)
    mask_pin_bin[np.where(pin_mask == [255, 0, 0])[:2]] = True
    # 2) Find intersection between pin mask and label mask:
    intersection = np.where((label_mask == True) & (mask_pin_bin == True))
    # 3) Round pixel colours:
    img_r = round_pixel_colours(image)
    # 4) Get strings of rounded colours in label mask:
    cols_round = [
        str(p) for p in img_r[np.where((label_mask == True) & (mask_pin_bin == False))]
    ]
    # 5) Count instances:
    count_ = Counter(cols_round)
    vals = list(count_.values())
    # 6) Find mode colour:
    m = np.int_(re.findall("\d+", np.array(list(count_.keys()))[np.argmax(vals)]))
    if sum(m) < 30:
        m = [255, 255, 255]
    # 7) Get new pixel colours:
    v = len(intersection[0])
    a = list(np.random.randint(m[0] - epsilon, m[0] + epsilon, size=(1, v))[0])
    b = list(np.random.randint(m[1] - epsilon, m[1] + epsilon, size=(1, v))[0])
    c = list(np.random.randint(m[2] - epsilon, m[2] + epsilon, size=(1, v))[0])
    new_pixel_colours = np.array([a, b, c]).T
    # 8) Update pixels:
    new_image = deepcopy(image)
    new_image[intersection] = new_pixel_colours
    return new_image
