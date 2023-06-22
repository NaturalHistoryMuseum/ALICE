import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from skimage import measure


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
    x = contours[0][:, 1]
    y = contours[0][:, 0]

    mnx = min(x)
    mxx = max(x)
    mny = min(y)
    mxy = max(y)

    mnx_ = np.argmin(x)
    mxx_ = np.argmax(x)
    mny_ = np.argmin(y)
    mxy_ = np.argmax(y)

    return contours, x, y, mnx, mxx, mny, mxy, mnx_, mxx_, mny_, mxy_


# POINT-DISTANCE FUNCTIONS #
############################


def dist_two_points(x1, y1, x2, y2):
    # Finds distance between two points in R2.
    d = np.sqrt((y2 - y1) ** 2 + (x2 - x2) ** 2)
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
