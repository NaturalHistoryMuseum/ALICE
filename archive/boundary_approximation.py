import numpy as np
from copy import deepcopy
from skimage import io, measure
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import active_contour
import cv2
import re
import statistics as stats


#####################
# CONTOUR FUNCTIONS #
#####################


def find_top_contours(img, k=20, modification=None):
    # Aim: Using a marching squares algorithm, find the most relevant contours
    # based on range (y axis) and number of points.
    # Input: a greyscaled image.

    # 1) Threshold image:
    img_gaussian = gaussian(img, 3)
    thresh = threshold_otsu(img_gaussian)
    binary = img_gaussian > thresh

    # 2) Find all contours:
    contours = measure.find_contours(binary, 0.8)

    # 3) Filter for "top" contours:
    # Find length (no. coordinates) of all contours:
    lens = [len(c[:, 0]) for c in contours]
    # Set lower bound for length:
    lb = np.floor(np.average(lens))
    # Create subset based on lower bound:
    inds = np.where(np.array(lens) >= lb)[0]
    # Find the y range of the contours:
    y_ranges = [np.round((max(c[:, 0]) - min(c[:, 0])), 4) for c in contours]
    # Filter for those with top k ranges:
    top_inds = np.argsort(np.array(y_ranges)[inds])[-k:]
    top_inds = inds[top_inds]

    top_contours = []
    n1, n2 = np.shape(img)
    n1 = np.round(n1, -1)
    n2 = np.round(n2, -1)

    for i in top_inds:
        c = contours[i]
        if modification == None:
            top_contours.append([c[:, 1], c[:, 0]])
        else:
            x = np.round(c[:, 1], -1)
            y = np.round(c[:, 0], -1)

            m = modification

            if sum([m in x, m in y, n2 - m in x, n1 - m in y]) < 3:
                top_contours.append([c[:, 1], c[:, 0]])

    return top_contours


def find_pin_contours(contours, perc=50):
    # Aim: Using the function find_top_contours, filter for the contours that
    # are assumed to protray segments of pins.

    pinch = []
    min_yvalues = []

    for xy in contours:
        x = xy[0]
        y = xy[1]
        p = (max(y) - min(y)) / (max(x) - min(x))
        p = np.round(p, 3)
        pinch.append(p)
        min_yvalues.append(max(y))  # Since the photos, using io, are upsite down.

    pin_inds = np.where(np.array(pinch) >= np.percentile(pinch, perc))[0]

    k = np.argmax([(max(xy[1]) - min(xy[1])) for xy in contours])

    pin_inds = np.unique(np.append(pin_inds, k))

    pin_contours = np.array(contours, dtype="object")[pin_inds]

    base_x = np.average([np.average(p_contour[0]) for p_contour in pin_contours])
    base_y = max(min_yvalues)

    return pin_contours, np.round([base_x, base_y], 4)


#####################
# ELLIPSE FUNCTIONS #
#####################


def get_ellipse(mpx, mpy, diamx, diamy, npoints=100):
    # Aim: Create an ellipse based on the diameters on the x,y axis and the midpoint.

    # 1) Denote radius.
    a = diamx / 2  # radius on the x-axis
    b = diamy / 2  # radius on the y-axis

    # 2) Define t.
    t = np.linspace(0, 2 * np.pi, npoints)

    # 3) Compute ellipse.
    x = mpx + (a * np.cos(t))
    y = mpy + (b * np.sin(t))

    return x, y


def in_ellipse(xpoint, ypoint, xc, yc, dx, dy):
    # Aim: Check whether a point lies within an ellipse.

    # Denote radius.
    rx = dx / 2
    ry = dy / 2

    # Compute ellipse interval.
    a = ((xpoint - xc) / rx) ** 2
    b = ((ypoint - yc) / ry) ** 2

    # Decide if point is in ellipse.
    inEllipse = False
    if a + b <= 1:
        inEllipse = True

    return inEllipse


##################################
# REGION APPROXIMATION FUNCTIONS #
##################################


def init_region_ellipse(pin_contours, base_point):
    # Aim: Create initial boundary ellipse

    mn = min([min(p[1]) for p in pin_contours])
    diamy = base_point[1] - mn
    diamx = max([max(p[0]) for p in pin_contours]) - min(
        [min(p[0]) for p in pin_contours]
    )
    mpy = mn + (diamy / 2)

    mpx = base_point[0]
    diamx = diamx * 1.2
    diamy = diamy * 1.2

    xe, ye = get_ellipse(mpx, mpy, diamx, diamy)

    return xe, ye, [mpx, mpy, diamx, diamy]


def approximate_region(pth, eps=1.15):
    # TEST FUCNTION #
    # Aim: Approximate the boundary region around a pinned specimen and labels.

    # 1) Find appropriate contours:
    img = io.imread(pth, as_gray=True)
    top_contours = find_top_contours(img)
    # 2) Approximate initial boundary ellipse:
    pin_contours, base_point = find_pin_contours(top_contours)
    init_x, init_y, ellipse_info = init_region_ellipse(pin_contours, base_point)
    # 3) Filter for contours who cross the initial boundary:
    in_boundary = [
        (
            np.array(
                [
                    in_ellipse(
                        c[0][i],
                        c[1][i],
                        ellipse_info[0],
                        ellipse_info[1],
                        ellipse_info[2],
                        ellipse_info[3],
                    )
                    for i in range(0, len(c[0]))
                ]
            ).any()
            == True
        )
        for c in top_contours
    ]
    filtered_contours = np.array(top_contours, dtype="object")[in_boundary]
    # 4) Compute diameters for new boundary ellipse:
    # Diameter in x-axis
    range_x = [[min(xy[0]), max(xy[0])] for xy in filtered_contours]
    xdiam = np.max(range_x) - np.min(range_x)
    # Diameter in y-axis
    range_y = [[min(xy[1]), max(xy[1])] for xy in filtered_contours]
    ydiam = np.max(range_y) - np.min(range_y)
    # 5) Fix ellipse centre:
    mpx = base_point[0]
    mn_y = min(
        [
            min(xy[1][np.where((xy[0] < max(init_x)) & (xy[0] > min(init_x)))[0]])
            for xy in filtered_contours
        ]
    )
    mpy = (mn_y + base_point[1]) / 2
    # 6) Create boundary ellipse:
    ellipse_x, ellipse_y = get_ellipse(mpx, mpy, xdiam * eps, ydiam * eps)
    # 7) Limit ellipse to image boundaries:
    boundary_x = [min(max(x, 5), np.shape(img)[1] - 5) for x in ellipse_x]
    boundary_y = [min(max(y, 5), np.shape(img)[0] - 5) for y in ellipse_y]

    return (
        filtered_contours,
        pin_contours,
        base_point,
        [ellipse_x, ellipse_y],
        [boundary_x, boundary_y],
    )


########################
# PIN REGION FUNCTIONS #
########################


def filter_pin_contours(pin_contours, ellipse_info, proportion=0.5):
    # Aim: filter possible pin contours to only include those who are within (or partly within) the boundary ellipse.

    in_boundary = [
        len(
            np.where(
                np.array(
                    [
                        in_ellipse(
                            c[0][i],
                            c[1][i],
                            ellipse_info[0],
                            ellipse_info[1],
                            ellipse_info[2],
                            ellipse_info[3],
                        )
                        for i in range(0, len(c[0]))
                    ]
                )
                == True
            )[0]
        )
        / len(c[0])
        for c in pin_contours
    ]
    filtered_pin_contours = np.array(pin_contours, dtype="object")[
        np.where(np.array(in_boundary) > proportion)[0]
    ]

    top_pin_ind = np.argmax(
        [
            (max(xy[1]) - min(xy[1])) / (max(xy[0]) - min(xy[0]))
            for xy in filtered_pin_contours
        ]
    )

    return filtered_pin_contours, top_pin_ind


def pin_line(img, filtered_pin_contours, pin_k_inds, fill_colour=[255, 0, 0], nseg=30):
    # OLD FUNCTION #

    filled_img = deepcopy(img)
    for pin_k in pin_k_inds:
        x = filtered_pin_contours[pin_k][0]
        y = filtered_pin_contours[pin_k][1]
        a3 = np.array([np.array([x, y]).T], dtype=np.int32)
        cv2.fillPoly(filled_img, a3, fill_colour)

    inds_ = np.where(filled_img == fill_colour)[0:2]

    ##########################################################

    segments = np.int_(np.linspace(0, np.shape(img)[1], nseg + 1))

    count_seg = [
        len(np.where(filled_img[:, segments[i - 1] : segments[i]] == fill_colour)[0])
        for i in range(1, len(segments))
    ]

    a, b = [segments[np.argmax(count_seg)], segments[np.argmax(count_seg) + 1]]

    ###########################################################

    region_cols1 = img[inds_]

    region_cols2 = img[
        inds_[0][np.where((inds_[1] <= b) & (inds_[1] >= a))[0]],
        inds_[1][np.where((inds_[1] <= b) & (inds_[1] >= a))[0]],
    ]

    ############################################################

    lines = []

    for region_cols in [region_cols1, region_cols2]:

        b = [
            np.average(img[np.shape(img)[0] - 10 :, 0:10, 0]),
            np.average(img[np.shape(img)[0] - 10 :, 0:10, 1]),
            np.average(img[np.shape(img)[0] - 10 :, 0:10, 2]),
        ]
        tst_2 = [
            str(pixel_col_round(p))
            for p in region_cols
            if pixel_col_round(p) != pixel_col_round(b)
        ]

        a = stats.mode(tst_2)
        bla = re.findall("\d+", a)
        bla_img = convert_img(img, np.int_(bla))

        count = [
            len(np.where(bla_img[:, i] == fill_colour)[0])
            for i in range(0, np.shape(bla_img)[1])
        ]

        p = np.argmax(count)

        lines.append([p, np.shape(img)[0] - 1])

    return lines, filled_img


###################
# MISC FUNCTIONS #
##################


def pixel_col_round(pixels, base=5):
    # Aim: round colours to the nearest multiple of your choosing (default 5).
    # This function is primarily used to round RGB values.

    rounded_colours = []
    for x in pixels:
        rounded_colours.append(int(base * round(float(x) / base)))
    return rounded_colours


def removeReps(T, F):
    # Aim: remove repeated (x,y) coordinates from a function.

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


def convert_img(img, colour, new_col=[255, 0, 0]):
    # Aim: Convert

    part1 = img[:, :, 0]
    part2 = img[:, :, 1]
    part3 = img[:, :, 2]

    v = np.where(
        ((part1 >= colour[0] - 5) & (part1 <= colour[0] + 5))
        & ((part2 >= colour[1] - 5) & (part2 <= colour[1] + 5))
        & ((part3 >= colour[2] - 5) & (part3 <= colour[2] + 5))
    )

    img2 = deepcopy(img)
    img2[v] = new_col
    return img2
