import numpy as np
import skimage.io as io
from copy import deepcopy
import cv2

try:
    from source_functions.alignment_helper_functions import *
except:
    from alignment_helper_functions import *

"""
A collection of functions used to transform a single label in an image
to a segmented label in a 2d perspective. Steps include:
1. Finding the corners of the label in the original image.
2. Perspective transformation.      
"""

# ----------------
# Corner Functions
# -----------------


def find_corners(
    mask_pth, img_pth, ydist=True, combine_extra_masks=False, paths=True, label_method=1
):
    # Aim: Find four corners of label.

    # Input: path to resulting mask from CNN instance segmentation / path to original image.
    # Output: x and y coordinates of the four corners.

    # 1) Load mask:
    if paths == True:
        mask = np.load(mask_pth)
    else:
        mask = deepcopy(mask_pth)
    # (1.2) Check how many contours are found in mask:
    # If multiple, then we can combine masks:
    if combine_extra_masks == True:
        contours = measure.find_contours(mask, 0.8)
        if len(contours) > 1:
            mask = combine_masks(mask, contours)
    # 2) Load original image:
    if paths == True:
        img_orig = io.imread(img_pth)
    else:
        img_orig = deepcopy(img_pth)
    # 3) Find Direction
    new_lines, _, swapped, contours = find_direction(mask)
    # 4) Find Corners
    if label_method == 0:
        corners_x, corners_y = find_label_corners(
            new_lines, contours, swapped, ydistance=ydist
        )
    else:
        corners_x, corners_y = find_label_corners_v2(new_lines, contours)

    return mask, img_orig, corners_x, corners_y, contours


def define_label_sides(corners_x, corners_y):
    # Aim: Locate the rectangle's (label's) "short side" and "long side".

    # Input: coordinates of corners.
    # Ouput: index of short / long sides and dictionary of distances between corners.

    # 1) Find distances between corners:
    dists = np.zeros((6, 3))

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

    # 2) Convert distance array to dictionary:
    dists_dict = {}
    for p in dists:
        dists_dict[tuple(p[:2])] = p[2]

    # 3) Find short side and long side:
    all_pairs = [[0, 1], [1, 2], [2, 3], [0, 3]]
    pairs_dists = [dists_dict[(p[0], p[1])] for p in all_pairs]
    dist_index = np.argsort(pairs_dists)

    short_inds = [all_pairs[dist_index[0]]]
    short = [p for p in all_pairs if any(item in p for item in short_inds[0]) == False][
        0
    ]
    short_inds.append(short)

    long_inds = [
        p
        for p in all_pairs
        if (any(item in p for item in short_inds[0]) == True) and (p != short_inds[0])
    ]

    return short_inds, long_inds, pairs_dists, dists_dict


def reconfigure_corner(
    short_inds, long_inds, dists_dict, corners_x, corners_y, method=0
):
    # Aim: Update corners of parallelogram.

    # Input: corners / dictionary of distances between corners.
    # Ouput: updated corners.

    u = np.argmax([dists_dict[(p[0], p[1])] for p in short_inds])
    v = np.argmax([dists_dict[(p[0], p[1])] for p in long_inds])

    if method == 1:  # Altnerative method in choosing the template "long side".
        q = short_inds[u]
        v = np.argmin(
            [
                abs(
                    90
                    - get_angle(
                        [corners_y[p[0]], corners_y[p[1]]],
                        [corners_x[p[0]], corners_x[p[1]]],
                        [corners_y[q[0]], corners_y[q[1]]],
                        [corners_x[q[0]], corners_x[q[1]]],
                    )
                )
                for p in long_inds
            ]
        )

    inds = np.array([short_inds[u], long_inds[v]]).flatten()

    missing_corner_ind = [v for v in range(0, 4) if v not in inds][0]

    repeated_corner_ind = [v for v in inds if len(np.where(inds == v)[0]) == 2][0]

    point_ind = inds[np.where(inds != repeated_corner_ind)[0]][0]
    point_to_change_ind = inds[np.where(inds != repeated_corner_ind)[0]][-1]

    y_incrt = abs(corners_y[repeated_corner_ind] - corners_y[point_ind])
    x_incrt = abs(corners_x[repeated_corner_ind] - corners_x[point_ind])

    if (corners_x[missing_corner_ind] > corners_x[point_to_change_ind]) == False:
        x_incrt = x_incrt * -1

    if (corners_y[missing_corner_ind] > corners_y[point_to_change_ind]) == False:
        y_incrt = y_incrt * -1

    new_corner_x = corners_x[point_to_change_ind] + x_incrt
    new_corner_y = corners_y[point_to_change_ind] + y_incrt

    corners_x_updated = deepcopy(corners_x)
    corners_y_updated = deepcopy(corners_y)

    corners_x_updated[missing_corner_ind] = new_corner_x
    corners_y_updated[missing_corner_ind] = new_corner_y

    return corners_x_updated, corners_y_updated


def reconfigure_corner_global(
    short_inds,
    long_inds,
    dists_dict,
    corners_x,
    corners_y,
    method=0,
    original_mask=None,
    original_image=None,
):
    if method != "both":
        corners_x_updated, corners_y_updated = reconfigure_corner(
            short_inds, long_inds, dists_dict, corners_x, corners_y, method=method
        )
    else:
        corners_1 = reconfigure_corner(
            short_inds, long_inds, dists_dict, corners_x, corners_y, method=0
        )
        corners_2 = reconfigure_corner(
            short_inds, long_inds, dists_dict, corners_x, corners_y, method=1
        )
        best_index, _ = compare_corners_with_mask(
            corners_1, corners_2, original_mask, original_image
        )
        corners_x_updated, corners_y_updated = [corners_1, corners_2][best_index]
    return corners_x_updated, corners_y_updated


def backup_corner_method(contours, npoints=80, return_dict=False):
    # Aim: Back-up method of finding corners.

    # Input: contour around label mask.
    # Ouput: corners of label.

    contour_x, contour_y = reparam(contours[0], contours[1], npoints)

    tst = np.zeros((len(contour_x), 1, 2), dtype="int32")
    tst[:, 0, 0] = np.int_(contour_y)
    tst[:, 0, 1] = np.int_(contour_x)

    peri = cv2.arcLength(tst, True)
    approx = cv2.approxPolyDP(tst, 0.02 * peri, True)

    new_approx = deepcopy(approx)
    if len(approx) != 4:
        inds = []
        a = np.argmin(approx[:, 0, 1])
        inds.append(a)
        b = [item for item in np.argsort(approx[:, 0, 1]) if item not in inds][-1]
        inds.append(b)
        c = [item for item in np.argsort(approx[:, 0, 0]) if item not in inds][0]
        inds.append(c)
        d = [item for item in np.argsort(approx[:, 0, 0]) if item not in inds][-1]
        new_approx = approx[np.sort([a, b, c, d]), :, :]

    corners_x = list(new_approx[:, 0, 1])
    corners_y = list(new_approx[:, 0, 0])
    if return_dict == False:
        return corners_x, corners_y
    else:
        return corners_x, corners_y, new_approx, approx


# --------------------------
# Corner-Checking Functions
# --------------------------


def check_angles(corners_x2, corners_y2, short_inds, long_inds, min_angle=5):
    # Aim: Check if the sides have appropriate angles for a label paralleogram,
    # where appropriateness is based on the parameter min_angle

    # Input: corners of labels and index of short / long sides.
    # Ouput: binary - based on check that any angle is less than min_angle.

    a1 = get_angle(
        np.array(corners_x2)[short_inds[0]],
        np.array(corners_y2)[short_inds[0]],
        np.array(corners_x2)[long_inds[0]],
        np.array(corners_y2)[long_inds[0]],
    )

    a2 = get_angle(
        np.array(corners_x2)[short_inds[1]],
        np.array(corners_y2)[short_inds[1]],
        np.array(corners_x2)[long_inds[0]],
        np.array(corners_y2)[long_inds[0]],
    )

    a3 = get_angle(
        np.array(corners_x2)[short_inds[1]],
        np.array(corners_y2)[short_inds[1]],
        np.array(corners_x2)[long_inds[1]],
        np.array(corners_y2)[long_inds[1]],
    )

    a4 = get_angle(
        np.array(corners_x2)[short_inds[0]],
        np.array(corners_y2)[short_inds[0]],
        np.array(corners_x2)[long_inds[1]],
        np.array(corners_y2)[long_inds[1]],
    )

    A = any(angle < min_angle for angle in [a1, a2, a3, a4])

    return A


def intersection_check(corners_x2, corners_y2, long_inds):
    # Aim: Check if the long sides of label intersect.

    # Input: corners of labels and index long sides.
    # Ouput: binary - based on intersection check.

    A1 = {"x": corners_x2[long_inds[0][0]], "y": corners_y2[long_inds[0][0]]}
    B1 = {"x": corners_x2[long_inds[0][1]], "y": corners_y2[long_inds[0][1]]}
    C1 = {"x": corners_x2[long_inds[1][0]], "y": corners_y2[long_inds[1][0]]}
    D1 = {"x": corners_x2[long_inds[1][1]], "y": corners_y2[long_inds[1][1]]}

    return intersect(A1, B1, C1, D1)


def check_corners(corners_x2, corners_y2, short_inds, long_inds, min_angle=5):
    # Aim: Check appropriateness of label corners.

    # Input: corners of labels and index of short / long sides.
    # Ouput: binary - based on check.

    """
    We base our appropriateness criteria on the angles between the sides
    and instances of line intersections. If there is an angle in the label
    that we deem too small (based on min_angle), or if the two long sides
    of the parallelogram intersect, we redo the corner finding step but with
    a different function than previous.
    """

    check1 = check_angles(
        corners_x2, corners_y2, short_inds, long_inds, min_angle=min_angle
    )
    check2 = intersection_check(corners_x2, corners_y2, long_inds)

    return any(i == True for i in [check1, check2])


def compare_corners(
    img_orig, corners_x, corners_y, corners_x_updated, corners_y_updated, min_bound=0.1
):
    # Aim: Compare corners_x/y with corners_x/y_updated in order to decide which to use
    # for the succeeding transformation step. This is based on the percentage increase of
    # the mask that the updated corners cover.

    # Input: original corners. updated corners, and original image.
    # Output: the chosen corners.

    tst1 = np.zeros((1, 5, 2))
    tst1[0, :4, 0] = corners_x
    tst1[0, :4, 1] = corners_y
    tst1[0, 4, 0] = corners_x[0]
    tst1[0, 4, 1] = corners_y[0]

    tst2 = np.zeros((1, 5, 2))
    tst2[0, :4, 0] = corners_x_updated
    tst2[0, :4, 1] = corners_y_updated
    tst2[0, 4, 0] = corners_x_updated[0]
    tst2[0, 4, 1] = corners_y_updated[0]

    img_filled1 = deepcopy(img_orig)
    img_filled1 = cv2.fillPoly(img_filled1, tst1.astype(np.int32), [255, 0, 0])
    img_filled2 = deepcopy(img_orig)
    img_filled2 = cv2.fillPoly(img_filled2, tst2.astype(np.int32), [0, 255, 0])

    a = len(np.where(img_filled1 == [255, 0, 0])[0])
    b = len(np.where(img_filled2 == [0, 255, 0])[0])

    l = np.round((b - a) / b, 3)

    if l > min_bound:
        return corners_x_updated, corners_y_updated
    else:
        return corners_x, corners_y


def compare_corners_with_mask(corners1, corners2, original_mask, img_orig):
    # Aim: Compare corners_x/y with corners_x/y_updated in order to decide which to use
    # for the succeeding transformation step. This is based on the percentage increase of
    # the mask that the updated corners cover.

    # Input: two sets of possible label corners, original mask and image.
    # Output: the total intersection of the new corners and the original mask, and the index of
    # the set of corners with largest intersection.

    # Create filled images with corners:
    a3 = np.array([np.array([corners1[0], corners1[1]]).T], dtype=np.int32)
    I = deepcopy(img_orig)
    img_filled1 = cv2.fillPoly(I, a3, [255, 0, 0])
    a3 = np.array([np.array([corners2[0], corners2[1]]).T], dtype=np.int32)
    I = deepcopy(img_orig)
    img_filled2 = cv2.fillPoly(I, a3, [255, 0, 0])
    # Get binary masks from filled images:
    mask1 = np.full(np.shape(original_mask), False)
    mask1[np.where(img_filled1 == [255, 0, 0])[:2]] = True
    mask2 = np.full(np.shape(original_mask), False)
    mask2[np.where(img_filled2 == [255, 0, 0])[:2]] = True
    # Compute intersections:
    I1 = len(np.where((original_mask == True) & (mask1 == True))[0])
    I2 = len(np.where((original_mask == True) & (mask2 == True))[0])

    return np.argmax([I1, I2]), [I1, I2]


# --------------------------
# Perspective Transformation
# --------------------------


def perspective_transform(
    corners_x_updated,
    corners_y_updated,
    img_orig,
    dists_dict,
    short_inds,
    long_inds,
    box_epsilon=1.1,
    fixed_dim=False,
    dimension=(0, 0),
    return_box=False,
    corner_index_method=1,
):
    # Aim: apply a perspective transformation to warp the label into a 2d viewpoint.

    # Input: corners / dictionary of distances between corners.
    # Output: segmented image of warped label.

    mx_long = max(
        [
            dists_dict[(long_inds[0][0], long_inds[0][1])],
            dists_dict[(long_inds[1][0], long_inds[1][1])],
        ]
    )
    mx_short = max(
        [
            dists_dict[(short_inds[0][0], short_inds[0][1])],
            dists_dict[(short_inds[1][0], short_inds[1][1])],
        ]
    )

    # STEP 2: Create desired box.
    #############################

    box_x = [0, 0, mx_long, mx_long, 0]
    box_y = [0, mx_short, mx_short, 0, 0]

    # STEP 3: Match corners.
    ########################

    # Match corners of desired box with current box:
    corner_inds = {}

    if corner_index_method == 0:
        u, v = [short_inds[i][0] for i in range(0, 2)]
        p = np.argmin([corners_x_updated[u], corners_x_updated[v]])
        u, v = short_inds[p]

        i, j = np.argsort([corners_y_updated[u], corners_y_updated[v]])
        k = [u, v][i]
        corner_inds["bot_left"] = [corners_x_updated[k], corners_y_updated[k]]
        k = [u, v][j]
        corner_inds["top_left"] = [corners_x_updated[k], corners_y_updated[k]]

        p = (p + 1) % 2
        u, v = short_inds[p]

        i, j = np.argsort([corners_y_updated[u], corners_y_updated[v]])
        k = [u, v][i]
        corner_inds["bot_right"] = [corners_x_updated[k], corners_y_updated[k]]
        k = [u, v][j]
        corner_inds["top_right"] = [corners_x_updated[k], corners_y_updated[k]]
    elif corner_index_method == 1:
        b = np.argsort(
            [
                np.average(np.array(corners_y_updated)[short_inds[i]])
                for i in range(0, 2)
            ]
        )
        c = np.argsort(np.array(corners_x_updated)[short_inds[b[0]]])
        top_left = short_inds[b[0]][c[0]]
        bot_left = short_inds[b[0]][c[1]]
        d = np.argsort(np.array(corners_x_updated)[short_inds[b[1]]])
        top_right = short_inds[b[1]][d[0]]
        bot_right = short_inds[b[1]][d[1]]

        corner_inds["top_left"] = [
            corners_x_updated[top_left],
            corners_y_updated[top_left],
        ]
        corner_inds["bot_left"] = [
            corners_x_updated[bot_left],
            corners_y_updated[bot_left],
        ]
        corner_inds["top_right"] = [
            corners_x_updated[top_right],
            corners_y_updated[top_right],
        ]
        corner_inds["bot_right"] = [
            corners_x_updated[bot_right],
            corners_y_updated[bot_right],
        ]

    order = ["bot_left", "top_left", "top_right", "bot_right"]
    pts1 = []
    pts2 = []

    mx = (max(corners_x_updated) + min(corners_x_updated)) / 2
    my = (max(corners_y_updated) + min(corners_y_updated)) / 2

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

    img_filled = deepcopy(img_orig)
    a3 = np.array([np.array([pts1[:, 0], pts1[:, 1]]).T], dtype=np.int32)

    # Fill background (exterior of label contour) in one colour (fill_col)
    fill_col = [255, 0, 0]
    img_filled = cv2.fillPoly(img_filled, a3, fill_col)

    img_s = deepcopy(img_orig)
    img_s[np.where(img_filled != [255, 0, 0])[:2]] = 0

    M = cv2.getPerspectiveTransform(pts1, pts2)
    # Slightly adjust boundary:

    if fixed_dim == False:
        dim = (int(max(pts2[:, 0])), int(max(pts2[:, 1])))
    else:
        dim = dimension

    img_warped = cv2.warpPerspective(img_s, M, dim)

    if return_box == True:
        return img_warped, img_s, pts1, pts2
    else:
        return img_warped, img_s
