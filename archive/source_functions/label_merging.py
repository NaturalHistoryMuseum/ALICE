import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
from skimage import measure
from copy import deepcopy
import cv2
import re
from imutils.perspective import four_point_transform
import pytesseract

try:
    from source_functions.alignment_helper_functions import *
    from source_functions.label_transformation import *
except:
    from alignment_helper_functions import *
    from label_transformation import *


"""
A collection of functions used to merge labels together in order to create
one image of the specimen's label. Steps include:
1. Aligning (registering) labels with a template.
2. Label merging.      
"""


# ----------------------------
# Template Selection Functions
# ----------------------------

"""
Note: before aligning images, a template needs to be selected.
The following functions can be helpful in this selection process.
"""


def bin_image(image_orig, bound_percentile=50):
    # Aim: threshold image based on median colour.

    # Input: image.
    # Output: binarized image.

    image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
    m = np.percentile(image.flatten(), bound_percentile)
    image2 = deepcopy(image)
    image2[np.where(image > m)] = 255
    image2[np.where(image < m)] = 0
    return image2


def max_tesseract_osd_count(image, values=np.int_(np.linspace(3, 10, 8))):
    # Aim: rotate image if required and find the max min characters found with tesseract.

    # Input: image.
    # Output: image (correctly orientated), max min characters, binarized image.

    """
    Note that this function is used as a helper function for select_template, in order to find
    a template a label for the alignment stage.
    Also note that the current usage of the tesseract function here is inefficient, thus it
    requires changing in the future. For now, we keep as is.
    """

    img_bin = bin_image(image)

    max_total = min(values)
    for j in values:
        try:
            osd = pytesseract.image_to_osd(
                img_bin,
                config="--psm 0 -c min_characters_to_try=" + str(j) + " script=Latin",
            )
            max_total = j
        except:
            pass

    max_ = 0
    if max_total >= max(values):
        max_ = 1

    try:
        rotation = np.int_(re.findall("\d+", osd)[1:3])
        k1 = re.search("Orientation confidence: ", osd).span()[1]
        k2 = re.search("Script", osd).span()[0]
        prct = float(osd[k1 : k2 - 1])

        if (rotation[0] == 180) and (prct > 0.1):
            image = imutils.rotate_bound(image, rotation[1])
            img_bin = imutils.rotate_bound(img_bin, rotation[1])
    except:
        pass

    return max_total, max_, image, img_bin


def select_template_oldMethod(all_transformed_images):
    # Aim: select a template image for the alignment stage.

    # Input: list of images (ALICE images from all angles).
    # Output: a template image, and the correctly orientated list of images.

    """
    OLD METHOD!

    Note that the correctly orientated list of images could be the same as the original list
    if no correction was needed (using max_tesseract_osd_count)
    """

    total_max_vals = []
    angles = []
    all_imgs = []
    for img in all_transformed_images:
        max_total, max_, image, img_bin = max_tesseract_osd_count(img)
        total_max_vals.append(max_total)
        theta = adjust_alignment(image, np.linspace(-10, 10, 21))[0]
        angles.append(theta)
        all_imgs.append(image)

    inds = np.where(np.array(total_max_vals) == max(total_max_vals))[0]
    k = inds[np.argmin(abs(np.array(angles)[inds]))]

    template_label = all_imgs[k]

    return template_label, all_imgs


def select_template(all_transformed_images, min_letters=3):
    # Aim: select a template image for the alignment stage.

    # Input: list of images (ALICE images from all angles).
    # Output: a template image, and the correctly orientated list of images.

    """
    Note 1: that the correctly orientated list of images could be the same as the original list
    if no correction was needed (using max_tesseract_osd_count)
    Note 2: this is an updated version of the original select_tempalte. In this version, the
    template chosen is based on the image with the most letters found using basic OCR.
    """

    angles = []
    all_imgs = []
    total_letters_found = []
    for img in all_transformed_images:
        _, _, image, img_bin = max_tesseract_osd_count(img)

        ocr_results = pytesseract.image_to_string(
            img_bin, config="--psm 11 script=Latin"
        )

        total_letters_found.append(len(ocr_results))
        theta = adjust_alignment(image, np.linspace(-10, 10, 21))[0]
        angles.append(theta)
        all_imgs.append(image)

    letters_lower_bound = np.sort(total_letters_found)[-2]
    if letters_lower_bound <= min_letters:
        letters_lower_bound = np.max(total_letters_found)
    inds = np.where(np.array(total_letters_found) >= letters_lower_bound)[0]
    k = inds[np.argmin(abs(np.array(angles)[inds]))]

    template_label = all_imgs[k]

    return template_label, all_imgs


# -------------------
# Alignment Functions
# -------------------


def align_image(image, template, matches_bound=0.9):
    # Aim: align an image with a template.

    # Input: image / template.
    # Ouput: aligned image.

    img1_color = deepcopy(image)  # Image to be aligned.
    img2_color = deepcopy(template)  # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    dists = [m.distance for m in matches]
    lst = [matches[i] for i in np.argsort(dists)]
    matches = tuple(lst)

    # Take the top 90 % matches forward.
    matches = matches[: int(len(matches) * matches_bound)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color, homography, (width, height))

    return transformed_img


def align_merged_label_original(merged):  # ORIGINAL FUNCTION BEFORE EDITS!
    # Aim: correct the alignment of a (merged) label.

    # Input: image to align.
    # Output: corrected image.

    # Binarize image
    if len(np.shape(merged)) == 3:
        image = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    else:
        image = deepcopy(merged)
    img_bin = deepcopy(image)
    img_bin[np.where(image != 0)] = 1
    img_bin[:, 0] = 0
    img_bin[:, -1] = 0
    img_bin[0, :] = 0
    img_bin[-1, :] = 0

    # Find contours and sort for largest contour
    cnts = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    for c in cnts:
        # Perform contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            break

    if displayCnt is None:
        displayCnt = new_corners(img_bin)

    # Obtain birds' eye view of image
    warped = four_point_transform(merged, displayCnt.reshape(4, 2))

    return warped


def new_corners(img_bin):
    contours = measure.find_contours(img_bin, 0.8)
    c = contours[np.argmax([len(c[:, 0]) for c in contours])]
    x = c[:, 1]
    y = c[:, 0]
    corners = np.zeros((4, 1, 2), dtype="int32")
    corners[0, :, :] = np.array([min(x), min(y)], dtype="int32")
    corners[1, :, :] = np.array([min(x), max(y)], dtype="int32")
    corners[2, :, :] = np.array([max(x), max(y)], dtype="int32")
    corners[3, :, :] = np.array([max(x), min(y)], dtype="int32")
    return corners


def align_merged_label(merged):
    # Aim: correct the alignment of a (merged) label.

    # Input: image to align.
    # Output: corrected image.

    # Binarize image

    if len(np.shape(merged)) == 3:
        image = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    else:
        image = deepcopy(merged)
    img_bin = deepcopy(image)
    img_bin[np.where(image != 0)] = 1
    img_bin[:, 0] = 0
    img_bin[:, -1] = 0
    img_bin[0, :] = 0
    img_bin[-1, :] = 0

    # Find contours and sort for largest contour
    cnts = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    for c in cnts:
        # Perform contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            break

    if displayCnt is None:
        displayCnt = new_corners(img_bin)

    ratio = (np.max(displayCnt[:, :, 0]) - np.min(displayCnt[:, :, 0])) / np.shape(
        img_bin
    )[1]

    if ratio < 0.25:
        img_bin_new = deepcopy(img_bin)
        if len(cnts) > 1:
            c = cnts[
                -1
            ]  # since it's sorted in area size and we want the smallest area.
            c2 = np.reshape(c, (1, np.shape(c)[0], 2))
            img_bin_new = cv2.fillPoly(img_bin, c2, [0, 0, 0])
        displayCnt = new_corners(img_bin_new)

    # Obtain birds' eye view of image
    warped = four_point_transform(merged, displayCnt.reshape(4, 2))

    return warped


# ----------------
# Merge Functions
# ----------------


def merge_label(all_labels, method=0):
    # Aim: merge multiple labels together.

    # Input: label images to merge.
    # Output: merged label.

    """
    Method 0 - original method by Ginger (https://github.com/NaturalHistoryMuseum/ALICE/blob/master/ALICE/models/viewsets.py#L38-L49)
    Method 1 - alternative method (darkest values per pixel)
    """

    if method == 0:
        I = np.median(np.stack([v for v in all_labels]), axis=0)
        merged_label = np.array(I, dtype="uint8")
    else:
        bw_images = []
        bw_images_mx = []
        for img in all_labels:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bw_images.append(image)
            image2 = deepcopy(image)
            image2[np.where(image == 0)] = 254
            bw_images_mx.append(image2)

        merged_label = np.min(bw_images_mx, 0)

    return merged_label


# ----------------------
# Other Helper Functions
# ----------------------


def check_ocr(img, backup_bound_percentile=60):
    # Aim: basic OCR to see if transformed image has any text.

    # Input: transformed label.
    # Output: OCR text result.

    bI = bin_image(img)
    ocr_results = pytesseract.image_to_string(bI, config="--psm 11 script=Latin")
    ocr_res = " ".join(re.findall("\w+", ocr_results))
    if len(ocr_res) == 0:
        bI = bin_image(img, bound_percentile=backup_bound_percentile)
        ocr_results = pytesseract.image_to_string(bI, config="--psm 11 script=Latin")
        ocr_res = " ".join(re.findall("\w+", ocr_results))

    return ocr_res
